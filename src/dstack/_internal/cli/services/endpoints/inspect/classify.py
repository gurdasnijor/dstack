"""Evidence-ladder classification of a model into ranked runtime candidates.

The ladder, from strongest to weakest evidence (see the endpoint handoff):

1. ``exact_recipe``      — a versioned recipe validated for this model/variant.
2. ``support_registry``  — the architecture/pipeline is in the deployed
                           runtime's pinned support registry.
3. ``generic_adapter``   — a documented generic adapter contract applies; a
                           smoke test is mandatory before trusting it.
4. ``discovery_hint``    — only a Hugging Face app/library/tag or model-card
                           claim supports the pairing; never proof.
5. ``research``          — metadata is missing or contradictory; the agent must
                           research before any runtime claim.

Classification is metadata-only and never a substitute for a real smoke test.
"""

from __future__ import annotations

import enum
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional

from dstack._internal.cli.services.endpoints.inspect import registry as registries
from dstack._internal.cli.services.endpoints.inspect.runtime import (
    resolve_runtime,
    select_capability_path,
)
from dstack._internal.cli.services.endpoints.inspect.sizing import (
    build_frontier,
    resolve_workload,
)

INSPECT_VERSION = 1

MODALITY_LANES: Mapping[str, Optional[str]] = {
    "text": "general-llm",
    "vlm": "general-vlm",
    "image": "general-image",
    "video": "general-video",
    "gguf": "general-gguf",
    "research": None,
}


class EvidenceLevel(str, enum.Enum):
    EXACT_RECIPE = "exact_recipe"
    SUPPORT_REGISTRY = "support_registry"
    GENERIC_ADAPTER = "generic_adapter"
    DISCOVERY_HINT = "discovery_hint"
    RESEARCH = "research"


_LEVEL_RANK = {
    EvidenceLevel.EXACT_RECIPE: 0,
    EvidenceLevel.SUPPORT_REGISTRY: 1,
    EvidenceLevel.GENERIC_ADAPTER: 2,
    EvidenceLevel.DISCOVERY_HINT: 3,
    EvidenceLevel.RESEARCH: 4,
}

_LEVEL_CONFIDENCE = {
    EvidenceLevel.EXACT_RECIPE: "high",
    EvidenceLevel.SUPPORT_REGISTRY: "high",
    EvidenceLevel.GENERIC_ADAPTER: "medium",
    EvidenceLevel.DISCOVERY_HINT: "low",
    EvidenceLevel.RESEARCH: "low",
}

_SMOKE_TESTS: Mapping[str, Mapping[str, str]] = {
    "text": {
        "api": "chat_completions",
        "request": (
            "one streaming chat completion plus one request with a harmless "
            "function tool and tool_choice:auto"
        ),
        "validates": "token content and tool-call parsing, not only HTTP 200",
    },
    "vlm": {
        "api": "chat_completions",
        "request": "one chat completion that includes at least one image input",
        "validates": "the model consumed the image and produced grounded tokens",
    },
    "gguf": {
        "api": "chat_completions",
        "request": "one streaming chat completion through the llama-server API",
        "validates": "token content from the selected GGUF quantization",
    },
    "image": {
        "api": "images_generations",
        "request": "one 1024x1024 generation with response_format=b64_json",
        "validates": "decodable image bytes with the requested dimensions",
    },
    "video": {
        "api": "video_generation",
        "request": "one representative text/image-to-video request at the declared workload",
        "validates": "a decodable video with the expected duration and frame count",
    },
    "research": {
        "api": "model_native",
        "request": "the repository's own representative example for the chosen path",
        "validates": "the documented model-native output, decoded and checked",
    },
}


@dataclass(frozen=True)
class Evidence:
    fact: str
    source: str

    def to_data(self) -> dict[str, str]:
        return {"fact": self.fact, "source": self.source}


@dataclass(frozen=True)
class RuntimeCandidate:
    runtime: str
    evidence_level: EvidenceLevel
    confidence: str
    evidence: tuple[Evidence, ...]
    missing_facts: tuple[str, ...] = ()
    launch_hints: tuple[str, ...] = ()
    registry_version: Optional[str] = None
    min_compute_capability: Optional[float] = None
    smoke_test: Mapping[str, str] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def to_data(self) -> dict[str, Any]:
        return {
            "runtime": self.runtime,
            "evidence_level": self.evidence_level.value,
            "confidence": self.confidence,
            "evidence": [item.to_data() for item in self.evidence],
            "missing_facts": list(self.missing_facts),
            "launch_hints": list(self.launch_hints),
            "registry_version": self.registry_version,
            "min_compute_capability": self.min_compute_capability,
            "smoke_test": dict(self.smoke_test),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class NegativeResult:
    runtime: str
    reason: str
    evidence: tuple[Evidence, ...] = ()

    def to_data(self) -> dict[str, Any]:
        return {
            "runtime": self.runtime,
            "reason": self.reason,
            "evidence": [item.to_data() for item in self.evidence],
        }


@dataclass(frozen=True)
class ModelInspection:
    model: str
    requested_revision: Optional[str]
    revision: str
    modality: str
    library: str
    pipeline_class: Optional[str]
    classification: str  # "supported" | "needs_research" | "unsupported"
    candidates: tuple[RuntimeCandidate, ...]
    negative_results: tuple[NegativeResult, ...]
    missing_facts: tuple[str, ...]
    notes: tuple[str, ...]
    hardware_hint: Optional[Mapping[str, Any]]
    ir: Mapping[str, Any]

    def to_data(self) -> dict[str, Any]:
        return {
            "inspect_version": INSPECT_VERSION,
            "model": self.model,
            "requested_revision": self.requested_revision,
            "revision": self.revision,
            "modality": self.modality,
            "library": self.library,
            "pipeline_class": self.pipeline_class,
            "classification": self.classification,
            "candidates": [candidate.to_data() for candidate in self.candidates],
            "negative_results": [negative.to_data() for negative in self.negative_results],
            "missing_facts": list(self.missing_facts),
            "notes": list(self.notes),
            "hardware_hint": dict(self.hardware_hint) if self.hardware_hint else None,
            "ir": dict(self.ir),
        }


def _candidate_order(candidate: RuntimeCandidate) -> tuple[int, str]:
    return (_LEVEL_RANK[candidate.evidence_level], candidate.runtime)


def _smoke_test(modality: str) -> Mapping[str, str]:
    return dict(_SMOKE_TESTS.get(modality, _SMOKE_TESTS["research"]))


# Synthetic lane-default templates used only to evaluate capability paths for
# a specific candidate runtime.
_ANNOTATION_TEMPLATES: Mapping[str, Mapping[str, Any]] = {
    "sglang": {"image": "sglang:lane-default", "commands": ["sglang serve model"]},
    "diffusers": {"image": "diffusers:lane-default", "commands": ["python serve.py"]},
}


def _capability_annotation(
    ir: Mapping[str, Any], runtime_name: str, lane: Optional[str]
) -> tuple[Optional[float], tuple[str, ...], tuple[str, ...]]:
    """Annotate one candidate runtime with its capability-path decision."""
    template = _ANNOTATION_TEMPLATES.get(runtime_name)
    if template is None and lane is None:
        return None, (), ()
    runtime = resolve_runtime(template, lane or "research")
    if runtime.engine != runtime_name:
        return None, (), ()
    decision = select_capability_path(runtime, ir)
    if decision.selected is None:
        quant = (ir.get("quantization") or {}).get("method")
        missing = (
            (
                f"quantization method {quant!r} has no capability path recorded for "
                f"{runtime_name}; verify support on a pinned image before planning"
            ),
        )
        return decision.effective_min_cc, missing, ()
    notes = (decision.selected.note,) if decision.selected.note else ()
    return decision.effective_min_cc, (), notes


def _hardware_hint(ir: Mapping[str, Any], lane: Optional[str]) -> Optional[dict[str, Any]]:
    """Offer-independent hardware floor from the deterministic frontier."""
    if lane is None:
        return None
    try:
        runtime = resolve_runtime(None, lane)
        workload = resolve_workload(ir, runtime)
        frontier = build_frontier(ir, workload, runtime)
    except Exception as error:  # estimation is advisory; classification must not fail
        return {"error": f"hardware estimation unavailable: {error}"}
    star = next((tier for tier in frontier.tiers if tier.star), None)
    if star is None:
        return None
    return {
        "tier": star.id,
        "strategy": star.strategy,
        "resources": dict(star.resources),
        "confidence": star.confidence,
        "workload": {
            key: value
            for key, value in asdict(workload).items()
            if value is not None and key != "provenance"
        },
        "note": ("offer-independent floor estimate from lane defaults; not a placement decision"),
    }


def _recipe_candidate(recipe: registries.ExactRecipe, modality: str) -> RuntimeCandidate:
    return RuntimeCandidate(
        runtime=recipe.runtime,
        evidence_level=EvidenceLevel.EXACT_RECIPE,
        confidence=_LEVEL_CONFIDENCE[EvidenceLevel.EXACT_RECIPE],
        evidence=(Evidence(fact=recipe.evidence, source=f"exact recipe ({recipe.verified_at})"),),
        launch_hints=recipe.launch_hints,
        registry_version=f"exact-recipe:{recipe.verified_at}",
        smoke_test=_smoke_test(modality),
        notes=recipe.notes,
    )


def classify_model(
    ir: Mapping[str, Any],
    *,
    requested_revision: Optional[str] = None,
) -> ModelInspection:
    """Classify a Model Shape IR into ranked runtime candidates with evidence."""
    modality = str(ir.get("modality") or "research")
    library = str(ir.get("library") or "unknown")
    pipeline_class = ir.get("pipeline_class")
    lane = MODALITY_LANES.get(modality)
    model = str(ir.get("model") or "")
    candidates: list[RuntimeCandidate] = []
    negatives: list[NegativeResult] = []
    missing_facts: list[str] = []
    notes: list[str] = []

    recipe = registries.find_exact_recipe(model)
    components = list(ir.get("components") or [])
    has_weights = bool(components)

    if not has_weights:
        return ModelInspection(
            model=model,
            requested_revision=requested_revision,
            revision=str(ir.get("revision") or ""),
            modality=modality,
            library=library,
            pipeline_class=pipeline_class,
            classification="unsupported",
            candidates=(),
            negative_results=(
                NegativeResult(
                    runtime="*",
                    reason=(
                        "repository has no loadable weight files "
                        "(no safetensors/pickle/ckpt/GGUF siblings)"
                    ),
                ),
            ),
            missing_facts=("no weight files were found in the repository manifest",),
            notes=(
                "this repository cannot be served as a model endpoint; "
                "reject quickly instead of researching runtimes",
            ),
            hardware_hint=None,
            ir=ir,
        )

    if recipe is not None:
        candidates.append(_recipe_candidate(recipe, modality))

    if modality == "gguf":
        gguf = ir.get("gguf") or {}
        registry = registries.LLAMACPP_REGISTRY
        evidence = [
            Evidence(
                fact=(
                    f"repository contains GGUF checkpoints; selected "
                    f"{gguf.get('selected_file')!r} "
                    f"(reason: {gguf.get('selection_reason')})"
                ),
                source="file manifest",
            ),
            Evidence(
                fact="llama.cpp serves GGUF natively through llama-server",
                source=registry.source,
            ),
        ]
        min_cc, cap_missing, cap_notes = _capability_annotation(ir, "llama.cpp", lane)
        candidates.append(
            RuntimeCandidate(
                runtime="llama.cpp",
                evidence_level=EvidenceLevel.SUPPORT_REGISTRY,
                confidence=_LEVEL_CONFIDENCE[EvidenceLevel.SUPPORT_REGISTRY],
                evidence=tuple(evidence),
                missing_facts=(
                    "no gguf-parser memory estimate has been produced yet for the "
                    "selected quantization",
                    *registry.notes[1:],
                    *cap_missing,
                ),
                registry_version=registry.version,
                min_compute_capability=min_cc,
                smoke_test=_smoke_test("gguf"),
                notes=cap_notes,
            )
        )
        if not registries.VLLM_REGISTRY.gguf:
            notes.append(
                "GGUF on vLLM is not pinned as supported at the current image "
                "digest; do not plan for it without new evidence"
            )
    elif modality in {"text", "vlm"}:
        architectures = list((ir.get("text_arch") or {}).get("architectures") or [])
        if not architectures:
            missing_facts.append(
                "config.json declares no architectures; runtime support cannot be classified"
            )
            negatives.append(
                NegativeResult(
                    runtime="vllm",
                    reason="no config.architectures to match against the support registry",
                )
            )
            notes.append("research the model class before choosing a serving runtime")
        else:
            matched_any = False
            for registry in (registries.VLLM_REGISTRY, registries.SGLANG_REGISTRY):
                min_cc, cap_missing, cap_notes = _capability_annotation(ir, registry.runtime, lane)
                matched = [arch for arch in architectures if registry.supports_architecture(arch)]
                if matched:
                    matched_any = True
                    candidates.append(
                        RuntimeCandidate(
                            runtime=registry.runtime,
                            evidence_level=EvidenceLevel.SUPPORT_REGISTRY,
                            confidence=_LEVEL_CONFIDENCE[EvidenceLevel.SUPPORT_REGISTRY],
                            evidence=(
                                Evidence(
                                    fact=(
                                        f"config.architectures {matched} present in the "
                                        f"pinned {registry.runtime} support registry"
                                    ),
                                    source=registry.source,
                                ),
                            ),
                            missing_facts=(*registry.notes[:1], *cap_missing)
                            if registry.runtime == "sglang"
                            else cap_missing,
                            registry_version=registry.version,
                            min_compute_capability=min_cc,
                            smoke_test=_smoke_test(modality),
                            notes=cap_notes,
                        )
                    )
                else:
                    candidates.append(
                        RuntimeCandidate(
                            runtime=registry.runtime,
                            evidence_level=EvidenceLevel.DISCOVERY_HINT,
                            confidence=_LEVEL_CONFIDENCE[EvidenceLevel.DISCOVERY_HINT],
                            evidence=(
                                Evidence(
                                    fact=(
                                        f"library is transformers with architectures "
                                        f"{architectures}, but none appear in the pinned "
                                        f"{registry.runtime} registry"
                                    ),
                                    source="config.json + pinned registry",
                                ),
                            ),
                            missing_facts=(
                                f"architectures {architectures} are absent from the pinned "
                                f"{registry.runtime} registry; verify against the live "
                                "registry or prove support with a smoke test",
                            ),
                            registry_version=registry.version,
                            min_compute_capability=min_cc,
                            smoke_test=_smoke_test(modality),
                        )
                    )
            if not matched_any:
                missing_facts.append(
                    f"architectures {architectures} matched no pinned support registry"
                )
    elif modality in {"image", "video"}:
        if pipeline_class:
            min_cc, cap_missing, cap_notes = _capability_annotation(ir, "vllm-omni", lane)
            if str(pipeline_class) in registries.VLLM_OMNI_REGISTRY.pipelines:
                registry = registries.VLLM_OMNI_REGISTRY
                candidates.append(
                    RuntimeCandidate(
                        runtime="vllm-omni",
                        evidence_level=EvidenceLevel.SUPPORT_REGISTRY,
                        confidence=_LEVEL_CONFIDENCE[EvidenceLevel.SUPPORT_REGISTRY],
                        evidence=(
                            Evidence(
                                fact=(
                                    f"model_index.json pipeline {pipeline_class!r} has a "
                                    "native vLLM-Omni implementation"
                                ),
                                source=registry.source,
                            ),
                        ),
                        missing_facts=cap_missing,
                        registry_version=registry.version,
                        min_compute_capability=min_cc,
                        smoke_test=_smoke_test(modality),
                        notes=cap_notes,
                    )
                )
            else:
                registry = registries.VLLM_OMNI_REGISTRY
                candidates.append(
                    RuntimeCandidate(
                        runtime="vllm-omni",
                        evidence_level=EvidenceLevel.GENERIC_ADAPTER,
                        confidence=_LEVEL_CONFIDENCE[EvidenceLevel.GENERIC_ADAPTER],
                        evidence=(
                            Evidence(
                                fact=(
                                    f"pipeline {pipeline_class!r} is not in the native "
                                    "vLLM-Omni set; the documented generic Diffusers "
                                    "adapter applies to standard model_index pipelines"
                                ),
                                source=registry.source,
                            ),
                        ),
                        missing_facts=(
                            f"generic-adapter compatibility for {pipeline_class!r} is "
                            "unproven until a real generation succeeds",
                            *cap_missing,
                        ),
                        launch_hints=("--omni", "--diffusion-load-format diffusers"),
                        registry_version=registry.version,
                        min_compute_capability=min_cc,
                        smoke_test=_smoke_test(modality),
                        notes=cap_notes,
                    )
                )
            adapter = registries.DIFFUSERS_ADAPTER_REGISTRY
            diffusers_min_cc = _capability_annotation(ir, "diffusers", lane)[0]
            candidates.append(
                RuntimeCandidate(
                    runtime="diffusers",
                    evidence_level=EvidenceLevel.GENERIC_ADAPTER,
                    confidence=_LEVEL_CONFIDENCE[EvidenceLevel.GENERIC_ADAPTER],
                    evidence=(
                        Evidence(
                            fact=(
                                f"standard model_index.json with pipeline "
                                f"{pipeline_class!r} loads through "
                                "DiffusionPipeline.from_pretrained"
                            ),
                            source=adapter.source,
                        ),
                    ),
                    missing_facts=(
                        "a Diffusers service wrapper (image, command, probe) must "
                        "still be selected and smoke-tested",
                    ),
                    registry_version=adapter.version,
                    min_compute_capability=diffusers_min_cc,
                    smoke_test=_smoke_test(modality),
                )
            )
        else:
            missing_facts.append(
                f"{modality} tags are present but there is no standard model_index.json"
            )
            negatives.append(
                NegativeResult(
                    runtime="diffusers",
                    reason=(
                        "no standard model_index.json: do not assume "
                        "DiffusionPipeline.from_pretrained() will load this repository"
                    ),
                    evidence=(
                        Evidence(
                            fact="file manifest lacks model_index.json",
                            source="file manifest",
                        ),
                    ),
                )
            )
            if recipe is None:
                notes.append(
                    "research an exact supported recipe or a compatible "
                    "Diffusers-layout variant repository"
                )
    else:  # research modality
        missing_facts.append(
            "modality metadata indicates a research/custom model without a "
            "standard loadable pipeline"
        )
        if library == "diffusers" and not pipeline_class:
            negatives.append(
                NegativeResult(
                    runtime="diffusers",
                    reason=(
                        "library_name=diffusers is only a claim: config and "
                        "model_index.json are missing, so the standard Diffusers "
                        "loading path is unproven"
                    ),
                )
            )

    # Keep only the strongest candidate per runtime (an exact recipe supersedes
    # the same runtime's registry match).
    deduped: list[RuntimeCandidate] = []
    seen_runtimes: set[str] = set()
    for candidate in sorted(candidates, key=_candidate_order):
        if candidate.runtime in seen_runtimes:
            continue
        seen_runtimes.add(candidate.runtime)
        deduped.append(candidate)
    candidates = deduped

    if not candidates:
        classification = "needs_research"
        notes.append(
            "no runtime candidate passed the evidence ladder; use the normalized "
            "snapshot and the unresolved questions above to drive research"
        )
    elif all(
        candidate.evidence_level in {EvidenceLevel.DISCOVERY_HINT, EvidenceLevel.RESEARCH}
        for candidate in candidates
    ):
        classification = "needs_research"
    else:
        classification = "supported"

    ordered = tuple(candidates)
    return ModelInspection(
        model=model,
        requested_revision=requested_revision,
        revision=str(ir.get("revision") or ""),
        modality=modality,
        library=library,
        pipeline_class=pipeline_class,
        classification=classification,
        candidates=ordered,
        negative_results=tuple(negatives),
        missing_facts=tuple(missing_facts),
        notes=tuple(notes),
        hardware_hint=_hardware_hint(ir, lane),
        ir=ir,
    )


__all__ = [
    "INSPECT_VERSION",
    "Evidence",
    "EvidenceLevel",
    "ModelInspection",
    "NegativeResult",
    "RuntimeCandidate",
    "classify_model",
]
