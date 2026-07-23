"""Workload resolution, analytic memory estimation, and deterministic tiers.

The estimation behavior (workload defaults, KV/activation arithmetic, GGUF
parser-backed tiers and low-confidence fallback, research floor, deterministic
tier ordering) is ported from gurdasnijor/dh @
e9ce1b951c9bf08adf57d6576cef5a4897ada3ac (``dh/model.py``, ``dh/estimate.py``,
``dh/frontier.py``) and locked by golden parity fixtures on observable
outputs. dh surface that the endpoint flow does not need — explicit workload
flags, offer enrichment, template resource merging, and launch-flag rewriting
— is deliberately not ported.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

from dstack._internal.cli.services.endpoints.inspect import gguf as gguf_parser
from dstack._internal.cli.services.endpoints.inspect.runtime import (
    ENGINE_BASE_FLOORS,
    NATIVE_OMNI_PIPELINES,
    CapabilityPath,
    Runtime,
    capability_paths,
)
from dstack._internal.cli.services.endpoints.inspect.runtime import (
    cfg_mode as resolve_cfg_mode,
)

GIB = 1024**3
ANALYTIC_CONSTANTS: Mapping[str, Any] = {
    "k_act_text": 12,
    "k_cudagraph": 4,
    "engine_fixed": {
        "vllm": int(1.5 * GIB),
        "sglang": int(1.5 * GIB),
        "vllm-omni": 1 * GIB,
        "diffusers": 1 * GIB,
        "llama.cpp": 1 * GIB,
        "unknown": 1 * GIB,
    },
    "k_vlm_act": 8,
    "k_te_act": 4,
    "k_latent_copies": 4,
    "k_act_dit": 24,
    "k_vae_decode": 2.5,
    "k_tiling": 0.4,
}
CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}


@dataclass(frozen=True)
class Workload:
    modality: str
    engine_key: str
    context: int | None = None
    batch: int | None = None
    kv_dtype: str | None = None
    image_tokens: int | None = None
    width: int | None = None
    height: int | None = None
    frames: int | None = None
    cfg: bool | None = None
    cfg_mode: str | None = None
    images_per_prompt: int | None = None
    prompt_tokens: int | None = None
    provenance: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Estimate:
    strategy: str
    vram_per_gpu_bytes: int | None
    host_ram_bytes: int
    disk_bytes: int
    breakdown: Mapping[str, Any]
    confidence: str
    notes: tuple[str, ...] = ()
    launchable: bool = True


def _choose(template: Any, model_default: Any, global_default: Any) -> tuple[Any, str]:
    if template is not None:
        return template, "template"
    if model_default is not None:
        return model_default, "model config"
    return global_default, "default"


def resolve_workload(ir: Mapping[str, Any], runtime: Runtime) -> Workload:
    """Resolve the default sizing workload for a model on a runtime."""
    modality = str(ir.get("modality") or "research")
    provenance: dict[str, str] = {}
    if modality in {"text", "vlm", "gguf"}:
        text = ir.get("text_arch") or {}
        maximum = text.get("max_position_embeddings")
        model_context = min(32768, int(maximum)) if maximum else None
        context, provenance["context"] = _choose(runtime.context, model_context, 32768)
        batch, provenance["batch"] = _choose(None, None, 8)
        kv_dtype, provenance["kv_dtype"] = _choose(runtime.kv_cache_dtype, None, "auto")
        image_tokens = None
        if modality == "vlm":
            image_tokens, provenance["image_tokens"] = _choose(None, None, 576)
        return Workload(
            modality=modality,
            engine_key=runtime.engine_key,
            context=int(context),
            batch=int(batch),
            kv_dtype=str(kv_dtype),
            image_tokens=int(image_tokens) if image_tokens is not None else None,
            provenance=provenance,
        )
    if modality in {"image", "video"}:
        architecture = ir.get("diffusion_arch") or {}
        sample = architecture.get("default_sample_size")
        spatial = int(architecture.get("vae_spatial_factor") or 8)
        if isinstance(sample, (list, tuple)) and len(sample) >= 2:
            model_height, model_width = int(sample[-2]) * spatial, int(sample[-1]) * spatial
        elif isinstance(sample, (int, float, str)):
            model_width = model_height = int(sample) * spatial
        else:
            model_width = model_height = None
        global_width, global_height = (1024, 1024) if modality == "image" else (1280, 720)
        width, provenance["width"] = _choose(None, model_width, global_width)
        height, provenance["height"] = _choose(None, model_height, global_height)
        frames, provenance["frames"] = _choose(
            None, 1 if modality == "image" else None, 1 if modality == "image" else 81
        )
        provenance["cfg"] = "default"
        images, provenance["images_per_prompt"] = _choose(None, None, 1)
        prompt_tokens, provenance["prompt_tokens"] = _choose(None, None, 256)
        provenance["cfg_mode"] = "runtime registry"
        return Workload(
            modality=modality,
            engine_key=runtime.engine_key,
            width=int(width),
            height=int(height),
            frames=int(frames),
            cfg=True,
            cfg_mode=resolve_cfg_mode(runtime, ir),
            images_per_prompt=int(images),
            prompt_tokens=int(prompt_tokens),
            provenance=provenance,
        )
    return Workload(modality="research", engine_key=runtime.engine_key, provenance={})


def _confidence(values: Sequence[str], default: str = "high") -> str:
    present = [value for value in values if value in CONFIDENCE_ORDER]
    return min(present or [default], key=CONFIDENCE_ORDER.__getitem__)


def _lower_confidence(value: str) -> str:
    return {"high": "medium", "medium": "low", "low": "low"}[value]


def _component_totals(ir: Mapping[str, Any]) -> tuple[int, int, str]:
    components = list(ir.get("components") or [])
    resident = sum(int(item.get("resident_weight_bytes") or 0) for item in components)
    checkpoint = sum(int(item.get("checkpoint_bytes") or 0) for item in components)
    confidence = _confidence(
        [str(item.get("residency_confidence") or "low") for item in components],
        "low",
    )
    return resident, checkpoint, confidence


def _compute_dtype_bytes(ir: Mapping[str, Any]) -> int:
    dtypes = {str(item.get("dtype") or "unknown") for item in ir.get("components") or []}
    if dtypes == {"fp32"}:
        return 4
    return 2


def _kv_dtype_bytes(workload: Workload, ir: Mapping[str, Any]) -> int:
    value = str(workload.kv_dtype or "auto").lower()
    if value.startswith("fp8") or value in {"int8", "uint8"}:
        return 1
    if value in {"fp32", "float32"}:
        return 4
    if value == "auto":
        return _compute_dtype_bytes(ir)
    return 2


def _engine_fixed(runtime: Runtime) -> int:
    return int(
        ANALYTIC_CONSTANTS["engine_fixed"].get(
            runtime.engine, ANALYTIC_CONSTANTS["engine_fixed"]["unknown"]
        )
    )


def _disk_bytes(ir: Mapping[str, Any]) -> int:
    _resident, checkpoint, _confidence_value = _component_totals(ir)
    return math.ceil(checkpoint * 2.0 + 30 * GIB)


def _text_kv_bytes(
    architecture: Mapping[str, Any], workload: Workload, kv_bytes: int
) -> tuple[int, Mapping[str, int]]:
    context = int(workload.context or 0)
    batch = int(workload.batch or 0)
    kv_heads = int(architecture["num_kv_heads"])
    head_dim = int(architecture["head_dim"])
    sliding_window = int(architecture.get("sliding_window") or context)
    linear_state_size = int(
        architecture.get("linear_state_size") or int(architecture.get("hidden_size") or 0) * 2
    )
    full = sliding = linear = 0
    for attention_type in architecture["attention_types"]:
        if attention_type == "linear":
            linear += 2 * batch * linear_state_size * kv_bytes
        elif attention_type == "sliding":
            sliding += 2 * batch * min(context, sliding_window) * kv_heads * head_dim * kv_bytes
        else:
            full += 2 * batch * context * kv_heads * head_dim * kv_bytes
    return full + sliding + linear, {
        "full_attention": full,
        "sliding_attention": sliding,
        "linear_state": linear,
    }


def estimate_text(
    ir: Mapping[str, Any],
    workload: Workload,
    runtime: Runtime,
    *,
    tp: int = 1,
) -> Estimate:
    architecture = ir.get("text_arch")
    if not isinstance(architecture, Mapping):
        raise ValueError("text architecture metadata is required")
    if tp <= 0:
        raise ValueError("tensor parallel size must be positive")
    resident, _checkpoint, component_confidence = _component_totals(ir)
    kv_width = _kv_dtype_bytes(workload, ir)
    kv_total, kv_breakdown = _text_kv_bytes(architecture, workload, kv_width)
    kv_heads = int(architecture["num_kv_heads"])
    tp_kv_shards = min(tp, kv_heads)
    kv_per_gpu = math.ceil(kv_total / tp_kv_shards)
    weights_per_gpu = math.ceil(resident / tp)
    dtype_bytes = _compute_dtype_bytes(ir)
    batch = int(workload.batch or 0)
    context = int(workload.context or 0)
    hidden = int(architecture["hidden_size"])
    layers = int(architecture["num_layers"])
    activation_tokens = min(context, runtime.chunked_prefill_tokens)
    activations = (
        int(ANALYTIC_CONSTANTS["k_act_text"]) * batch * activation_tokens * hidden * dtype_bytes
    )
    cudagraph = (
        0
        if runtime.enforce_eager
        else int(ANALYTIC_CONSTANTS["k_cudagraph"]) * batch * hidden * layers * dtype_bytes
    )
    vlm_activation = 0
    if workload.modality == "vlm":
        vlm_activation = (
            int(ANALYTIC_CONSTANTS["k_vlm_act"])
            * batch
            * int(workload.image_tokens or 0)
            * hidden
            * dtype_bytes
        )
    fixed = _engine_fixed(runtime)
    required = weights_per_gpu + kv_per_gpu + activations + cudagraph + vlm_activation + fixed
    provisioned = math.ceil(required / runtime.gpu_memory_utilization)
    confidence = component_confidence
    notes: list[str] = []
    if architecture.get("linear_state_source") == "fallback" and "linear" in architecture.get(
        "attention_types", []
    ):
        confidence = _lower_confidence(confidence)
        notes.append("linear-attention state size used the hidden-size fallback")
    if not runtime.from_template:
        confidence = _lower_confidence(confidence)
        notes.append("runtime settings are defaults; no template was resolved")
    if tp > kv_heads:
        notes.append(f"KV heads replicate beyond tp={kv_heads}; KV memory no longer falls per GPU")
    return Estimate(
        strategy=f"tp={tp}",
        vram_per_gpu_bytes=provisioned,
        host_ram_bytes=max(32 * GIB, math.ceil(resident * 0.5)),
        disk_bytes=_disk_bytes(ir),
        breakdown={
            "weights_total": resident,
            "weights_per_gpu": weights_per_gpu,
            "kv_total": kv_total,
            "kv_per_gpu": kv_per_gpu,
            "kv_by_attention_type": kv_breakdown,
            "tp_kv_shards": tp_kv_shards,
            "activations": activations,
            "vlm_encoder_activations": vlm_activation,
            "cudagraph": cudagraph,
            "engine_fixed": fixed,
            "vram_required": required,
            "gpu_memory_utilization": runtime.gpu_memory_utilization,
        },
        confidence=confidence,
        notes=tuple(notes),
    )


def _diffusion_geometry(
    architecture: Mapping[str, Any], workload: Workload
) -> Mapping[str, int | float]:
    width = int(workload.width or 0)
    height = int(workload.height or 0)
    frames = int(workload.frames or 1)
    spatial = int(architecture.get("vae_spatial_factor") or 8)
    temporal = int(architecture.get("vae_temporal_factor") or 1)
    transformer = architecture.get("transformer") or {}
    patch = int(transformer.get("patch_size") or 1)
    temporal_patch = int(transformer.get("temporal_patch_size") or 1)
    lat_h = math.ceil(height / spatial)
    lat_w = math.ceil(width / spatial)
    lat_t = math.ceil(frames / temporal)
    seq_tokens = (
        math.ceil(lat_h / patch) * math.ceil(lat_w / patch) * math.ceil(lat_t / temporal_patch)
    )
    cfg_multiplier = 2 if workload.cfg and workload.cfg_mode == "concat" else 1
    tx_batch = int(workload.images_per_prompt or 1) * cfg_multiplier
    return {
        "lat_h": lat_h,
        "lat_w": lat_w,
        "lat_t": lat_t,
        "seq_tokens": seq_tokens,
        "cfg_multiplier": cfg_multiplier,
        "tx_batch": tx_batch,
    }


def _phase_weights(ir: Mapping[str, Any], runtime: Runtime) -> Mapping[str, int]:
    weights = {
        str(item["name"]): int(item.get("resident_weight_bytes") or 0)
        for item in ir.get("components") or []
    }
    encode = sum(
        value
        for name, value in weights.items()
        if name.startswith(("text_encoder", "audio_encoder", "image_encoder"))
    )
    denoise = sum(
        value
        for name, value in weights.items()
        if name.startswith("transformer") or name == "unet"
    )
    decode = sum(value for name, value in weights.items() if name.startswith("vae"))
    if runtime.engine == "vllm-omni":
        encode += decode
        denoise += decode
    return {"encode": encode, "denoise": denoise, "decode": decode}


def estimate_diffusion(
    ir: Mapping[str, Any],
    workload: Workload,
    runtime: Runtime,
    *,
    strategy: str = "resident",
) -> Estimate:
    if strategy not in {"resident", "offload", "offload+tiling"}:
        raise ValueError(f"unsupported diffusion strategy: {strategy}")
    architecture = ir.get("diffusion_arch")
    if not isinstance(architecture, Mapping):
        raise ValueError("diffusion architecture metadata is required")
    resident, _checkpoint, component_confidence = _component_totals(ir)
    geometry = _diffusion_geometry(architecture, workload)
    tx_batch = int(geometry["tx_batch"])
    latent_channels = int(architecture.get("latent_channels") or 4)
    te_hidden = int(architecture.get("text_encoder_hidden_size") or 4096)
    transformer = architecture.get("transformer") or {}
    tx_hidden = int(transformer.get("hidden_size") or 4096)
    prompt_tokens = int(workload.prompt_tokens or 256)
    ws_encode = int(ANALYTIC_CONSTANTS["k_te_act"]) * prompt_tokens * te_hidden * 2
    latent_state = (
        tx_batch
        * latent_channels
        * int(geometry["lat_t"])
        * int(geometry["lat_h"])
        * int(geometry["lat_w"])
        * 2
        * int(ANALYTIC_CONSTANTS["k_latent_copies"])
    )
    attention = (
        int(ANALYTIC_CONSTANTS["k_act_dit"])
        * tx_batch
        * int(geometry["seq_tokens"])
        * tx_hidden
        * 2
    )
    ws_denoise = latent_state + attention
    ws_decode = math.ceil(
        float(ANALYTIC_CONSTANTS["k_vae_decode"])
        * int(workload.width or 0)
        * int(workload.height or 0)
        * int(workload.frames or 1)
        * 3
        * 4
    )
    if strategy == "offload+tiling":
        ws_decode = math.ceil(ws_decode * float(ANALYTIC_CONSTANTS["k_tiling"]))
    working_sets = {
        "encode": ws_encode,
        "denoise": ws_denoise,
        "decode": ws_decode,
    }
    fixed = _engine_fixed(runtime)
    phase_weights = _phase_weights(ir, runtime)
    if strategy == "resident":
        peak_phase = max(working_sets, key=working_sets.__getitem__)
        peak = resident + working_sets[peak_phase] + fixed
        weights_at_peak = resident
    else:
        phase_totals = {name: phase_weights[name] + working_sets[name] for name in working_sets}
        peak_phase = max(phase_totals, key=phase_totals.__getitem__)
        peak = phase_totals[peak_phase] + fixed
        weights_at_peak = phase_weights[peak_phase]
    confidence = _confidence([component_confidence, "medium"])
    notes: list[str] = []
    if architecture.get("text_encoder_hidden_source") == "fallback" or not transformer.get(
        "hidden_size"
    ):
        confidence = _lower_confidence(confidence)
        notes.append("one or more diffusion hidden sizes used a fallback")
    if workload.cfg_mode == "separate" and workload.cfg:
        notes.append("separate CFG uses one-batch peak memory and approximately 2x wall time")
    return Estimate(
        strategy=strategy,
        vram_per_gpu_bytes=math.ceil(peak),
        host_ram_bytes=math.ceil(resident * 1.1 + fixed),
        disk_bytes=_disk_bytes(ir),
        breakdown={
            "weights_total": resident,
            "weights_at_peak": weights_at_peak,
            "phase_weights": phase_weights,
            "working_sets": working_sets,
            "peak_phase": peak_phase,
            "engine_fixed": fixed,
            "geometry": geometry,
            "latent_state": latent_state,
            "attention_activations": attention,
        },
        confidence=confidence,
        notes=tuple(notes),
    )


def estimate_gguf_fallback(
    ir: Mapping[str, Any],
    workload: Workload,
    runtime: Runtime,
    *,
    strategy: str,
) -> Estimate:
    if strategy not in {"full-offload", "cpu-only"}:
        raise ValueError(f"unsupported GGUF fallback strategy: {strategy}")
    resident, _checkpoint, _component_confidence = _component_totals(ir)
    architecture = ir.get("text_arch")
    kv_total = 0
    if isinstance(architecture, Mapping):
        kv_total, _parts = _text_kv_bytes(architecture, workload, _kv_dtype_bytes(workload, ir))
    fixed = _engine_fixed(runtime)
    if strategy == "full-offload":
        vram = resident + kv_total + fixed
        host = fixed + math.ceil(resident * 0.1)
    else:
        vram = None
        host = math.ceil(resident * 1.1 + kv_total + fixed)
    notes = ["gguf-parser unavailable; selected-file fallback is a low-confidence floor"]
    if not architecture:
        notes.append("no text architecture was available; KV memory is omitted")
    return Estimate(
        strategy=strategy,
        vram_per_gpu_bytes=vram,
        host_ram_bytes=host,
        disk_bytes=_disk_bytes(ir),
        breakdown={
            "weights": resident,
            "kv": kv_total,
            "engine_fixed": fixed,
        },
        confidence="low",
        notes=tuple(notes),
    )


def estimate_gguf_from_parser(ir: Mapping[str, Any], result: Mapping[str, Any]) -> list[Estimate]:
    return [
        Estimate(
            strategy=item["strategy"],
            vram_per_gpu_bytes=(None if item["strategy"] == "cpu-only" else item["vram_bytes"]),
            host_ram_bytes=item["host_memory_bytes"],
            disk_bytes=_disk_bytes(ir),
            breakdown={"gguf_parser": item["raw"], "gpu_layers": item["gpu_layers"]},
            confidence="high",
            notes=("memory values reported by gguf-parser",),
        )
        for item in gguf_parser.configurations(result)
    ]


def estimate_research_floor(ir: Mapping[str, Any], runtime: Runtime) -> Estimate:
    resident, _checkpoint, confidence = _component_totals(ir)
    fixed = _engine_fixed(runtime)
    return Estimate(
        strategy="research-floor",
        vram_per_gpu_bytes=resident + fixed if resident else None,
        host_ram_bytes=math.ceil(resident * 1.1 + fixed),
        disk_bytes=_disk_bytes(ir),
        breakdown={"weights_floor": resident, "engine_fixed": fixed},
        confidence=_lower_confidence(confidence),
        notes=(
            "weights-only research floor; runtime and workload working sets are model-specific",
        ),
        launchable=False,
    )


def estimate_all(
    ir: Mapping[str, Any],
    workload: Workload,
    runtime: Runtime,
    *,
    gguf_parser_result: Mapping[str, Any] | None = None,
) -> list[Estimate]:
    modality = str(ir.get("modality"))
    if modality in {"text", "vlm"}:
        return [estimate_text(ir, workload, runtime)]
    if modality in {"image", "video"}:
        return [
            estimate_diffusion(ir, workload, runtime, strategy=strategy)
            for strategy in ("resident", "offload", "offload+tiling")
        ]
    if modality == "gguf":
        if gguf_parser_result is not None:
            parsed = estimate_gguf_from_parser(ir, gguf_parser_result)
            if parsed:
                return parsed
        fallback = [
            estimate_gguf_fallback(ir, workload, runtime, strategy=strategy)
            for strategy in ("full-offload", "cpu-only")
        ]
        if gguf_parser_result is not None:
            fallback = [
                replace(
                    item,
                    notes=item.notes
                    + (
                        "gguf-parser output had no recognized memory configurations; "
                        "using the selected-file fallback",
                    ),
                )
                for item in fallback
            ]
        return fallback
    return [estimate_research_floor(ir, runtime)]


@dataclass(frozen=True)
class Tier:
    """One offer-independent hardware tier for a strategy and capability path."""

    id: str
    strategy: str
    gpu_count: int
    vram_per_gpu_bytes: int | None
    host_ram_bytes: int
    disk_bytes: int
    min_compute_capability: float | None
    capability_path: tuple[str, str | None, str] | None
    capability_preference_rank: int
    resources: Mapping[str, Any]
    confidence: str
    notes: tuple[str, ...] = ()
    launchable: bool = True
    star: bool = False


@dataclass(frozen=True)
class Frontier:
    tiers: tuple[Tier, ...]
    star: str | None


def valid_tp(num_attention_heads: int, num_kv_heads: int, tp: int) -> bool:
    if tp <= 0 or num_attention_heads % tp:
        return False
    if tp <= num_kv_heads:
        return num_kv_heads % tp == 0
    return tp % num_kv_heads == 0


def _ceil_gib(value: int) -> int:
    return max(1, math.ceil(value / GIB))


def render_resources(
    estimate: Estimate,
    *,
    gpu_count: int,
    min_compute_capability: float | None,
) -> dict[str, Any]:
    resources: dict[str, Any] = {
        "memory": f"{_ceil_gib(estimate.host_ram_bytes)}GB..",
        "disk": f"{_ceil_gib(estimate.disk_bytes)}GB..",
    }
    if estimate.vram_per_gpu_bytes is not None:
        gpu: dict[str, Any] = {
            "memory": f"{_ceil_gib(estimate.vram_per_gpu_bytes)}GB..",
            "count": gpu_count,
        }
        if min_compute_capability is not None:
            gpu["compute_capability"] = float(min_compute_capability)
        resources["gpu"] = gpu
    if gpu_count > 1:
        resources["shm_size"] = "16GB"
    return resources


def _tier_id(strategy: str, path: CapabilityPath | None) -> str:
    suffix = "research" if path is None else "-".join(str(value or "none") for value in path.key)
    raw = f"{strategy}-{suffix}".lower()
    return re.sub(r"[^a-z0-9]+", "-", raw).strip("-")


def _effective_floor(runtime: Runtime, path: CapabilityPath) -> float | None:
    values = [
        value
        for value in (ENGINE_BASE_FLOORS.get(runtime.engine), path.min_cc)
        if value is not None
    ]
    return max(values) if values else None


def _tier(
    estimate: Estimate,
    runtime: Runtime,
    path: CapabilityPath | None,
    *,
    gpu_count: int,
) -> Tier:
    floor = _effective_floor(runtime, path) if path is not None else None
    notes = list(estimate.notes)
    if path is not None and path.note:
        notes.append(path.note)
    return Tier(
        id=_tier_id(estimate.strategy, path),
        strategy=estimate.strategy,
        gpu_count=gpu_count,
        vram_per_gpu_bytes=estimate.vram_per_gpu_bytes,
        host_ram_bytes=estimate.host_ram_bytes,
        disk_bytes=estimate.disk_bytes,
        min_compute_capability=floor,
        capability_path=path.key if path is not None else None,
        capability_preference_rank=path.preference_rank if path is not None else 10_000,
        resources=render_resources(
            estimate,
            gpu_count=gpu_count,
            min_compute_capability=floor,
        ),
        confidence=estimate.confidence,
        notes=tuple(notes),
        launchable=estimate.launchable and path is not None,
    )


def _operational_rank(strategy: str) -> int:
    if strategy in {"resident", "full-offload"} or strategy.startswith("tp="):
        return 0
    if strategy == "offload":
        return 1
    if strategy == "offload+tiling":
        return 2
    if strategy == "cpu-only":
        return 3
    return 4


def deterministic_sort_key(tier: Tier) -> tuple[float, int, int, int, str]:
    return (
        float(tier.vram_per_gpu_bytes) if tier.vram_per_gpu_bytes is not None else float("inf"),
        tier.gpu_count,
        _operational_rank(tier.strategy),
        tier.capability_preference_rank,
        tier.id,
    )


def _candidate_text_estimates(
    ir: Mapping[str, Any], workload: Workload, runtime: Runtime, max_tp: int
) -> list[tuple[Estimate, int]]:
    architecture = ir.get("text_arch") or {}
    heads = int(architecture.get("num_heads") or architecture.get("num_attention_heads") or 0)
    kv_heads = int(architecture.get("num_kv_heads") or 0)
    candidates = [1]
    candidates.extend(tp for tp in (2, 4, 8, 16) if tp <= max_tp and valid_tp(heads, kv_heads, tp))
    return [(estimate_text(ir, workload, runtime, tp=tp), tp) for tp in candidates]


def build_frontier(
    ir: Mapping[str, Any],
    workload: Workload,
    runtime: Runtime,
    *,
    gguf_parser_result: Mapping[str, Any] | None = None,
    max_tp: int = 16,
) -> Frontier:
    """Build the offer-independent tier set with its deterministic star."""
    eligible, _advisory = capability_paths(runtime, ir)
    modality = str(ir.get("modality") or "research")
    tiers: list[Tier] = []
    if modality in {"text", "vlm"}:
        for estimate, tp in _candidate_text_estimates(ir, workload, runtime, max_tp):
            for path in eligible:
                tiers.append(_tier(estimate, runtime, path, gpu_count=tp))
    elif modality == "research":
        estimate = estimate_all(ir, workload, runtime)[0]
        tiers.append(_tier(estimate, runtime, None, gpu_count=1))
    else:
        for estimate in estimate_all(ir, workload, runtime, gguf_parser_result=gguf_parser_result):
            if (
                modality in {"image", "video"}
                and runtime.engine == "vllm-omni"
                and runtime.pipeline_backend == "native"
                and str(ir.get("pipeline_class") or "") not in NATIVE_OMNI_PIPELINES
                and estimate.strategy != "resident"
            ):
                continue
            for path in eligible:
                tiers.append(
                    _tier(
                        estimate,
                        runtime,
                        path,
                        gpu_count=0 if estimate.vram_per_gpu_bytes is None else 1,
                    )
                )

    tiers.sort(key=lambda item: item.id)
    launchable = [tier for tier in tiers if tier.launchable]
    star = min(launchable, key=deterministic_sort_key).id if launchable else None
    return Frontier(
        tuple(replace(tier, star=tier.id == star) for tier in tiers),
        star,
    )


__all__ = [
    "ANALYTIC_CONSTANTS",
    "Estimate",
    "Frontier",
    "GIB",
    "Tier",
    "Workload",
    "build_frontier",
    "deterministic_sort_key",
    "estimate_all",
    "estimate_diffusion",
    "estimate_gguf_fallback",
    "estimate_gguf_from_parser",
    "estimate_research_floor",
    "estimate_text",
    "render_resources",
    "resolve_workload",
    "valid_tp",
]
