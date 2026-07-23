"""Template runtime resolution, engine fingerprints, and capability paths.

The fingerprinting and capability-predicate behavior is ported from
gurdasnijor/dh @ e9ce1b951c9bf08adf57d6576cef5a4897ada3ac (``dh/runtime.py``)
and locked by golden fixtures. dh's launch-flag rewriting and engine-log
telemetry are deliberately not ported; the endpoint flow does not mutate
launch commands.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping

DEFAULT_GPU_MEMORY_UTILIZATION = 0.90
DEFAULT_CHUNKED_PREFILL_TOKENS = 8192
CURRENT_VLLM_IMAGE_DIGEST = (
    "vllm/vllm-openai:nightly@sha256:"
    "7f2bc168366c77fbd8329368f00310d208531c14ece6c2de31a6611ef99f6ec8"
)
# Marlin paths remain advisory unless this digest exactly matches an image that
# has been profiled with the corresponding kernel. Replace it only with evidence
# from the deployed template image, never a floating tag.


@dataclass(frozen=True)
class EngineFingerprint:
    image: str
    engine_key: str
    flags: tuple[str, ...]

    @property
    def key(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:24]


@dataclass(frozen=True)
class Runtime:
    lane: str
    engine: str
    engine_key: str
    image: str
    command: str
    env: Mapping[str, str | None]
    gpu_memory_utilization: float
    kv_cache_dtype: str
    enforce_eager: bool
    chunked_prefill_tokens: int
    context: int | None
    pipeline_backend: str | None
    fingerprint: EngineFingerprint
    from_template: bool = True


@dataclass(frozen=True)
class CapabilityPath:
    key: tuple[str, str | None, str]
    min_cc: float | None
    image_predicate: Mapping[str, Any]
    model_predicate: Mapping[str, Any]
    runtime_overrides: tuple[str, ...] = ()
    k_resident: float = 1.0
    preference_rank: int = 10
    launchable: bool = True
    note: str | None = None


@dataclass(frozen=True)
class CapabilityDecision:
    selected: CapabilityPath | None
    eligible: tuple[CapabilityPath, ...]
    advisory: tuple[CapabilityPath, ...]
    effective_min_cc: float | None


ENGINE_BASE_FLOORS: Mapping[str, float | None] = {
    "vllm": 7.5,
    "vllm-omni": 8.0,
    "sglang": 8.0,
    "llama.cpp": None,
    "diffusers": 7.5,
}

CFG_MODES: Mapping[str, str] = {
    "FluxPipeline": "distilled",
    "FluxImg2ImgPipeline": "distilled",
}

NATIVE_OMNI_PIPELINES = frozenset({"GlmImagePipeline", "HunyuanVideo15Pipeline", "WanPipeline"})


CAPABILITY_PATHS: tuple[CapabilityPath, ...] = (
    CapabilityPath(
        ("vllm", None, "base"),
        7.5,
        {"contains": "vllm"},
        {"dtype_in": ("fp16", "fp32", "unknown", "mixed")},
    ),
    CapabilityPath(
        ("vllm", None, "bf16_native"),
        8.0,
        {"contains": "vllm"},
        {"dtype_in": ("bf16",)},
        note="native BF16",
    ),
    CapabilityPath(
        ("vllm", None, "bf16_as_fp16"),
        7.5,
        {"contains": "vllm"},
        {"dtype_in": ("bf16",)},
        runtime_overrides=("--dtype", "float16"),
        preference_rank=20,
        note="loads BF16 checkpoint as FP16; numerics differ",
    ),
    CapabilityPath(
        ("vllm", "fp8", "native"),
        8.9,
        {"contains": "vllm"},
        {"quant_method": "fp8"},
    ),
    CapabilityPath(
        ("vllm", "mxfp4", "native"),
        9.0,
        {"contains": "vllm"},
        {"quant_method": "mxfp4"},
    ),
    CapabilityPath(
        ("vllm", "mxfp4", "marlin"),
        8.0,
        {"image_in": (CURRENT_VLLM_IMAGE_DIGEST,)},
        {"quant_method": "mxfp4"},
        runtime_overrides=("--quantization", "mxfp4_marlin"),
        preference_rank=20,
        note="Marlin MXFP4 path pinned to the profiled vLLM image",
    ),
    CapabilityPath(
        ("vllm", "awq", "base"),
        7.5,
        {"contains": "vllm"},
        {"quant_method": "awq"},
    ),
    CapabilityPath(
        ("vllm", "awq", "marlin"),
        8.0,
        {"image_in": (CURRENT_VLLM_IMAGE_DIGEST,)},
        {"quant_method": "awq"},
        runtime_overrides=("--quantization", "awq_marlin"),
        preference_rank=20,
        note="faster Marlin path",
    ),
    CapabilityPath(
        ("vllm", "gptq", "base"),
        7.5,
        {"contains": "vllm"},
        {"quant_method": "gptq"},
    ),
    CapabilityPath(
        ("sglang", None, "base"),
        8.0,
        {"contains": "sglang"},
        {},
    ),
    CapabilityPath(
        ("llama.cpp", None, "base"),
        None,
        {},
        {},
        note="pre-7.0 support is low confidence",
    ),
    CapabilityPath(
        ("vllm-omni", None, "base"),
        8.0,
        {"contains": "vllm-omni"},
        {},
    ),
    CapabilityPath(
        ("diffusers", None, "base"),
        7.5,
        {},
        {},
    ),
)


MEMORY_FLAGS = (
    "--gpu-memory-utilization",
    "--kv-cache-dtype",
    "--dtype",
    "--quantization",
    "--tensor-parallel-size",
    "--max-num-batched-tokens",
    "--max-model-len",
    "--enable-cpu-offload",
    "--vae-use-tiling",
    "--enforce-eager",
    "--n-gpu-layers",
    "--ctx-size",
    "--cache-type-k",
    "--cache-type-v",
    "--split-mode",
    "--tensor-split",
    "--main-gpu",
    "--flash-attn",
    "--no-kv-offload",
    "--mmap",
    "--no-mmap",
    "--swa-full",
    "--batch-size",
    "--ubatch-size",
)


def template_env(template: Mapping[str, Any]) -> dict[str, str | None]:
    raw = template.get("env") or {}
    if isinstance(raw, Mapping):
        return {str(key): None if value is None else str(value) for key, value in raw.items()}
    result: dict[str, str | None] = {}
    for item in raw:
        name, separator, value = str(item).partition("=")
        result[name] = value if separator else None
    return result


def _flag_value(command: str, flag: str) -> str | None:
    pattern = re.compile(rf"(?<!\S){re.escape(flag)}(?:=|\s+(?:\\\s*\n\s*)?)([^\s\\]+)")
    match = pattern.search(command)
    if not match:
        return None
    return match.group(1).strip("'\"")


def _has_flag(command: str, flag: str) -> bool:
    return bool(re.search(rf"(?<!\S){re.escape(flag)}(?=\s|=|$)", command))


def _resolve_env_reference(value: str | None, env: Mapping[str, str | None]) -> str | None:
    if not value:
        return value
    match = re.fullmatch(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?", value)
    if match:
        return env.get(match.group(1))
    return value


def _int_value(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def _detect_engine(image: str, command: str) -> str:
    lowered = f"{image} {command}".lower()
    if "llama-server" in lowered or "llama.cpp" in lowered:
        return "llama.cpp"
    if "sglang" in lowered:
        return "sglang"
    if "vllm" in lowered and ("--omni" in command or "vllm-omni" in lowered):
        return "vllm-omni"
    if "vllm" in lowered:
        return "vllm"
    if "diffusers" in lowered:
        return "diffusers"
    return "unknown"


def _fingerprint_flags(
    command: str, env: Mapping[str, str | None], pipeline_backend: str | None
) -> tuple[str, ...]:
    values: list[str] = []
    for flag in MEMORY_FLAGS:
        if value := _flag_value(command, flag):
            values.append(f"{flag}={_resolve_env_reference(value, env) or value}")
        elif _has_flag(command, flag):
            values.append(flag)
    if pipeline_backend:
        values.append(f"PIPELINE_BACKEND={pipeline_backend}")
    return tuple(sorted(values))


def resolve_runtime(
    template: Mapping[str, Any] | None,
    lane: str,
    *,
    pipeline_backend: str | None = None,
) -> Runtime:
    from_template = bool(template)
    if not template:
        if lane in {"general-llm", "general-vlm"}:
            template = {
                "image": "vllm:lane-default",
                "commands": ["vllm serve model"],
            }
        elif lane in {"general-image", "general-video"}:
            template = {
                "image": "vllm-omni:lane-default",
                "commands": ["vllm serve model --omni"],
            }
        elif lane == "general-gguf":
            template = {
                "image": "llama.cpp:lane-default",
                "commands": ["llama-server"],
            }
        else:
            template = {}
    commands = list(template.get("commands") or [])
    command = str(commands[-1]) if commands else ""
    image = str(template.get("image") or "")
    env = template_env(template)
    engine = _detect_engine(image, command)
    backend = pipeline_backend or env.get("PIPELINE_BACKEND")
    if engine == "vllm-omni":
        backend = backend or "native"
        engine_key = f"vllm-omni:{backend}"
    else:
        engine_key = engine
    util_value = _flag_value(command, "--gpu-memory-utilization")
    util_value = _resolve_env_reference(util_value, env)
    try:
        gpu_memory_utilization = (
            float(util_value) if util_value is not None else DEFAULT_GPU_MEMORY_UTILIZATION
        )
    except ValueError:
        gpu_memory_utilization = DEFAULT_GPU_MEMORY_UTILIZATION
    kv_cache_dtype = (
        _resolve_env_reference(_flag_value(command, "--kv-cache-dtype"), env) or "auto"
    )
    chunked = (
        _int_value(_resolve_env_reference(_flag_value(command, "--max-num-batched-tokens"), env))
        or DEFAULT_CHUNKED_PREFILL_TOKENS
    )
    context_value = _resolve_env_reference(_flag_value(command, "--max-model-len"), env)
    if context_value is None:
        context_value = env.get("MAX_MODEL_LEN") or env.get("CONTEXT_SIZE")
    context = _int_value(context_value)
    flags = _fingerprint_flags(command, env, backend)
    fingerprint = EngineFingerprint(image=image, engine_key=engine_key, flags=flags)
    return Runtime(
        lane=lane,
        engine=engine,
        engine_key=engine_key,
        image=image,
        command=command,
        env=env,
        gpu_memory_utilization=gpu_memory_utilization,
        kv_cache_dtype=kv_cache_dtype,
        enforce_eager=_has_flag(command, "--enforce-eager"),
        chunked_prefill_tokens=chunked,
        context=context,
        pipeline_backend=backend,
        fingerprint=fingerprint,
        from_template=from_template,
    )


def _engine_family(runtime: Runtime) -> str:
    return runtime.engine


def _model_dtype(ir: Mapping[str, Any]) -> str:
    values = {str(component.get("dtype") or "unknown") for component in ir.get("components") or []}
    return next(iter(values)) if len(values) == 1 else "mixed"


def _quant_method(ir: Mapping[str, Any]) -> str | None:
    method = (ir.get("quantization") or {}).get("method")
    if isinstance(method, str) and method.startswith("gguf:"):
        return None
    return str(method) if method else None


def _image_matches(predicate: Mapping[str, Any], image: str) -> bool:
    if contains := predicate.get("contains"):
        if str(contains).lower() not in image.lower():
            return False
    if images := predicate.get("image_in"):
        if image not in images:
            return False
    if prefix := predicate.get("prefix"):
        if not image.startswith(str(prefix)):
            return False
    return True


def _model_matches(predicate: Mapping[str, Any], ir: Mapping[str, Any]) -> bool:
    if quant := predicate.get("quant_method"):
        if _quant_method(ir) != quant:
            return False
    if dtypes := predicate.get("dtype_in"):
        if _model_dtype(ir) not in dtypes:
            return False
    if pattern := predicate.get("architecture_regex"):
        architectures = " ".join((ir.get("text_arch") or {}).get("architectures") or [])
        if not re.search(str(pattern), architectures):
            return False
    return True


def _path_order(path: CapabilityPath) -> tuple[int, float, tuple[str, str | None, str]]:
    return (
        path.preference_rank,
        float("inf") if path.min_cc is None else path.min_cc,
        path.key,
    )


def capability_paths(
    runtime: Runtime, ir: Mapping[str, Any]
) -> tuple[tuple[CapabilityPath, ...], tuple[CapabilityPath, ...]]:
    engine = _engine_family(runtime)
    quant = _quant_method(ir)
    eligible: list[CapabilityPath] = []
    advisory: list[CapabilityPath] = []
    for path in CAPABILITY_PATHS:
        if path.key[0] != engine:
            continue
        if quant is not None and path.key[1] != quant:
            continue
        if quant is None and path.key[1] is not None:
            continue
        model_matches = _model_matches(path.model_predicate, ir)
        image_matches = _image_matches(path.image_predicate, runtime.image)
        if model_matches and image_matches and path.launchable:
            eligible.append(path)
        elif model_matches:
            advisory.append(path)
    return tuple(sorted(eligible, key=_path_order)), tuple(sorted(advisory, key=_path_order))


def select_capability_path(runtime: Runtime, ir: Mapping[str, Any]) -> CapabilityDecision:
    eligible, advisory = capability_paths(runtime, ir)
    selected = eligible[0] if eligible else None
    base_floor = ENGINE_BASE_FLOORS.get(runtime.engine)
    floors = [
        value for value in (base_floor, selected.min_cc if selected else None) if value is not None
    ]
    effective = max(floors) if floors else None
    return CapabilityDecision(selected, eligible, advisory, effective)


def cfg_mode(runtime: Runtime, ir: Mapping[str, Any]) -> str:
    """Resolve peak-memory CFG behavior from the concrete pipeline registry."""
    pipeline_class = str(ir.get("pipeline_class") or "")
    return CFG_MODES.get(pipeline_class, "concat")


__all__ = [
    "CAPABILITY_PATHS",
    "CURRENT_VLLM_IMAGE_DIGEST",
    "CapabilityDecision",
    "CapabilityPath",
    "ENGINE_BASE_FLOORS",
    "EngineFingerprint",
    "NATIVE_OMNI_PIPELINES",
    "Runtime",
    "capability_paths",
    "cfg_mode",
    "resolve_runtime",
    "select_capability_path",
]
