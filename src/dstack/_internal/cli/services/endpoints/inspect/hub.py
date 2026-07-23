"""Revision-pinned, metadata-only Model Shape IR construction.

Ported from gurdasnijor/dh @ e9ce1b951c9bf08adf57d6576cef5a4897ada3ac
(``dh/hub.py``). The IR construction, weight/GGUF selection, and architecture
extraction are kept byte-for-byte compatible with the reference; parity is
locked by golden fixtures. Local changes relative to dh:

- the user-level IR cache is removed; the raw normalized snapshot is stored in
  the preset-creation session instead (see :func:`fetch_hub_snapshot`);
- live fetching returns a :class:`HubSnapshot` so the endpoint controller can
  persist exactly what the classifier saw.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

IR_VERSION = 2
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt")
GGUF_SUFFIX = ".gguf"
QUANT_SELECTOR = re.compile(r"^(?:I?Q\d(?:_[A-Z0-9]+)*|BF16|F16|F32)$", re.IGNORECASE)
SHARD = re.compile(r"-(\d{5})-of-(\d{5})(?=\.[^.]+$)", re.IGNORECASE)
VARIANT = re.compile(r"(?:^|[._-])(fp16|bf16|fp32)(?=[._-]|$)", re.IGNORECASE)


class ModelMetadataError(ValueError):
    """The repository metadata cannot form a safe, deterministic IR."""


@dataclass(frozen=True)
class WeightFile:
    name: str
    size: int


@dataclass(frozen=True)
class HubSnapshot:
    """Raw normalized Hub metadata captured for one immutable revision."""

    repo: str
    requested_revision: str | None
    revision: str
    model_info: dict[str, Any]
    documents: dict[str, Any]
    huggingface_hub_version: str
    fetched_files: tuple[str, ...] = field(default_factory=tuple)

    def to_data(self) -> dict[str, Any]:
        return {
            "snapshot_version": 1,
            "repo": self.repo,
            "requested_revision": self.requested_revision,
            "revision": self.revision,
            "model_info": self.model_info,
            "documents": self.documents,
            "huggingface_hub_version": self.huggingface_hub_version,
            "fetched_files": list(self.fetched_files),
        }


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json", exclude_none=True))
    if hasattr(value, "dict"):
        return _jsonable(value.dict(exclude_none=True))
    if hasattr(value, "__dict__"):
        return {
            key: _jsonable(item)
            for key, item in vars(value).items()
            if not key.startswith("_") and item is not None
        }
    return str(value)


def split_model_selector(model: str) -> tuple[str, str | None]:
    """Split llama.cpp-style ``repo:QUANT`` without treating tags as revisions."""
    if ":" not in model:
        return model, None
    repo, candidate = model.rsplit(":", 1)
    normalized = normalize_quant(candidate)
    if "/" in repo and QUANT_SELECTOR.fullmatch(normalized):
        return repo, normalized
    return model, None


def normalize_quant(value: str) -> str:
    return re.sub(r"[-\s]+", "_", value.strip().upper())


def _filename(item: Any) -> str:
    if isinstance(item, Mapping):
        return str(item.get("rfilename") or item.get("name") or "")
    return str(getattr(item, "rfilename", "") or getattr(item, "name", ""))


def _filesize(item: Any) -> int:
    if isinstance(item, Mapping):
        return int(item.get("size") or 0)
    return int(getattr(item, "size", 0) or 0)


def _weight_records(siblings: Iterable[Any]) -> list[WeightFile]:
    return [
        WeightFile(_filename(item), _filesize(item))
        for item in siblings
        if _filename(item).lower().endswith(WEIGHT_SUFFIXES)
    ]


def _component(filename: str) -> str:
    parts = PurePosixPath(filename).parts
    return parts[0] if len(parts) > 1 else "root"


def _family(filename: str) -> str:
    suffix = PurePosixPath(filename).suffix.lower()
    if suffix == ".safetensors":
        return "safetensors"
    if suffix in {".bin", ".pt", ".pth"}:
        return "pickle"
    return "ckpt"


def _variant(filename: str) -> str | None:
    match = VARIANT.search(PurePosixPath(filename).name)
    return match.group(1).lower() if match else None


def _logical_stem(filename: str) -> str:
    name = PurePosixPath(filename).name
    suffix = PurePosixPath(name).suffix
    stem = name[: -len(suffix)] if suffix else name
    stem = SHARD.sub("", f"{stem}{suffix}")
    if suffix and stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    stem = VARIANT.sub("", stem)
    return re.sub(r"[._-]+", "-", stem).strip("-").lower()


def _complete_shards(files: Sequence[WeightFile]) -> bool:
    matches = [SHARD.search(item.name) for item in files]
    if not matches or any(match is None for match in matches):
        return False
    totals = {int(match.group(2)) for match in matches if match is not None}
    indexes = {int(match.group(1)) for match in matches if match is not None}
    return len(totals) == 1 and indexes == set(range(1, next(iter(totals)) + 1))


def _index_files(
    component: str,
    candidates: Sequence[WeightFile],
    documents: Mapping[str, Any],
) -> list[str] | None:
    available = {item.name for item in candidates}
    for document_name, document in sorted(documents.items()):
        if not document_name.endswith(".index.json") or not isinstance(document, Mapping):
            continue
        weight_map = document.get("weight_map")
        if not isinstance(weight_map, Mapping):
            continue
        parent = PurePosixPath(document_name).parent
        referenced: list[str] = []
        for value in weight_map.values():
            filename = str(value)
            if filename not in available and str(parent) != ".":
                filename = (parent / filename).as_posix()
            if filename in available and _component(filename) == component:
                referenced.append(filename)
        selected = sorted(set(referenced))
        if selected:
            return selected
    return None


def _select_component_files(
    component: str,
    records: Sequence[WeightFile],
    documents: Mapping[str, Any],
    variant: str | None,
) -> tuple[list[str], str | None]:
    by_family: dict[str, list[WeightFile]] = {}
    for item in records:
        by_family.setdefault(_family(item.name), []).append(item)
    family = next(
        (name for name in ("safetensors", "pickle", "ckpt") if by_family.get(name)),
        None,
    )
    if family is None:
        return [], None
    candidates = by_family[family]
    requested_variant = variant.lower() if variant else None
    if requested_variant:
        selected_variant = [
            item for item in candidates if _variant(item.name) == requested_variant
        ]
        if not selected_variant:
            raise ModelMetadataError(
                f"component {component} has no {requested_variant} weight variant"
            )
        candidates = selected_variant
        source_variant: str | None = requested_variant
    else:
        non_variant = [item for item in candidates if _variant(item.name) is None]
        if non_variant:
            candidates = non_variant
            source_variant = None
        else:
            variants = sorted(
                {variant for item in candidates if (variant := _variant(item.name)) is not None}
            )
            if len(variants) != 1:
                raise ModelMetadataError(
                    f"component {component} has ambiguous weight variants: {variants}"
                )
            source_variant = variants[0]

    authoritative = _index_files(component, candidates, documents)
    if authoritative:
        return authoritative, source_variant

    logical: dict[tuple[str, str], list[WeightFile]] = {}
    for item in candidates:
        kind = "sharded" if SHARD.search(item.name) else "single"
        logical.setdefault((_logical_stem(item.name), kind), []).append(item)
    viable = [
        files
        for (_stem, kind), files in logical.items()
        if kind == "single" or _complete_shards(files)
    ]
    if not viable:
        raise ModelMetadataError(f"component {component} has an incomplete shard set")
    chosen = min(
        viable,
        key=lambda files: (
            -sum(item.size for item in files),
            len(files),
            tuple(sorted(item.name for item in files)),
        ),
    )
    return sorted(item.name for item in chosen), source_variant


def _select_weights_with_variants(
    siblings: Iterable[Any],
    *,
    documents: Mapping[str, Any] | None = None,
    variant: str | None = None,
    component_names: Sequence[str] | None = None,
) -> tuple[dict[str, list[str]], dict[str, str | None]]:
    documents = documents or {}
    grouped: dict[str, list[WeightFile]] = {}
    allowed = set(component_names or [])
    for item in _weight_records(siblings):
        component = _component(item.name)
        if allowed and component not in allowed:
            continue
        grouped.setdefault(component, []).append(item)
    selected: dict[str, list[str]] = {}
    variants: dict[str, str | None] = {}
    for component in sorted(grouped):
        files, source_variant = _select_component_files(
            component, grouped[component], documents, variant
        )
        if files:
            selected[component] = files
            variants[component] = source_variant
    return selected, variants


def _gguf_quant(filename: str) -> str:
    stem = PurePosixPath(filename).name.removesuffix(GGUF_SUFFIX)
    stem = re.sub(r"-\d{5}-of-\d{5}$", "", stem)
    matches = re.findall(
        r"(?:^|[-.])((?:I?Q\d(?:_[A-Z0-9]+)*)|BF16|F16|F32)(?=$|[-.])",
        stem.upper(),
    )
    return normalize_quant(matches[-1]) if matches else "UNKNOWN"


def _gguf_sets(siblings: Iterable[Any]) -> list[dict[str, Any]]:
    grouped: dict[str, list[WeightFile]] = {}
    for item in siblings:
        filename = _filename(item)
        if not filename.lower().endswith(GGUF_SUFFIX):
            continue
        key = re.sub(r"-\d{5}-of-\d{5}(?=\.gguf$)", "", filename, flags=re.I)
        grouped.setdefault(key, []).append(WeightFile(filename, _filesize(item)))
    result = []
    for key, files in grouped.items():
        files = sorted(files, key=lambda item: item.name)
        if any(SHARD.search(item.name) for item in files) and not _complete_shards(files):
            raise ModelMetadataError(f"incomplete GGUF shard set: {key}")
        result.append(
            {
                "quant": _gguf_quant(key),
                "file": files[0].name,
                "checkpoint_bytes": sum(item.size for item in files),
                "shard_set": [item.name for item in files],
            }
        )
    return sorted(result, key=lambda item: (item["quant"], item["file"]))


def select_gguf(
    siblings: Iterable[Any],
    *,
    selector: str | None = None,
    gguf_file: str | None = None,
) -> dict[str, Any] | None:
    available = _gguf_sets(siblings)
    if not available:
        return None
    selected: dict[str, Any] | None = None
    normalized_selector = normalize_quant(selector) if selector else None
    if gguf_file:
        selected = next(
            (
                item
                for item in available
                if gguf_file == item["file"] or gguf_file in item["shard_set"]
            ),
            None,
        )
        if selected is None:
            raise ModelMetadataError(f"GGUF file not found: {gguf_file}")
        reason = "explicit"
        normalized_selector = selected["quant"]
    elif normalized_selector:
        matching = [item for item in available if item["quant"] == normalized_selector]
        if not matching:
            raise ModelMetadataError(f"GGUF quant not found: {normalized_selector}")
        selected = min(
            matching,
            key=lambda item: (-item["checkpoint_bytes"], item["file"]),
        )
        reason = "selector"
    else:
        q4km = [item for item in available if item["quant"] == "Q4_K_M"]
        if q4km:
            selected = min(
                q4km,
                key=lambda item: (-item["checkpoint_bytes"], item["file"]),
            )
            reason = "default_q4km"
        else:
            selected = min(
                available,
                key=lambda item: (-item["checkpoint_bytes"], item["file"]),
            )
            reason = "largest"
    return {
        "selected_file": selected["file"],
        "selector": normalized_selector,
        "selection_reason": reason,
        "selected_shard_set": selected["shard_set"],
        "available_quants": available,
    }


def _config(documents: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = documents.get(name)
    return dict(value) if isinstance(value, Mapping) else {}


def _dtype(value: Any) -> str:
    normalized = str(value or "").lower().replace("torch.", "")
    if normalized in {"bfloat16", "bf16"}:
        return "bf16"
    if normalized in {"float16", "half", "fp16", "f16"}:
        return "fp16"
    if normalized in {"float32", "fp32", "f32"}:
        return "fp32"
    if "float8" in normalized or normalized in {"fp8", "f8"}:
        return "fp8"
    if normalized in {"int4", "uint4"}:
        return "int4"
    return "unknown"


def _dtype_bytes(value: str) -> float | None:
    return {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "F8": 1}.get(value.upper().split("_")[0])


def _quantization(config: Mapping[str, Any], tags: Sequence[str]) -> dict[str, Any]:
    text = config.get("text_config") or config.get("language_config") or {}
    raw = config.get("quantization_config") or (
        text.get("quantization_config") if isinstance(text, Mapping) else None
    )
    raw = dict(raw) if isinstance(raw, Mapping) else {}
    tag_text = " ".join(tags).lower()
    method = str(raw.get("quant_method") or raw.get("method") or "").lower()
    raw_text = json.dumps(raw, sort_keys=True).lower()
    if "awq" in method or "awq" in tag_text:
        normalized = "awq"
    elif "gptq" in method or "gptq" in tag_text:
        normalized = "gptq"
    elif "mxfp4" in method or "mxfp4" in raw_text or "mxfp4" in tag_text:
        normalized = "mxfp4"
    elif "bitsandbytes" in method or "bnb" in method or "bitsandbytes" in raw_text:
        normalized = "bnb"
    elif "fp8" in tag_text or "float8" in raw_text or '"num_bits": 8' in raw_text:
        normalized = "fp8"
    else:
        normalized = method or None
    bits = raw.get("bits")
    if bits is None:
        found = {int(value) for value in re.findall(r'"num_bits":\s*(\d+)', raw_text)}
        bits = next(iter(found)) if len(found) == 1 else None
    return {"method": normalized, "bits": bits, "config": raw}


def _modality(filenames: Sequence[str], info: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    if any(name.lower().endswith(GGUF_SUFFIX) for name in filenames):
        return "gguf"
    tags = " ".join(str(tag) for tag in info.get("tags") or []).lower()
    pipeline = str(info.get("pipeline_tag") or "").lower()
    metadata = f"{tags} {pipeline}"
    if any(
        value in metadata
        for value in (
            "image-to-3d",
            "text-to-3d",
            "world-model",
            "world model",
            "motion-generation",
        )
    ):
        return "research"
    if any(value in metadata for value in ("text-to-video", "image-to-video", "video-generation")):
        return "video"
    if any(value in metadata for value in ("text-to-image", "image-to-image", "image-generation")):
        return "image"
    if config.get("vision_config") or any(
        value in metadata for value in ("image-text", "vision-language", "visual")
    ):
        return "vlm"
    return "text"


def _attention_type(value: Any) -> str:
    normalized = str(value).lower()
    if "sliding" in normalized:
        return "sliding"
    if any(token in normalized for token in ("linear", "mamba", "recurrent")):
        return "linear"
    return "full"


def _text_arch(config: Mapping[str, Any], safetensors: Mapping[str, Any]) -> dict[str, Any] | None:
    text = config.get("text_config") or config.get("language_config") or config
    if not isinstance(text, Mapping):
        return None
    raw_hidden = text.get("hidden_size") or text.get("d_model")
    raw_layers = text.get("num_hidden_layers") or text.get("num_layers") or text.get("n_layer")
    raw_heads = text.get("num_attention_heads") or text.get("n_head")
    if raw_hidden is None or raw_layers is None or raw_heads is None:
        return None
    hidden, layers, heads = int(raw_hidden), int(raw_layers), int(raw_heads)
    kv_heads = int(text.get("num_key_value_heads") or text.get("num_kv_heads") or heads)
    head_dim = int(text.get("head_dim") or hidden // heads)
    raw_types = text.get("layer_types")
    if isinstance(raw_types, Sequence) and not isinstance(raw_types, (str, bytes)):
        attention_types = [_attention_type(value) for value in raw_types]
        if len(attention_types) != layers:
            raise ModelMetadataError(
                f"layer_types has {len(attention_types)} entries for {layers} layers"
            )
    else:
        attention_types = ["full"] * layers
    parameters = safetensors.get("parameters") or {}
    total_params = int(safetensors.get("total") or sum(parameters.values()) or 0) or None
    num_experts = text.get("num_experts") or text.get("num_local_experts")
    moe = (
        {
            "total_params": total_params,
            "active_params": None,
            "num_experts": int(num_experts),
        }
        if num_experts
        else None
    )
    return {
        "architectures": list(config.get("architectures") or text.get("architectures") or []),
        "hidden_size": hidden,
        "num_layers": layers,
        "num_heads": heads,
        "num_kv_heads": kv_heads,
        "head_dim": head_dim,
        "max_position_embeddings": int(
            text.get("max_position_embeddings") or text.get("n_positions") or 32768
        ),
        "attention_types": attention_types,
        "sliding_window": int(text["sliding_window"]) if text.get("sliding_window") else None,
        "linear_state_size": int(
            text.get("linear_state_size") or text.get("state_size") or hidden * 2
        ),
        "linear_state_source": "config"
        if text.get("linear_state_size") or text.get("state_size")
        else "fallback",
        "moe": moe,
        "hybrid": len(set(attention_types)) > 1 or "linear" in attention_types,
    }


def _component_order(model_index: Mapping[str, Any]) -> list[str]:
    return [
        key
        for key, value in model_index.items()
        if not key.startswith("_")
        and isinstance(value, list)
        and len(value) == 2
        and key not in {"scheduler", "tokenizer", "tokenizer_2", "feature_extractor"}
    ]


def _diffusion_arch(
    documents: Mapping[str, Any], components: Sequence[Mapping[str, Any]]
) -> dict[str, Any] | None:
    model_index = _config(documents, "model_index.json")
    if not model_index:
        return None
    order = _component_order(model_index)
    transformer_name = "transformer" if "transformer/config.json" in documents else "unet"
    transformer = _config(documents, f"{transformer_name}/config.json")
    vae = _config(documents, "vae/config.json")
    if not transformer and not vae:
        return None
    block_channels = vae.get("block_out_channels") or vae.get("dim_mult") or []
    spatial_factor = 2 ** (len(block_channels) - 1) if block_channels else 8
    temporal_factor = vae.get("temporal_compression_ratio")
    if not temporal_factor:
        temporal = vae.get("temporal_downsample") or vae.get("temperal_downsample")
        temporal_factor = 2 ** sum(bool(value) for value in temporal) if temporal else 1
    patch = transformer.get("patch_size") or 1
    if isinstance(patch, Sequence) and not isinstance(patch, (str, bytes)):
        temporal_patch = int(patch[0])
        spatial_patch = int(patch[-1])
    else:
        spatial_patch = int(patch)
        temporal_patch = int(
            transformer.get("temporal_patch_size") or transformer.get("patch_size_t") or 1
        )
    heads = transformer.get("num_attention_heads")
    attention_dim = transformer.get("attention_head_dim")
    hidden = (
        transformer.get("hidden_size") or transformer.get("inner_dim") or transformer.get("dim")
    )
    if hidden is None and heads and isinstance(attention_dim, int):
        hidden = int(heads) * attention_dim
    if hidden is None and transformer.get("block_out_channels"):
        hidden = max(transformer["block_out_channels"])
    component_map = {str(item["name"]): item for item in components}
    tx_component = component_map.get(transformer_name) or component_map.get("transformer")
    text_encoder_sizes = []
    for name in order:
        if not name.startswith("text_encoder"):
            continue
        encoder = _config(documents, f"{name}/config.json")
        size = (
            encoder.get("hidden_size") or encoder.get("d_model") or encoder.get("projection_dim")
        )
        if size is not None:
            text_encoder_sizes.append(int(size))
    return {
        "components_order": [name for name in order if name in component_map],
        "latent_channels": int(
            vae.get("latent_channels")
            or vae.get("z_dim")
            or transformer.get("out_channels")
            or transformer.get("in_channels")
            or 4
        ),
        "vae_spatial_factor": int(spatial_factor),
        "vae_temporal_factor": int(temporal_factor),
        "transformer": {
            "hidden_size": int(hidden) if hidden is not None else None,
            "num_layers": int(transformer["num_layers"])
            if transformer.get("num_layers") is not None
            else None,
            "patch_size": spatial_patch,
            "temporal_patch_size": temporal_patch,
            "params": tx_component.get("params") if tx_component else None,
        },
        "text_encoder_hidden_size": max(text_encoder_sizes) if text_encoder_sizes else 4096,
        "text_encoder_hidden_source": "config" if text_encoder_sizes else "fallback",
        "default_sample_size": transformer.get("sample_size"),
    }


def build_model_shape(
    repo: str,
    revision: str,
    info: Mapping[str, Any],
    documents: Mapping[str, Any],
    *,
    variant: str | None = None,
    gguf_selector: str | None = None,
    gguf_file: str | None = None,
) -> dict[str, Any]:
    siblings = list(info.get("siblings") or [])
    filenames = [_filename(item) for item in siblings]
    config = _config(documents, "config.json")
    model_index = _config(documents, "model_index.json")
    tags = list(info.get("tags") or [])
    safetensors = _jsonable(info.get("safetensors")) or {}
    quantization = _quantization(config, tags)
    gguf = select_gguf(siblings, selector=gguf_selector, gguf_file=gguf_file)
    modality = _modality(filenames, info, config)
    component_names = _component_order(model_index) if model_index else None
    selected, source_variants = _select_weights_with_variants(
        siblings,
        documents=documents,
        variant=variant,
        component_names=component_names,
    )
    size_by_name = {_filename(item): _filesize(item) for item in siblings}
    components: list[dict[str, Any]] = []
    repo_parameters = safetensors.get("parameters") or {}
    repo_param_count = int(safetensors.get("total") or sum(repo_parameters.values()) or 0) or None
    for name, files in selected.items():
        checkpoint_bytes = sum(size_by_name[filename] for filename in files)
        component_config = _config(
            documents, "config.json" if name == "root" else f"{name}/config.json"
        )
        dtype = _dtype(
            component_config.get("torch_dtype")
            or component_config.get("dtype")
            or source_variants.get(name)
        )
        family = _family(files[0])
        params = repo_param_count if name == "root" and len(selected) == 1 else None
        if (
            name == "root"
            and len(selected) == 1
            and family == "safetensors"
            and not quantization["method"]
            and not source_variants.get(name)
            and repo_parameters
        ):
            byte_total = 0.0
            exact = True
            for key, count in repo_parameters.items():
                width = _dtype_bytes(str(key))
                if width is None:
                    exact = False
                    break
                byte_total += int(count) * width
            resident = int(byte_total) if exact else checkpoint_bytes
            confidence = "high" if exact else "medium"
        elif family == "safetensors":
            resident = checkpoint_bytes
            confidence = "medium"
        else:
            resident = checkpoint_bytes
            confidence = "low"
        components.append(
            {
                "name": name,
                "checkpoint_bytes": checkpoint_bytes,
                "resident_weight_bytes": resident,
                "residency_confidence": confidence,
                "params": params,
                "dtype": dtype,
                "source_variant": source_variants.get(name),
                "files": files,
            }
        )
    if gguf:
        selected_gguf = next(
            item for item in gguf["available_quants"] if item["file"] == gguf["selected_file"]
        )
        components = [
            {
                "name": "root",
                "checkpoint_bytes": selected_gguf["checkpoint_bytes"],
                "resident_weight_bytes": selected_gguf["checkpoint_bytes"],
                "residency_confidence": "low",
                "params": None,
                "dtype": "unknown",
                "source_variant": None,
                "files": selected_gguf["shard_set"],
            }
        ]
        quantization = {
            "method": f"gguf:{selected_gguf['quant']}",
            "bits": None,
            "config": {},
        }
    if modality == "gguf":
        library = "gguf"
    elif model_index:
        library = "diffusers"
    elif config:
        library = "transformers"
    else:
        library = "unknown"
    ir = {
        "ir_version": IR_VERSION,
        "model": repo,
        "revision": revision,
        "sha_pinned": True,
        "modality": modality,
        "library": library,
        "pipeline_class": model_index.get("_class_name") if model_index else None,
        "components": components,
        "quantization": quantization,
        "text_arch": _text_arch(config, safetensors)
        if modality in {"text", "vlm", "gguf"}
        else None,
        "diffusion_arch": _diffusion_arch(documents, components)
        if modality in {"image", "video"}
        else None,
        "gguf": gguf,
        "sources": {
            "config": "config.json" if config else "model_index.json" if model_index else None,
            "component_configs": sorted(
                name for name in documents if name.endswith("/config.json")
            ),
            "gguf_parser": False,
            "safetensors_headers": False,
        },
    }
    return ir


def build_model_shape_from_fixture(
    fixture: Mapping[str, Any],
    *,
    variant: str | None = None,
    gguf_selector: str | None = None,
    gguf_file: str | None = None,
) -> dict[str, Any]:
    return build_model_shape(
        str(fixture["repo"]),
        str(fixture["revision"]),
        fixture["model_info"],
        fixture.get("files") or {},
        variant=variant,
        gguf_selector=gguf_selector,
        gguf_file=gguf_file,
    )


def _metadata_names(filenames: Sequence[str]) -> list[str]:
    return sorted(
        filename
        for filename in filenames
        if filename == "config.json"
        or filename == "model_index.json"
        or filename.endswith("/config.json")
        or filename.endswith(".index.json")
        or PurePosixPath(filename).name in {"quantize_config.json", "quantization_config.json"}
    )


def fetch_hub_snapshot(
    model: str,
    *,
    revision: str | None = None,
    token: str | None = None,
) -> HubSnapshot:
    """Fetch SHA-pinned metadata and small classification files, never weights."""
    import huggingface_hub
    from huggingface_hub import HfApi, hf_hub_download

    repo, _selector = split_model_selector(model)
    api = HfApi(token=token)
    info_object = api.model_info(repo, revision=revision, files_metadata=True)
    sha = str(getattr(info_object, "sha", "") or "")
    if not sha:
        raise ModelMetadataError(f"Hugging Face did not return a revision SHA for {repo}")
    info = _jsonable(info_object)
    siblings = list(info.get("siblings") or [])
    filenames = [_filename(item) for item in siblings]
    documents: dict[str, Any] = {}
    fetched: list[str] = []
    for filename in _metadata_names(filenames):
        if filename.lower().endswith(WEIGHT_SUFFIXES + (GGUF_SUFFIX,)):
            raise ModelMetadataError(f"refusing metadata fetch for weight: {filename}")
        path = hf_hub_download(
            repo_id=repo,
            filename=filename,
            revision=sha,
            token=token,
        )
        try:
            value = json.loads(Path(path).read_text())
        except json.JSONDecodeError as error:
            raise ModelMetadataError(f"invalid JSON metadata: {filename}") from error
        documents[filename] = value
        fetched.append(filename)
    return HubSnapshot(
        repo=repo,
        requested_revision=revision,
        revision=sha,
        model_info=info,
        documents=documents,
        huggingface_hub_version=huggingface_hub.__version__,
        fetched_files=tuple(fetched),
    )


__all__ = [
    "IR_VERSION",
    "HubSnapshot",
    "ModelMetadataError",
    "build_model_shape",
    "build_model_shape_from_fixture",
    "fetch_hub_snapshot",
    "normalize_quant",
    "select_gguf",
    "split_model_selector",
]
