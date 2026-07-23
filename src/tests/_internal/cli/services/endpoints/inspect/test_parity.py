"""Golden parity tests against the dh reference implementation.

The goldens under ``fixtures/goldens/`` were produced by running dh (see
``fixtures/generate_goldens.py`` and ``goldens/provenance.json``). These tests
recompute the same observable outputs with the dstack port and require exact
equality: model shape IR, GGUF quant/file selection, gguf-parser output
normalization, runtime fingerprints, and deterministic frontier ordering.
"""

import json
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from dstack._internal.cli.services.endpoints.inspect import DH_SOURCE_COMMIT
from dstack._internal.cli.services.endpoints.inspect import gguf as gguf_module
from dstack._internal.cli.services.endpoints.inspect.hub import (
    build_model_shape_from_fixture,
    select_gguf,
)
from dstack._internal.cli.services.endpoints.inspect.runtime import (
    NATIVE_OMNI_PIPELINES,
    resolve_runtime,
)
from dstack._internal.cli.services.endpoints.inspect.sizing import (
    build_frontier,
    deterministic_sort_key,
    resolve_workload,
)

pytestmark = pytest.mark.windows

FIXTURES = Path(__file__).resolve().parent / "fixtures"
GOLDENS = FIXTURES / "goldens"

MODALITY_LANES = {
    "text": "general-llm",
    "vlm": "general-vlm",
    "image": "general-image",
    "video": "general-video",
    "gguf": "general-gguf",
    "research": None,
}


def _json_round_trip(value):
    return json.loads(json.dumps(value, sort_keys=True))


def _load_golden(name: str):
    return json.loads((GOLDENS / f"{name}.json").read_text())


def _load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / "models" / f"{name}.json").read_text())


def _model_fixture_names() -> list[str]:
    return sorted(
        path.stem for path in (FIXTURES / "models").glob("*.json") if path.stem != "manifest"
    )


def _tier_record(tier) -> dict:
    return {
        "id": tier.id,
        "strategy": tier.strategy,
        "gpu_count": tier.gpu_count,
        "vram_per_gpu_bytes": tier.vram_per_gpu_bytes,
        "host_ram_bytes": tier.host_ram_bytes,
        "disk_bytes": tier.disk_bytes,
        "min_compute_capability": tier.min_compute_capability,
        "capability_path": list(tier.capability_path) if tier.capability_path else None,
        "capability_preference_rank": tier.capability_preference_rank,
        "resources": dict(tier.resources),
        "confidence": tier.confidence,
        "notes": list(tier.notes),
        "launchable": tier.launchable,
        "star": tier.star,
    }


def _frontier_record(frontier) -> dict:
    launchable = [tier for tier in frontier.tiers if tier.launchable]
    return {
        "tiers": [_tier_record(tier) for tier in frontier.tiers],
        "star": frontier.star,
        "launchable_order": [tier.id for tier in sorted(launchable, key=deterministic_sort_key)],
    }


def test_goldens_were_generated_from_the_pinned_dh_commit():
    provenance = _load_golden("provenance")
    assert provenance["commit"] == DH_SOURCE_COMMIT
    assert provenance["reference"] == "github.com/gurdasnijor/dh"


@pytest.mark.parametrize("name", _model_fixture_names())
def test_model_shape_ir_matches_dh(name):
    golden = _load_golden("model_shapes")
    ir = build_model_shape_from_fixture(_load_fixture(name))
    assert _json_round_trip(ir) == golden[name]


def test_gguf_selection_matches_dh():
    golden = _load_golden("gguf_selection")
    for name, cases in golden.items():
        siblings = _load_fixture(name)["model_info"]["siblings"]
        for case, expected in cases.items():
            selector = "Q5_K_M" if case == "selector-q5km" else None
            selected = select_gguf(siblings, selector=selector)
            assert _json_round_trip(selected) == expected, f"{name}/{case}"


def test_gguf_parser_normalization_matches_dh():
    golden = _load_golden("gguf_normalization")
    parser_fixture = json.loads(
        (FIXTURES / "gguf-parser" / "qwen2.5-7b-q5-k-m-v0.25.0.json").read_text()
    )
    assert _json_round_trip(gguf_module.configurations(parser_fixture)) == golden


def test_runtime_fingerprints_match_dh():
    golden = _load_golden("runtime_fingerprints")
    for path in sorted((FIXTURES / "templates").glob("*.service.dstack.yml")):
        lane = path.name.replace(".service.dstack.yml", "")
        runtime = resolve_runtime(yaml.safe_load(path.read_text()), lane)
        record = {
            "engine": runtime.engine,
            "engine_key": runtime.engine_key,
            "image": runtime.image,
            "gpu_memory_utilization": runtime.gpu_memory_utilization,
            "kv_cache_dtype": runtime.kv_cache_dtype,
            "enforce_eager": runtime.enforce_eager,
            "chunked_prefill_tokens": runtime.chunked_prefill_tokens,
            "context": runtime.context,
            "pipeline_backend": runtime.pipeline_backend,
            "fingerprint": {
                "image": runtime.fingerprint.image,
                "engine_key": runtime.fingerprint.engine_key,
                "flags": list(runtime.fingerprint.flags),
                "key": runtime.fingerprint.key,
            },
        }
        assert _json_round_trip(record) == golden[lane], lane


@pytest.mark.parametrize("name", _model_fixture_names())
def test_deterministic_frontier_matches_dh(name):
    golden = _load_golden("frontiers")[name]
    ir = build_model_shape_from_fixture(_load_fixture(name))
    lane = MODALITY_LANES[ir["modality"]]
    assert golden["lane"] == lane
    pipeline_backend = None
    if lane in {"general-image", "general-video"}:
        pipeline_backend = (
            "native" if ir.get("pipeline_class") in NATIVE_OMNI_PIPELINES else "diffusers"
        )
    assert golden["pipeline_backend"] == pipeline_backend
    runtime = resolve_runtime(None, lane or "research", pipeline_backend=pipeline_backend)
    workload = resolve_workload(ir, runtime)
    assert _json_round_trip(asdict(workload)) == golden["workload"]
    frontier = build_frontier(ir, workload, runtime)
    assert _json_round_trip(_frontier_record(frontier)) == golden["cases"]["offline"]
    if "parser" in golden["cases"]:
        parser_fixture = json.loads(
            (FIXTURES / "gguf-parser" / "qwen2.5-7b-q5-k-m-v0.25.0.json").read_text()
        )
        parser_frontier = build_frontier(ir, workload, runtime, gguf_parser_result=parser_fixture)
        assert _json_round_trip(_frontier_record(parser_frontier)) == golden["cases"]["parser"]
