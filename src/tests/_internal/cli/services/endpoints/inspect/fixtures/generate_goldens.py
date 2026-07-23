"""Regenerate golden parity fixtures from the dh reference implementation.

The golden files under ``goldens/`` are produced by running the *reference*
implementation (gurdasnijor/dh) against the captured fixtures in this
directory. The dstack port under
``dstack._internal.cli.services.endpoints.inspect`` must reproduce these
observable outputs exactly; ``test_parity.py`` enforces it. The goldens
deliberately record inputs->outputs only, never dh's internal representations.

Usage (requires a checkout of dh and its virtualenv; no network is used):

    /path/to/dh/.venv/bin/python generate_goldens.py /path/to/dh

The dh commit is recorded in ``goldens/provenance.json`` automatically.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import yaml

FIXTURES = Path(__file__).resolve().parent
GOLDENS = FIXTURES / "goldens"

MODALITY_LANES = {
    "text": "general-llm",
    "vlm": "general-vlm",
    "image": "general-image",
    "video": "general-video",
    "gguf": "general-gguf",
    "research": None,
}

GGUF_SELECTIONS = {
    "bartowski-qwen2-5-7b-instruct-gguf": [
        {"case": "default", "selector": None, "gguf_file": None},
        {"case": "selector-q5km", "selector": "Q5_K_M", "gguf_file": None},
    ],
    "unsloth-qwen3-30b-a3b-gguf": [
        {"case": "default", "selector": None, "gguf_file": None},
    ],
}

PARSER_FIXTURE = "qwen2.5-7b-q5-k-m-v0.25.0.json"


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


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
        "resources": _jsonable(dict(tier.resources)),
        "confidence": tier.confidence,
        "notes": list(tier.notes),
        "launchable": tier.launchable,
        "star": tier.star,
    }


def _frontier_record(frontier, deterministic_sort_key) -> dict:
    launchable = [tier for tier in frontier.tiers if tier.launchable]
    return {
        "tiers": [_tier_record(tier) for tier in frontier.tiers],
        "star": frontier.star,
        "launchable_order": [tier.id for tier in sorted(launchable, key=deterministic_sort_key)],
    }


def _workload_record(workload) -> dict:
    return _jsonable(asdict(workload))


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: generate_goldens.py /path/to/dh")
    dh_root = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(dh_root))

    from dh import frontier as frontier_module
    from dh import gguf as gguf_module
    from dh import hub as hub_module
    from dh import model as model_module
    from dh import runtime as runtime_module

    GOLDENS.mkdir(exist_ok=True)

    commit = subprocess.run(
        ["git", "-C", str(dh_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    (GOLDENS / "provenance.json").write_text(
        json.dumps(
            {
                "reference": "github.com/gurdasnijor/dh",
                "commit": commit,
                "behaviors": [
                    "model shape IR",
                    "GGUF quant/file selection",
                    "gguf-parser output normalization",
                    "runtime fingerprints",
                    "deterministic frontier ordering",
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    # 1. Model Shape IR parity.
    model_shapes = {}
    for name in _model_fixture_names():
        model_shapes[name] = hub_module.build_model_shape_from_fixture(_load_fixture(name))
    (GOLDENS / "model_shapes.json").write_text(
        json.dumps(_jsonable(model_shapes), indent=2, sort_keys=True) + "\n"
    )

    # 2. GGUF quant/file selection parity.
    gguf_selection = {}
    for name, cases in GGUF_SELECTIONS.items():
        siblings = _load_fixture(name)["model_info"]["siblings"]
        gguf_selection[name] = {
            case["case"]: hub_module.select_gguf(
                siblings, selector=case["selector"], gguf_file=case["gguf_file"]
            )
            for case in cases
        }
    (GOLDENS / "gguf_selection.json").write_text(
        json.dumps(_jsonable(gguf_selection), indent=2, sort_keys=True) + "\n"
    )

    # 3. gguf-parser output normalization parity.
    parser_fixture = json.loads((FIXTURES / "gguf-parser" / PARSER_FIXTURE).read_text())
    (GOLDENS / "gguf_normalization.json").write_text(
        json.dumps(_jsonable(gguf_module.configurations(parser_fixture)), indent=2, sort_keys=True)
        + "\n"
    )

    # 4. Runtime fingerprint parity for every captured template.
    fingerprints = {}
    for path in sorted((FIXTURES / "templates").glob("*.service.dstack.yml")):
        lane = path.name.replace(".service.dstack.yml", "")
        runtime = runtime_module.resolve_runtime(yaml.safe_load(path.read_text()), lane)
        fingerprints[lane] = {
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
    (GOLDENS / "runtime_fingerprints.json").write_text(
        json.dumps(_jsonable(fingerprints), indent=2, sort_keys=True) + "\n"
    )

    # 5. Deterministic frontier ordering parity for every fixture on its lane
    #    defaults. Lane defaults (no template) are what the endpoint
    #    `inspect_model` stage resolves; offer enrichment is deliberately out
    #    of scope.
    frontiers = {}
    for name in _model_fixture_names():
        ir = hub_module.build_model_shape_from_fixture(_load_fixture(name))
        lane = MODALITY_LANES[ir["modality"]]
        pipeline_backend = None
        if lane in {"general-image", "general-video"}:
            pipeline_backend = (
                "native"
                if ir.get("pipeline_class") in runtime_module.NATIVE_OMNI_PIPELINES
                else "diffusers"
            )
        runtime = runtime_module.resolve_runtime(
            None, lane or "research", pipeline_backend=pipeline_backend
        )
        workload = model_module.resolve_workload(ir, runtime)
        cases = {
            "offline": _frontier_record(
                frontier_module.build_frontier(ir, workload, runtime),
                frontier_module.deterministic_sort_key,
            )
        }
        if name == "bartowski-qwen2-5-7b-instruct-gguf":
            cases["parser"] = _frontier_record(
                frontier_module.build_frontier(
                    ir, workload, runtime, gguf_parser_result=parser_fixture
                ),
                frontier_module.deterministic_sort_key,
            )
        frontiers[name] = {
            "lane": lane,
            "pipeline_backend": pipeline_backend,
            "workload": _workload_record(workload),
            "cases": cases,
        }
    (GOLDENS / "frontiers.json").write_text(
        json.dumps(_jsonable(frontiers), indent=2, sort_keys=True) + "\n"
    )

    print(f"goldens regenerated from dh @ {commit}")


if __name__ == "__main__":
    main()
