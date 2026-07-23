"""Evidence-ladder classification tests over the breadth-matrix fixtures."""

import json
from pathlib import Path

import pytest

from dstack._internal.cli.services.endpoints.inspect.classify import EvidenceLevel
from dstack._internal.cli.services.endpoints.inspect.service import inspection_from_snapshot

pytestmark = pytest.mark.windows

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _inspect(name: str, **kwargs):
    fixture = json.loads((FIXTURES / "models" / f"{name}.json").read_text())
    return inspection_from_snapshot(fixture, **kwargs)


def _candidate(inspection, runtime: str):
    matches = [item for item in inspection.candidates if item.runtime == runtime]
    assert len(matches) == 1, f"expected one {runtime} candidate"
    return matches[0]


class TestStandardText:
    def test_exact_recipe_ranks_first_and_carries_tool_parser_hints(self):
        inspection = _inspect("qwen-qwen2-5-7b-instruct")

        assert inspection.classification == "supported"
        assert inspection.modality == "text"
        first = inspection.candidates[0]
        assert first.runtime == "vllm"
        assert first.evidence_level == EvidenceLevel.EXACT_RECIPE
        assert first.confidence == "high"
        assert "--tool-call-parser hermes" in first.launch_hints
        # The recipe supersedes the vLLM registry match; one candidate per runtime.
        assert [item.runtime for item in inspection.candidates].count("vllm") == 1

    def test_registry_match_without_recipe_is_a_strong_candidate(self):
        inspection = _inspect("qwen-qwen3-8b")

        assert inspection.classification == "supported"
        vllm = _candidate(inspection, "vllm")
        assert vllm.evidence_level == EvidenceLevel.SUPPORT_REGISTRY
        assert vllm.confidence == "high"
        assert any("Qwen3ForCausalLM" in item.fact for item in vllm.evidence)
        sglang = _candidate(inspection, "sglang")
        assert sglang.evidence_level == EvidenceLevel.SUPPORT_REGISTRY
        assert inspection.candidates[0].evidence_level == EvidenceLevel.SUPPORT_REGISTRY
        assert all(candidate.smoke_test for candidate in inspection.candidates)

    def test_quantization_annotates_capability_floors_per_runtime(self):
        inspection = _inspect("neuralmagic-meta-llama-3-1-8b-instruct-fp8")

        vllm = _candidate(inspection, "vllm")
        assert vllm.min_compute_capability == 8.9  # fp8 native path
        sglang = _candidate(inspection, "sglang")
        assert any("no capability path recorded" in fact for fact in sglang.missing_facts)

    def test_unknown_architecture_downgrades_to_discovery_hint(self):
        fixture = json.loads((FIXTURES / "models" / "qwen-qwen2-5-7b-instruct.json").read_text())
        fixture["repo"] = "example/frontier-net-7b"
        fixture["files"]["config.json"]["architectures"] = ["FrontierNetForCausalLM"]

        inspection = inspection_from_snapshot(fixture)

        assert inspection.classification == "needs_research"
        for runtime in ("vllm", "sglang"):
            candidate = _candidate(inspection, runtime)
            assert candidate.evidence_level == EvidenceLevel.DISCOVERY_HINT
            assert candidate.confidence == "low"
            assert any("absent from the pinned" in fact for fact in candidate.missing_facts)
        assert any("matched no pinned support registry" in f for f in inspection.missing_facts)


class TestVisionLanguage:
    def test_vlm_architecture_matches_both_registries(self):
        inspection = _inspect("qwen-qwen2-5-vl-7b-instruct")

        assert inspection.modality == "vlm"
        assert inspection.classification == "supported"
        vllm = _candidate(inspection, "vllm")
        assert vllm.evidence_level == EvidenceLevel.SUPPORT_REGISTRY
        assert "image" in vllm.smoke_test["request"]


class TestGGUF:
    def test_gguf_yields_llamacpp_candidate_with_selection_evidence(self):
        inspection = _inspect("bartowski-qwen2-5-7b-instruct-gguf")

        assert inspection.modality == "gguf"
        assert inspection.classification == "supported"
        candidate = _candidate(inspection, "llama.cpp")
        assert candidate.evidence_level == EvidenceLevel.SUPPORT_REGISTRY
        assert any(
            "Q4_K_M" in item.fact or "gguf" in item.fact.lower() for item in candidate.evidence
        )
        assert any("gguf-parser" in fact for fact in candidate.missing_facts)
        assert any("GGUF on vLLM" in note for note in inspection.notes)

    def test_gguf_selector_changes_selected_quant(self):
        fixture = json.loads(
            (FIXTURES / "models" / "bartowski-qwen2-5-7b-instruct-gguf.json").read_text()
        )

        inspection = inspection_from_snapshot(fixture, gguf_selector="Q5_K_M")

        assert (inspection.ir.get("gguf") or {})["selector"] == "Q5_K_M"


class TestDiffusion:
    def test_sdxl_gets_adapter_candidates_requiring_smoke_tests(self):
        inspection = _inspect("stabilityai-stable-diffusion-xl-base-1-0")

        assert inspection.modality == "image"
        assert inspection.pipeline_class == "StableDiffusionXLPipeline"
        assert inspection.classification == "supported"
        omni = _candidate(inspection, "vllm-omni")
        assert omni.evidence_level == EvidenceLevel.GENERIC_ADAPTER
        assert "--omni" in omni.launch_hints
        assert any("unproven" in fact for fact in omni.missing_facts)
        diffusers = _candidate(inspection, "diffusers")
        assert diffusers.evidence_level == EvidenceLevel.GENERIC_ADAPTER
        assert diffusers.smoke_test["api"] == "images_generations"

    def test_native_wan_pipeline_is_a_registry_match(self):
        inspection = _inspect("wan-ai-wan2-2-t2v-a14b-diffusers")

        assert inspection.modality == "video"
        omni = _candidate(inspection, "vllm-omni")
        assert omni.evidence_level == EvidenceLevel.SUPPORT_REGISTRY
        assert inspection.candidates[0] is omni
        assert omni.smoke_test["api"] == "video_generation"

    def test_video_tags_without_model_index_are_a_negative_not_an_assumption(self):
        fixture = {
            "repo": "example/video-model-raw",
            "revision": "0" * 40,
            "model_info": {
                "pipeline_tag": "image-to-video",
                "library_name": "diffusers",
                "tags": ["diffusers", "image-to-video"],
                "siblings": [
                    {"rfilename": "transformer/model.safetensors", "size": 1000},
                ],
            },
            "files": {},
        }

        inspection = inspection_from_snapshot(fixture)

        assert inspection.modality == "video"
        assert inspection.classification == "needs_research"
        assert inspection.candidates == ()
        assert any(
            "DiffusionPipeline.from_pretrained" in negative.reason
            for negative in inspection.negative_results
        )
        assert any("model_index.json" in fact for fact in inspection.missing_facts)


class TestResearchAndUnsupported:
    def test_research_modality_requires_agent_research(self):
        inspection = _inspect("stabilityai-triposr")

        assert inspection.modality == "research"
        assert inspection.classification == "needs_research"
        assert inspection.candidates == ()

    def test_repository_without_weights_is_rejected_fast(self):
        fixture = {
            "repo": "example/not-a-model",
            "revision": "0" * 40,
            "model_info": {
                "tags": ["dataset"],
                "siblings": [{"rfilename": "README.md", "size": 10}],
            },
            "files": {},
        }

        inspection = inspection_from_snapshot(fixture)

        assert inspection.classification == "unsupported"
        assert inspection.candidates == ()
        assert any(
            "no loadable weight files" in negative.reason
            for negative in inspection.negative_results
        )
        assert any("reject quickly" in note for note in inspection.notes)


class TestEvidenceObjectShape:
    def test_to_data_is_json_serializable_and_carries_hardware_hint(self):
        inspection = _inspect("qwen-qwen2-5-7b-instruct")

        data = inspection.to_data()
        encoded = json.dumps(data)

        assert "candidates" in data and data["candidates"]
        assert data["hardware_hint"] is not None
        assert "resources" in data["hardware_hint"]
        assert "not a placement decision" in data["hardware_hint"]["note"]
        assert json.loads(encoded)["classification"] == "supported"

    def test_revision_is_the_immutable_fixture_sha(self):
        inspection = _inspect("qwen-qwen2-5-7b-instruct")

        assert len(inspection.revision) == 40
        assert inspection.ir["sha_pinned"] is True
