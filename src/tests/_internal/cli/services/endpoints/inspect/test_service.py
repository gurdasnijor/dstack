"""Tests for the `inspect_model` stage wiring into preset creation."""

import json
from pathlib import Path

import pytest

from dstack._internal.cli.models.endpoints import EndpointConfiguration
from dstack._internal.cli.services.endpoints.agent import (
    EndpointAgentSession,
    EndpointAgentWorkspace,
)
from dstack._internal.cli.services.endpoints.create import (
    _build_prompt,
    _run_inspection_stage,
)
from dstack._internal.cli.services.endpoints.inspect.hub import HubSnapshot
from dstack._internal.cli.services.endpoints.inspect.service import (
    _hub_repo_for_configuration,
    inspect_endpoint_model,
    inspection_from_snapshot,
)

pytestmark = pytest.mark.windows

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _configuration(**model) -> EndpointConfiguration:
    return EndpointConfiguration(name="qwen", model=model)


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / "models" / f"{name}.json").read_text())


class TestHubRepoDetection:
    def test_accepts_repo_ids_and_base_models(self):
        assert (
            _hub_repo_for_configuration(_configuration(repo="Qwen/Qwen2.5-7B-Instruct"))
            == "Qwen/Qwen2.5-7B-Instruct"
        )
        assert (
            _hub_repo_for_configuration(_configuration(base="Qwen/Qwen2.5-7B-Instruct"))
            == "Qwen/Qwen2.5-7B-Instruct"
        )
        assert (
            _hub_repo_for_configuration(
                _configuration(repo="Qwen/Qwen2.5-7B-Instruct", source="huggingface")
            )
            == "Qwen/Qwen2.5-7B-Instruct"
        )

    def test_rejects_urls_paths_and_non_hub_sources(self):
        assert _hub_repo_for_configuration(_configuration(repo="https://example.com/m")) is None
        assert _hub_repo_for_configuration(_configuration(repo="/models/local")) is None
        assert _hub_repo_for_configuration(_configuration(repo="./models/local")) is None
        assert _hub_repo_for_configuration(_configuration(repo="single-name")) is None
        assert (
            _hub_repo_for_configuration(
                _configuration(repo="Qwen/Qwen2.5-7B-Instruct", source="url")
            )
            is None
        )


class TestInspectEndpointModel:
    def test_network_failure_returns_error_not_exception(self, monkeypatch):
        def fail(*args, **kwargs):
            raise ConnectionError("hub unreachable")

        monkeypatch.setattr(
            "dstack._internal.cli.services.endpoints.inspect.service.fetch_hub_snapshot",
            fail,
        )

        result = inspect_endpoint_model(_configuration(repo="Qwen/Qwen2.5-7B-Instruct"))

        assert result.inspection is None
        assert result.error is not None
        assert "hub unreachable" in result.error

    def test_non_hub_source_is_skipped(self):
        result = inspect_endpoint_model(_configuration(repo="/models/local"))

        assert result.inspection is None
        assert result.skipped_reason is not None

    def test_snapshot_classification_round_trip(self, monkeypatch):
        fixture = _fixture("qwen-qwen2-5-7b-instruct")
        snapshot = HubSnapshot(
            repo=fixture["repo"],
            requested_revision=None,
            revision=fixture["revision"],
            model_info=fixture["model_info"],
            documents=fixture["files"],
            huggingface_hub_version="1.24.0",
            fetched_files=tuple(fixture["files"]),
        )
        monkeypatch.setattr(
            "dstack._internal.cli.services.endpoints.inspect.service.fetch_hub_snapshot",
            lambda *args, **kwargs: snapshot,
        )

        result = inspect_endpoint_model(_configuration(repo="Qwen/Qwen2.5-7B-Instruct"))

        assert result.error is None
        assert result.inspection is not None
        assert result.inspection.classification == "supported"
        assert result.snapshot is snapshot
        # The stored snapshot payload reproduces the same classification offline.
        replayed = inspection_from_snapshot(snapshot.to_data())
        assert replayed.to_data() == result.inspection.to_data()


class TestInspectionStage:
    @pytest.fixture
    def workspace(self, tmp_path) -> EndpointAgentWorkspace:
        workspace = EndpointAgentWorkspace(path=tmp_path / "w", dstack_home=tmp_path / "h")
        workspace.path.mkdir(parents=True)
        return workspace

    @pytest.fixture
    def agent_session(self, tmp_path) -> EndpointAgentSession:
        path = tmp_path / "session-running"
        path.mkdir()
        (path / "agent.log").touch()
        return EndpointAgentSession(path=path, timestamp="20260723-000000Z", debug=False)

    def test_persists_snapshot_and_evidence_and_returns_compact_object(
        self, monkeypatch, workspace, agent_session
    ):
        fixture = _fixture("qwen-qwen2-5-7b-instruct")
        snapshot = HubSnapshot(
            repo=fixture["repo"],
            requested_revision=None,
            revision=fixture["revision"],
            model_info=fixture["model_info"],
            documents=fixture["files"],
            huggingface_hub_version="1.24.0",
        )
        monkeypatch.setattr(
            "dstack._internal.cli.services.endpoints.inspect.service.fetch_hub_snapshot",
            lambda *args, **kwargs: snapshot,
        )

        inspection_data = _run_inspection_stage(
            configuration=_configuration(repo="Qwen/Qwen2.5-7B-Instruct"),
            workspace=workspace,
            agent_session=agent_session,
        )

        assert inspection_data is not None
        stored_snapshot = json.loads((agent_session.path / "inspection-snapshot.json").read_text())
        assert stored_snapshot["huggingface_hub_version"] == "1.24.0"
        assert stored_snapshot["revision"] == fixture["revision"]
        stored = json.loads((agent_session.path / "inspection.json").read_text())
        workspace_copy = json.loads((workspace.path / "inspection.json").read_text())
        assert stored == workspace_copy
        assert stored["classification"] == "supported"
        log = (agent_session.path / "agent.log").read_text()
        assert "Deterministic inspection pinned" in log

    def test_failure_keeps_creation_on_the_research_path(
        self, monkeypatch, workspace, agent_session
    ):
        def fail(*args, **kwargs):
            raise ConnectionError("offline")

        monkeypatch.setattr(
            "dstack._internal.cli.services.endpoints.inspect.service.fetch_hub_snapshot",
            fail,
        )

        inspection_data = _run_inspection_stage(
            configuration=_configuration(repo="Qwen/Qwen2.5-7B-Instruct"),
            workspace=workspace,
            agent_session=agent_session,
        )

        assert inspection_data is None
        assert not (workspace.path / "inspection.json").exists()
        assert "research path" in (agent_session.path / "agent.log").read_text()


class TestPromptEvidence:
    def test_prompt_contains_compact_evidence_without_ir(self):
        fixture = _fixture("qwen-qwen2-5-7b-instruct")
        inspection = inspection_from_snapshot(fixture)

        prompt = _build_prompt(
            configuration=_configuration(repo="Qwen/Qwen2.5-7B-Instruct"),
            build_name="qwen-build",
            allowed_fleets=("gpu-fleet",),
            inspection_data=inspection.to_data(),
        )

        assert "Deterministic model inspection (evidence object):" in prompt
        assert "exact_recipe" in prompt
        assert "Interpreting the inspection evidence object" in prompt
        assert '"ir"' not in prompt
        # The evidence block precedes the fixed-constraints block. The phrase
        # also occurs inside the system prompt body, so compare against the
        # final occurrence, which is the rendered block itself.
        assert prompt.index("Deterministic model inspection") < prompt.rindex(
            "Fixed endpoint constraints:"
        )

    def test_prompt_without_inspection_is_unchanged_in_shape(self):
        prompt = _build_prompt(
            configuration=_configuration(repo="Qwen/Qwen2.5-7B-Instruct"),
            build_name="qwen-build",
            allowed_fleets=("gpu-fleet",),
        )

        assert "Deterministic model inspection" not in prompt
        assert "Fixed endpoint constraints:" in prompt
