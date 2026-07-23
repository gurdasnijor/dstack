"""Tests for phase events and the workflow baseline summary."""

import json
from types import SimpleNamespace

import pytest

from dstack._internal.cli.models.endpoints import EndpointConfiguration
from dstack._internal.cli.services.endpoints.agent import (
    EndpointAgentProcessOutput,
    EndpointAgentSession,
)
from dstack._internal.cli.services.endpoints.create import _create_endpoint_preset
from dstack._internal.cli.services.endpoints.session import (
    EndpointSessionRecorder,
    load_events,
    write_baseline_summary,
)
from dstack._internal.core.errors import CLIError
from dstack._internal.core.models.runs import Run, RunStatus
from tests._internal.cli.endpoint_presets import (
    get_running_service_run,
    get_successful_endpoint_report,
)

pytestmark = pytest.mark.windows


def _agent_session(tmp_path) -> EndpointAgentSession:
    path = tmp_path / "session-running"
    path.mkdir()
    (path / "agent.log").touch()
    return EndpointAgentSession(path=path, timestamp="20260723-000000Z", debug=False)


class TestRecorder:
    def test_records_redacted_events_with_phase_durations(self, tmp_path):
        session = _agent_session(tmp_path)
        recorder = EndpointSessionRecorder(session, redacted_values=("hf-secret-value",))

        recorder.record("session_started", model="Qwen/Qwen2.5-7B-Instruct")
        recorder.phase_started("agent")
        recorder.phase_completed("agent", note="token hf-secret-value leaked")

        events = load_events(session.path)
        assert [event["event"] for event in events] == [
            "session_started",
            "phase_started",
            "phase_completed",
        ]
        assert events[2]["duration_seconds"] >= 0
        text = (session.path / "events.jsonl").read_text()
        assert "hf-secret-value" not in text
        assert "[redacted]" in text

    def test_fail_open_phases_closes_every_started_phase(self, tmp_path):
        recorder = EndpointSessionRecorder(_agent_session(tmp_path))
        recorder.phase_started("agent")
        recorder.phase_started("verification")

        recorder.fail_open_phases("boom")

        events = load_events(recorder.session.path)
        failed = {event["phase"] for event in events if event["event"] == "phase_failed"}
        assert failed == {"agent", "verification"}

    def test_write_failure_disables_events_without_raising(self, tmp_path):
        session = _agent_session(tmp_path)
        recorder = EndpointSessionRecorder(session)
        (session.path / "events.jsonl").mkdir()  # make appends fail

        recorder.record("session_started")
        recorder.phase_started("agent")


class TestBaselineSummary:
    def test_summarizes_phases_runs_and_failures(self, tmp_path):
        session = _agent_session(tmp_path)
        recorder = EndpointSessionRecorder(session)
        recorder.phase_started("inspection")
        recorder.phase_completed("inspection")
        recorder.phase_started("agent")
        recorder.phase_failed("agent", error="agent crashed")

        path = write_baseline_summary(
            recorder,
            submitted_run_names=["qwen-build-1", "qwen-build-2"],
            succeeded=False,
            failure_summary="agent crashed",
        )

        assert path is not None
        baseline = json.loads(path.read_text())
        assert baseline["succeeded"] is False
        assert baseline["failure_summary"] == "agent crashed"
        assert baseline["runs_submitted"] == 2
        assert baseline["phase_failures"] == 1
        assert set(baseline["phase_seconds"]) == {"inspection", "agent"}
        assert baseline["total_seconds"] >= 0
        # Cost baselines await billable runs captured by the controller.
        assert baseline["estimated_cost"] is None
        assert "controller" in baseline["estimated_cost_unavailable_reason"]


@pytest.fixture
def creation_context(tmp_path, monkeypatch):
    run = get_running_service_run()
    run_apis = _FakeRunAPIs(run)
    api = SimpleNamespace(
        project="main",
        runs=run_apis,
        client=SimpleNamespace(
            _token="dstack-secret",
            base_url="http://127.0.0.1:3000",
            runs=run_apis,
        ),
    )
    configuration = EndpointConfiguration(
        name="qwen-build",
        model={"base": "Qwen/Qwen3.5-27B"},
        context_length=8192,
        fleets=["gpu-fleet"],
    )
    # These flows exercise the legacy development shell path explicitly.
    monkeypatch.setenv("DSTACK_ENDPOINT_LEGACY_AGENT_SHELL", "1")
    monkeypatch.setattr(
        "dstack._internal.cli.services.endpoints.create.get_claude_auth",
        lambda: SimpleNamespace(
            api_key="anthropic-secret", executable="claude", effort=None, model="claude-test"
        ),
    )
    monkeypatch.setattr(
        "dstack._internal.cli.services.endpoints.create._get_build_name",
        lambda _: "qwen-build",
    )
    from dstack._internal.cli.services.endpoints.store import EndpointPresetStore

    return SimpleNamespace(
        api=api,
        configuration=configuration,
        run=run,
        store=EndpointPresetStore(tmp_path / "presets"),
    )


class TestCreateFlowInstrumentation:
    @pytest.mark.asyncio
    async def test_successful_flow_records_phases_and_baseline(
        self, creation_context, monkeypatch, tmp_path
    ):
        agent_session = _agent_session(tmp_path)

        async def run_agent(**kwargs):
            return EndpointAgentProcessOutput(
                report_data=json.loads(get_successful_endpoint_report(creation_context.run).json())
            )

        monkeypatch.setattr(
            "dstack._internal.cli.services.endpoints.create.run_endpoint_agent",
            run_agent,
        )

        await _create_endpoint_preset(
            api=creation_context.api,
            configuration=creation_context.configuration,
            store=creation_context.store,
            agent_session=agent_session,
        )

        events = load_events(agent_session.path)
        completed = [event["phase"] for event in events if event["event"] == "phase_completed"]
        assert completed == ["inspection", "agent", "verification", "cleanup"]
        assert events[0]["event"] == "session_started"
        baseline = json.loads((agent_session.path / "baseline.json").read_text())
        assert baseline["succeeded"] is True
        assert baseline["run_names"] == ["qwen-build-2"]
        assert "dstack-secret" not in (agent_session.path / "events.jsonl").read_text()

    @pytest.mark.asyncio
    async def test_failed_agent_records_failure_baseline_and_cleanup(
        self, creation_context, monkeypatch, tmp_path
    ):
        agent_session = _agent_session(tmp_path)

        async def run_agent(**kwargs):
            return EndpointAgentProcessOutput(error="Claude exploded")

        monkeypatch.setattr(
            "dstack._internal.cli.services.endpoints.create.run_endpoint_agent",
            run_agent,
        )

        with pytest.raises(CLIError, match="Claude exploded"):
            await _create_endpoint_preset(
                api=creation_context.api,
                configuration=creation_context.configuration,
                store=creation_context.store,
                agent_session=agent_session,
            )

        events = load_events(agent_session.path)
        names = [event["event"] for event in events]
        assert "phase_failed" in names
        assert names.count("phase_completed") >= 2  # inspection + cleanup
        baseline = json.loads((agent_session.path / "baseline.json").read_text())
        assert baseline["succeeded"] is False
        assert "Claude exploded" in baseline["failure_summary"]


class _FakeRunAPIs:
    def __init__(self, run: Run):
        self.run = run
        self.stopped_names: list[str] = []

    def get(self, *args):
        name = args[-1]
        return self.run if name == self.run.run_spec.run_name else None

    def stop(self, project, names, abort):
        self.stopped_names.extend(names)
        self.run.status = RunStatus.TERMINATED
