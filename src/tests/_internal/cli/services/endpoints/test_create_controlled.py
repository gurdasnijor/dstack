"""End-to-end controlled-mode preset creation with a scripted agent.

The scripted agent drives the live controller socket exactly as the installed
`endpoint` client would: submit -> stop-handoff -> submit -> verify ->
finalize. The dstack API is faked at the controller-API boundary; the
verification/save path uses the real report validation and preset store.
"""

import asyncio
import json
from types import SimpleNamespace
from typing import Optional

import pytest

from dstack._internal.cli.models.endpoints import EndpointConfiguration
from dstack._internal.cli.services.endpoints.agent import (
    EndpointAgentProcessOutput,
    EndpointAgentSession,
)
from dstack._internal.cli.services.endpoints.controller import RunInfo
from dstack._internal.cli.services.endpoints.create import _create_endpoint_preset
from dstack._internal.cli.services.endpoints.rpc import (
    CONTROLLER_SOCKET_ENV,
    CONTROLLER_TOKEN_ENV,
)
from dstack._internal.cli.services.endpoints.session import load_events
from dstack._internal.cli.services.endpoints.store import EndpointPresetStore
from dstack._internal.core.errors import CLIError
from dstack._internal.core.models.runs import Run, RunStatus
from tests._internal.cli.endpoint_presets import (
    get_running_service_run,
    get_successful_endpoint_report,
)

# Unix sockets: POSIX only (unmarked tests are skipped on Windows).

SERVICE_YAML = """
type: service
image: vllm/vllm-openai:v0.8.0
commands:
  - vllm serve Qwen/Qwen3.5-27B
port: 8000
fleets: [gpu-fleet]
resources:
  gpu: 24GB
"""


class FakeControllerAPI:
    def __init__(self):
        self.runs: dict[str, RunInfo] = {}
        self.stopped: list[str] = []

    def submit(self, configuration: dict, run_name: str) -> RunInfo:
        info = RunInfo(
            name=run_name,
            run_id=f"id-{run_name}",
            status="running",
            is_finished=False,
            service_url=f"https://svc/{run_name}",
            price_per_hour=0.5,
            instance_id="instance-1",
        )
        self.runs[run_name] = info
        return info

    def get(self, run_name: str) -> Optional[RunInfo]:
        return self.runs.get(run_name)

    def stop(self, run_names, *, abort: bool = False) -> None:
        self.stopped.extend(run_names)
        for name in run_names:
            info = self.runs.get(name)
            if info is not None:
                self.runs[name] = RunInfo(
                    name=info.name,
                    run_id=info.run_id,
                    status="terminated",
                    is_finished=True,
                )

    def logs(self, run_name: str) -> str:
        return "logs"

    def offers(self, filters):
        return []

    def http_request(self, service_url, *, method, path, body, headers):
        return {"status": 200, "body_base64": ""}


async def _op(env: dict, op: str, params: dict) -> dict:
    reader, writer = await asyncio.open_unix_connection(env[CONTROLLER_SOCKET_ENV])
    request = {"token": env[CONTROLLER_TOKEN_ENV], "op": op, "params": params}
    writer.write((json.dumps(request) + "\n").encode())
    await writer.drain()
    writer.write_eof()
    response = json.loads((await reader.readline()).decode())
    writer.close()
    return response


def _agent_session(tmp_path) -> EndpointAgentSession:
    path = tmp_path / "session-running"
    path.mkdir()
    (path / "agent.log").touch()
    return EndpointAgentSession(path=path, timestamp="20260723-000000Z", debug=False)


@pytest.fixture
def context(tmp_path, monkeypatch):
    run = get_running_service_run()
    run_apis = _FakeModelRunAPIs(run)
    api = SimpleNamespace(
        project="main",
        runs=run_apis,
        client=SimpleNamespace(
            _token="dstack-secret",
            base_url="http://127.0.0.1:3000",
            runs=run_apis,
        ),
    )
    controller_api = FakeControllerAPI()
    monkeypatch.setattr(
        "dstack._internal.cli.services.endpoints.create.DstackControllerAPI",
        lambda *args, **kwargs: controller_api,
    )
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
    monkeypatch.delenv("DSTACK_ENDPOINT_LEGACY_AGENT_SHELL", raising=False)
    return SimpleNamespace(
        api=api,
        controller_api=controller_api,
        run=run,
        configuration=EndpointConfiguration(
            name="qwen-build",
            model={"base": "Qwen/Qwen3.5-27B"},
            context_length=8192,
            fleets=["gpu-fleet"],
            env={"LICENSE": "license-secret"},
        ),
        source_configuration=EndpointConfiguration(
            name="qwen-build",
            model={"base": "Qwen/Qwen3.5-27B"},
            context_length=8192,
            fleets=["gpu-fleet"],
            env=["LICENSE"],
        ),
        store=EndpointPresetStore(tmp_path / "presets"),
    )


class TestControlledCreateFlow:
    @pytest.mark.asyncio
    async def test_scripted_agent_creates_preset_through_controller(
        self, context, monkeypatch, tmp_path
    ):
        agent_session = _agent_session(tmp_path)
        report_data = json.loads(get_successful_endpoint_report(context.run).json())
        workload = {"context_length": 8192}

        async def scripted_agent(**kwargs):
            env = kwargs["env"]
            workspace = kwargs["workspace"]
            # Secret isolation: the agent shell receives no credentials.
            assert "DSTACK_TOKEN" not in env
            assert "DSTACK_ENDPOINT_BEARER_TOKEN" not in env
            assert "LICENSE" not in env
            assert "dstack-secret" not in json.dumps(env)
            assert not (workspace.bin_path / "dstack").exists()
            assert (workspace.bin_path / "endpoint").exists()
            assert "Controlled Mutation Surface" in kwargs["prompt"]

            first = await _op(
                env,
                "submit_candidate",
                {
                    "configuration_yaml": SERVICE_YAML,
                    "purpose": "prototype",
                    "expected_workload": workload,
                },
            )
            assert first["ok"], first
            assert first["result"]["run_name"] == "qwen-build-1"
            stopped = await _op(
                env,
                "stop_and_handoff",
                {"run_name": "qwen-build-1", "handoff_requirements": {"instance_reuse": True}},
            )
            assert stopped["ok"], stopped
            second = await _op(
                env,
                "submit_candidate",
                {
                    "configuration_yaml": SERVICE_YAML + "\n# final\n",
                    "purpose": "final service",
                    "expected_workload": workload,
                },
            )
            assert second["ok"], second
            assert second["result"]["run_name"] == "qwen-build-2"
            verified = await _op(
                env,
                "verify_final",
                {"run_name": "qwen-build-2", "declared_workload": workload},
            )
            assert verified["ok"], verified
            finalized = await _op(
                env,
                "finalize_preset",
                {"run_name": "qwen-build-2", "report_metadata": report_data},
            )
            assert finalized["ok"], finalized
            return EndpointAgentProcessOutput(report_data=report_data)

        monkeypatch.setattr(
            "dstack._internal.cli.services.endpoints.create.run_endpoint_agent",
            scripted_agent,
        )

        result = await _create_endpoint_preset(
            api=context.api,
            configuration=context.configuration,
            source_configuration=context.source_configuration,
            store=context.store,
            agent_session=agent_session,
        )

        assert result.preset.base == "Qwen/Qwen3.5-27B"
        assert result.final_run_name == "qwen-build-2"
        assert context.store.list() == [result.preset]
        assert "license-secret" not in result.path.read_text()
        # Cleanup stopped the final service because keep_service was not set.
        assert "qwen-build-2" in context.controller_api.stopped
        events = {event["event"] for event in load_events(agent_session.path)}
        assert {"run_submitted", "run_stopped", "final_verified", "preset_finalized"} <= events
        state = json.loads((agent_session.path / "state.json").read_text())
        assert state["finalized"] is True
        assert state["runs_used"] == 2

    @pytest.mark.asyncio
    async def test_success_report_without_finalization_is_a_failure(
        self, context, monkeypatch, tmp_path
    ):
        agent_session = _agent_session(tmp_path)
        report_data = json.loads(get_successful_endpoint_report(context.run).json())

        async def scripted_agent(**kwargs):
            env = kwargs["env"]
            submitted = await _op(
                env,
                "submit_candidate",
                {
                    "configuration_yaml": SERVICE_YAML,
                    "purpose": "prototype",
                    "expected_workload": {"context_length": 8192},
                },
            )
            assert submitted["ok"], submitted
            return EndpointAgentProcessOutput(report_data=report_data)

        monkeypatch.setattr(
            "dstack._internal.cli.services.endpoints.create.run_endpoint_agent",
            scripted_agent,
        )

        with pytest.raises(CLIError, match="without finalizing"):
            await _create_endpoint_preset(
                api=context.api,
                configuration=context.configuration,
                source_configuration=context.source_configuration,
                store=context.store,
                agent_session=agent_session,
            )

        # The orphaned prototype was stopped by controller cleanup.
        assert context.controller_api.stopped == ["qwen-build-1"]
        assert context.store.list() == []
        baseline = json.loads((agent_session.path / "baseline.json").read_text())
        assert baseline["succeeded"] is False

    @pytest.mark.asyncio
    async def test_agent_crash_still_cleans_up_runs(self, context, monkeypatch, tmp_path):
        agent_session = _agent_session(tmp_path)

        async def crashing_agent(**kwargs):
            env = kwargs["env"]
            submitted = await _op(
                env,
                "submit_candidate",
                {
                    "configuration_yaml": SERVICE_YAML,
                    "purpose": "prototype",
                    "expected_workload": {"context_length": 8192},
                },
            )
            assert submitted["ok"], submitted
            raise RuntimeError("claude process crashed")

        monkeypatch.setattr(
            "dstack._internal.cli.services.endpoints.create.run_endpoint_agent",
            crashing_agent,
        )

        with pytest.raises(RuntimeError, match="claude process crashed"):
            await _create_endpoint_preset(
                api=context.api,
                configuration=context.configuration,
                source_configuration=context.source_configuration,
                store=context.store,
                agent_session=agent_session,
            )

        assert context.controller_api.stopped == ["qwen-build-1"]
        baseline = json.loads((agent_session.path / "baseline.json").read_text())
        assert baseline["succeeded"] is False
        assert "claude process crashed" in baseline["failure_summary"]


class _FakeModelRunAPIs:
    def __init__(self, run: Run):
        self.run = run
        self.stopped_names: list[str] = []

    def get(self, *args):
        name = args[-1]
        return self.run if name == self.run.run_spec.run_name else None

    def stop(self, project, names, abort):
        self.stopped_names.extend(names)
        self.run.status = RunStatus.TERMINATED
