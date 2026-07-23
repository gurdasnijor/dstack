"""Controller mutation-boundary tests with a fake dstack API and fake clock.

Covers the required matrix: deadline expiration, budget exhaustion, duplicate
submission, concurrent mutation, failed stop, lost cache handoff, weakened
workload, agent crash, final-service disappearance, missing artifacts, secret
isolation, and cleanup failure.
"""

import json
import shutil
import threading
from typing import Optional

import pytest

from dstack._internal.cli.services.endpoints.agent import EndpointAgentSession
from dstack._internal.cli.services.endpoints.controller import (
    ControllerError,
    ControllerPolicy,
    EndpointController,
    RunInfo,
    workload_weaker_than,
)
from dstack._internal.cli.services.endpoints.session import EndpointSessionRecorder, load_events

pytestmark = pytest.mark.windows

SERVICE_YAML = """
type: service
image: vllm/vllm-openai:v0.8.0
commands:
  - vllm serve Qwen/Qwen2.5-7B-Instruct
port: 8000
fleets: [gpu-fleet]
resources:
  gpu: 24GB
"""


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeAPI:
    def __init__(self):
        self.runs: dict[str, RunInfo] = {}
        self.submitted: list[str] = []
        self.stopped: list[str] = []
        self.ignore_stop = False
        self.stop_raises: Optional[Exception] = None
        self.price_per_hour: Optional[float] = 2.0
        self.next_instance_id = "instance-1"
        self.http_calls: list[dict] = []

    def submit(self, configuration: dict, run_name: str) -> RunInfo:
        info = RunInfo(
            name=run_name,
            run_id=f"id-{run_name}",
            status="running",
            is_finished=False,
            service_url=f"https://svc/{run_name}",
            price_per_hour=self.price_per_hour,
            instance_id=self.next_instance_id,
        )
        self.runs[run_name] = info
        self.submitted.append(run_name)
        return info

    def get(self, run_name: str) -> Optional[RunInfo]:
        return self.runs.get(run_name)

    def stop(self, run_names, *, abort: bool = False) -> None:
        if self.stop_raises is not None:
            raise self.stop_raises
        self.stopped.extend(run_names)
        if self.ignore_stop:
            return
        for name in run_names:
            info = self.runs.get(name)
            if info is not None:
                self.runs[name] = RunInfo(
                    name=info.name,
                    run_id=info.run_id,
                    status="terminated",
                    is_finished=True,
                    service_url=None,
                    price_per_hour=info.price_per_hour,
                    instance_id=info.instance_id,
                )

    def terminate(self, run_name: str) -> None:
        info = self.runs[run_name]
        self.runs[run_name] = RunInfo(
            name=info.name,
            run_id=info.run_id,
            status="terminated",
            is_finished=True,
            service_url=None,
            price_per_hour=info.price_per_hour,
            instance_id=info.instance_id,
        )

    def logs(self, run_name: str) -> str:
        return f"logs for {run_name}"

    def offers(self, filters):
        return [{"backend": "runpod", "price": 0.5}]

    def http_request(self, service_url, *, method, path, body, headers):
        self.http_calls.append(
            {
                "service_url": service_url,
                "method": method,
                "path": path,
                "headers": dict(headers),
            }
        )
        return {"status": 200, "body_base64": ""}


@pytest.fixture
def context(tmp_path):
    session_path = tmp_path / "session-running"
    session_path.mkdir()
    (session_path / "agent.log").touch()
    session = EndpointAgentSession(path=session_path, timestamp="20260723-000000Z", debug=False)
    recorder = EndpointSessionRecorder(session, redacted_values=("dstack-secret",))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    clock = FakeClock()
    api = FakeAPI()

    def build(policy: Optional[ControllerPolicy] = None, finalize=None) -> EndpointController:
        return EndpointController(
            policy=policy
            or ControllerPolicy(
                build_name="qwen-build",
                allowed_fleets=("gpu-fleet",),
                declared_workload={"modality": "text-generation", "context_length": 8192},
            ),
            api=api,
            recorder=recorder,
            artifacts_dir=session_path / "artifacts",
            workspace_dir=workspace,
            submissions_path=workspace / "submissions.jsonl",
            finalize=finalize,
            clock=clock,
            sleep=lambda seconds: clock.advance(seconds),
        )

    class Context:
        pass

    ctx = Context()
    ctx.session = session
    ctx.recorder = recorder
    ctx.workspace = workspace
    ctx.clock = clock
    ctx.api = api
    ctx.build = build
    return ctx


def _workload(**extra):
    return {"modality": "text-generation", "context_length": 8192, **extra}


def _submit(controller, yaml_text: str = SERVICE_YAML, purpose: str = "candidate"):
    return controller.submit_candidate(
        configuration_yaml=yaml_text,
        purpose=purpose,
        expected_workload=_workload(),
    )


class TestSubmitPolicy:
    def test_assigns_run_names_and_packages_artifacts(self, context):
        controller = context.build()

        result = _submit(controller)

        assert result["run_name"] == "qwen-build-1"
        assert context.api.submitted == ["qwen-build-1"]
        artifact = context.session.path / "artifacts" / "qwen-build-1" / "configuration.yml"
        assert artifact.read_text() == SERVICE_YAML
        submissions = (context.workspace / "submissions.jsonl").read_text()
        assert "qwen-build-1" in submissions
        state = json.loads((context.session.path / "state.json").read_text())
        assert state["runs_used"] == 1
        assert state["active_runs"] == ["qwen-build-1"]

    def test_deadline_expiration_refuses_submission(self, context):
        controller = context.build(
            ControllerPolicy(
                build_name="qwen-build",
                allowed_fleets=("gpu-fleet",),
                total_budget_seconds=60,
            )
        )
        context.clock.advance(61)

        with pytest.raises(ControllerError, match="deadline expired") as error:
            _submit(controller)

        assert error.value.failure_class == "budget"
        diagnostics = (context.session.path / "diagnostics.jsonl").read_text()
        assert "deadline expired" in diagnostics

    def test_run_count_budget_exhaustion(self, context):
        controller = context.build(
            ControllerPolicy(build_name="qwen-build", allowed_fleets=("gpu-fleet",), max_runs=1)
        )
        _submit(controller)
        controller.stop_and_handoff(run_name="qwen-build-1")

        with pytest.raises(ControllerError, match="run budget exhausted") as error:
            _submit(controller, SERVICE_YAML + "\n# retry")

        assert error.value.failure_class == "budget"

    def test_cost_budget_exhaustion(self, context):
        controller = context.build(
            ControllerPolicy(
                build_name="qwen-build",
                allowed_fleets=("gpu-fleet",),
                cost_budget=1.0,
                max_runs=5,
            )
        )
        _submit(controller)  # $2/hour
        context.clock.advance(3600)  # accrued ~$2 > budget $1
        controller.stop_and_handoff(run_name="qwen-build-1")

        with pytest.raises(ControllerError, match="cost budget exhausted") as error:
            _submit(controller, SERVICE_YAML + "\n# second")

        assert error.value.failure_class == "budget"
        assert controller.estimated_cost() >= 1.0

    def test_single_active_prototype_is_enforced(self, context):
        controller = context.build()
        _submit(controller)

        with pytest.raises(ControllerError, match="still active"):
            _submit(controller, SERVICE_YAML + "\n# second")

    def test_duplicate_submission_is_refused(self, context):
        controller = context.build()
        _submit(controller)
        controller.stop_and_handoff(run_name="qwen-build-1")

        with pytest.raises(ControllerError, match="duplicate submission") as error:
            _submit(controller)

        assert error.value.failure_class == "validation"

    def test_fleet_and_price_constraints_are_enforced(self, context):
        controller = context.build(
            ControllerPolicy(
                build_name="qwen-build",
                allowed_fleets=("gpu-fleet",),
                max_price=1.0,
            )
        )
        bad_fleet = SERVICE_YAML.replace("gpu-fleet", "other-fleet")
        with pytest.raises(ControllerError, match="not allowed"):
            _submit(controller, bad_fleet)

        with pytest.raises(ControllerError, match="max_price"):
            _submit(controller)  # omits max_price while the endpoint caps it

        priced = SERVICE_YAML + "max_price: 0.9\n"
        assert _submit(controller, priced)["run_name"] == "qwen-build-1"

    def test_weakened_workload_is_refused_at_submission(self, context):
        controller = context.build()

        with pytest.raises(ControllerError, match="weaker") as error:
            controller.submit_candidate(
                configuration_yaml=SERVICE_YAML,
                purpose="weak",
                expected_workload={"modality": "text-generation", "context_length": 4096},
            )

        assert error.value.failure_class == "validation"

    def test_concurrent_mutation_is_serialized(self, context):
        controller = context.build()
        results: list = []
        errors: list = []

        def submit(suffix):
            try:
                results.append(_submit(controller, SERVICE_YAML + f"\n# {suffix}"))
            except ControllerError as e:
                errors.append(e)

        threads = [threading.Thread(target=submit, args=(index,)) for index in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(results) == 1
        assert len(errors) == 1
        assert "still active" in str(errors[0])
        state = json.loads((context.session.path / "state.json").read_text())
        assert state["runs_used"] == 1


class TestStopAndHandoff:
    def test_waits_for_authoritative_terminal_state(self, context):
        controller = context.build()
        _submit(controller)

        result = controller.stop_and_handoff(run_name="qwen-build-1")

        assert result["status"] == "terminated"
        assert context.api.stopped == ["qwen-build-1"]

    def test_failed_stop_is_a_bounded_diagnostic_error(self, context):
        controller = context.build(
            ControllerPolicy(
                build_name="qwen-build",
                allowed_fleets=("gpu-fleet",),
                stop_timeout_seconds=30,
            )
        )
        _submit(controller)
        context.api.ignore_stop = True

        with pytest.raises(ControllerError, match="failed stop") as error:
            controller.stop_and_handoff(run_name="qwen-build-1")

        assert error.value.failure_class == "controller"
        assert (
            "did not reach a terminal state"
            in (context.session.path / "diagnostics.jsonl").read_text()
        )
        # The run is still tracked as active, so another submit is refused.
        with pytest.raises(ControllerError, match="still active"):
            _submit(controller, SERVICE_YAML + "\n# next")

    def test_lost_cache_handoff_is_detected_and_recorded(self, context):
        controller = context.build()
        _submit(controller)
        controller.stop_and_handoff(
            run_name="qwen-build-1",
            handoff_requirements={"instance_reuse": True, "volumes": ["cache"]},
        )
        context.api.next_instance_id = "instance-2"  # reuse fails

        result = _submit(controller, SERVICE_YAML + "\n# second")
        controller.get_run_status(run_name=result["run_name"])

        events = load_events(context.session.path)
        handoff = [event for event in events if event["event"] == "handoff_result"]
        assert len(handoff) == 1
        assert handoff[0]["result"] == "lost"
        assert handoff[0]["from_run"] == "qwen-build-1"

    def test_preserved_handoff_is_recorded_as_reused(self, context):
        controller = context.build()
        _submit(controller)
        controller.stop_and_handoff(
            run_name="qwen-build-1", handoff_requirements={"instance_reuse": True}
        )

        result = _submit(controller, SERVICE_YAML + "\n# second")
        controller.get_run_status(run_name=result["run_name"])

        events = load_events(context.session.path)
        handoff = [event for event in events if event["event"] == "handoff_result"]
        assert handoff[0]["result"] == "reused"


class TestVerifyAndFinalize:
    def _finalize(self, run_name, report_metadata):
        return {"preset_id": "abcd1234", "path": "/presets/abcd1234.yaml"}

    def test_verify_requires_running_service_and_declared_workload(self, context):
        controller = context.build()
        _submit(controller)

        with pytest.raises(ControllerError, match="weaker"):
            controller.verify_final(
                run_name="qwen-build-1",
                declared_workload={"modality": "text-generation", "context_length": 2048},
            )

        result = controller.verify_final(run_name="qwen-build-1", declared_workload=_workload())
        assert result["service_url"] == "https://svc/qwen-build-1"

    def test_verified_final_cannot_be_stopped_before_finalize(self, context):
        controller = context.build(finalize=self._finalize)
        _submit(controller)
        controller.verify_final(run_name="qwen-build-1", declared_workload=_workload())

        with pytest.raises(ControllerError, match="must stay running"):
            controller.stop_and_handoff(run_name="qwen-build-1")

        controller.finalize_preset(run_name="qwen-build-1", report_metadata={"success": True})
        assert controller.finalized_result == {
            "preset_id": "abcd1234",
            "path": "/presets/abcd1234.yaml",
        }

    def test_final_service_disappearance_fails_finalize(self, context):
        controller = context.build(finalize=self._finalize)
        _submit(controller)
        controller.verify_final(run_name="qwen-build-1", declared_workload=_workload())
        context.api.terminate("qwen-build-1")

        with pytest.raises(ControllerError, match="no longer running") as error:
            controller.finalize_preset(run_name="qwen-build-1", report_metadata={})

        assert error.value.failure_class == "runtime"
        assert "disappeared" in (context.session.path / "diagnostics.jsonl").read_text()

    def test_finalize_requires_verify_first(self, context):
        controller = context.build(finalize=self._finalize)
        _submit(controller)

        with pytest.raises(ControllerError, match="verify_final"):
            controller.finalize_preset(run_name="qwen-build-1", report_metadata={})

    def test_missing_artifacts_fail_finalize(self, context):
        controller = context.build(finalize=self._finalize)
        _submit(controller)
        controller.verify_final(run_name="qwen-build-1", declared_workload=_workload())
        shutil.rmtree(context.session.path / "artifacts")

        with pytest.raises(ControllerError, match="artifacts are missing") as error:
            controller.finalize_preset(run_name="qwen-build-1", report_metadata={})

        assert error.value.failure_class == "packaging"

    def test_no_mutations_after_finalize(self, context):
        controller = context.build(finalize=self._finalize)
        _submit(controller)
        controller.verify_final(run_name="qwen-build-1", declared_workload=_workload())
        controller.finalize_preset(run_name="qwen-build-1", report_metadata={})

        with pytest.raises(ControllerError, match="already finalized"):
            _submit(controller, SERVICE_YAML + "\n# extra")


class TestPackaging:
    def test_referenced_files_are_copied_before_submission(self, context):
        (context.workspace / "server.py").write_text("print('hi')\n")
        yaml_text = SERVICE_YAML + "files:\n  - local_path: server.py\n"
        controller = context.build()

        _submit(controller, yaml_text)

        packaged = context.session.path / "artifacts" / "qwen-build-1" / "server.py"
        assert packaged.read_text() == "print('hi')\n"

    def test_missing_referenced_file_refuses_submission(self, context):
        yaml_text = SERVICE_YAML + "files:\n  - local_path: missing.py\n"
        controller = context.build()

        with pytest.raises(ControllerError, match="does not exist") as error:
            _submit(controller, yaml_text)

        assert error.value.failure_class == "packaging"
        assert context.api.submitted == []


class TestCleanupAndCrash:
    def test_agent_crash_cleanup_stops_all_active_runs(self, context):
        controller = context.build()
        _submit(controller)
        controller.fail_session("agent crashed", failure_class="controller")

        result = controller.cleanup(keep_final=False)

        assert result["stopped"] == ["qwen-build-1"]
        assert context.api.stopped == ["qwen-build-1"]
        diagnostics = (context.session.path / "diagnostics.jsonl").read_text()
        assert "agent crashed" in diagnostics

    def test_cleanup_keeps_only_a_finalized_final_service(self, context):
        controller = context.build(finalize=lambda *_: {"preset_id": "x"})
        _submit(controller)
        controller.verify_final(run_name="qwen-build-1", declared_workload=_workload())
        controller.finalize_preset(run_name="qwen-build-1", report_metadata={})

        result = controller.cleanup(keep_final=True)

        assert result["kept_final"] == "qwen-build-1"
        assert context.api.stopped == []

    def test_cleanup_failure_is_reported_with_diagnostics(self, context):
        controller = context.build()
        _submit(controller)
        context.api.stop_raises = RuntimeError("stop API is down")

        with pytest.raises(ControllerError, match="cleanup failed") as error:
            controller.cleanup(keep_final=False)

        assert error.value.failure_class == "controller"
        assert "stop API is down" in (context.session.path / "diagnostics.jsonl").read_text()


class TestReadOnlyHelpers:
    def test_context_exposes_policy_and_state_without_secrets(self, context):
        controller = context.build()
        _submit(controller)

        data = controller.get_endpoint_context()

        assert data["policy"]["allowed_fleets"] == ["gpu-fleet"]
        assert data["state"]["runs_used"] == 1
        assert "dstack-secret" not in json.dumps(data)

    def test_brokered_http_strips_agent_supplied_auth_headers(self, context):
        controller = context.build()
        _submit(controller)

        controller.request_service_http(
            run_name="qwen-build-1",
            method="POST",
            path="/v1/chat/completions",
            headers={"Authorization": "Bearer fake", "Content-Type": "application/json"},
        )

        call = context.api.http_calls[0]
        assert "Authorization" not in call["headers"]
        assert call["headers"]["Content-Type"] == "application/json"
        assert call["service_url"] == "https://svc/qwen-build-1"


class TestWorkloadComparison:
    def test_numeric_and_string_comparisons(self):
        declared = {"modality": "video-generation", "frames": 49, "steps": 25}
        assert (
            workload_weaker_than(
                declared, {"modality": "video-generation", "frames": 49, "steps": 25}
            )
            is None
        )
        assert (
            workload_weaker_than(
                declared, {"modality": "video-generation", "frames": 64, "steps": 30}
            )
            is None
        )
        assert "frames" in workload_weaker_than(
            declared, {"modality": "video-generation", "frames": 25, "steps": 25}
        )
        assert "modality" in workload_weaker_than(
            declared, {"modality": "image-generation", "frames": 49, "steps": 25}
        )
        assert "steps" in workload_weaker_than(
            declared, {"modality": "video-generation", "frames": 49}
        )
