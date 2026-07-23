"""Deterministic, policy-enforcing controller for preset-creation mutations.

The parent ``dstack endpoint preset create`` process owns this controller and
its authoritative session state. The agent plans experiments but can mutate
run lifecycle only through the four typed operations —
``submit_candidate``, ``stop_and_handoff``, ``verify_final``, and
``finalize_preset`` — plus read-only helpers. Every mutation takes the state
lock, re-checks policy against the current state, and persists the updated
state atomically before returning.

The controller is unit-testable: the dstack API, clock, and sleep are
injected, and the RPC transport lives in ``rpc.py``.
"""

import hashlib
import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

import yaml

from dstack._internal.cli.services.endpoints.session import EndpointSessionRecorder

_STATE_FILENAME = "state.json"
_DIAGNOSTICS_FILENAME = "diagnostics.jsonl"

# Failure classification for structured diagnostics.
FAILURE_CLASSES = (
    "model",
    "runtime",
    "capacity",
    "budget",
    "validation",
    "packaging",
    "controller",
)


class ControllerError(Exception):
    """A refused or failed controller operation, with a failure class."""

    def __init__(self, message: str, *, failure_class: str = "controller"):
        assert failure_class in FAILURE_CLASSES
        super().__init__(message)
        self.failure_class = failure_class


@dataclass(frozen=True)
class RunInfo:
    """Authoritative run facts reported by the dstack API adapter."""

    name: str
    run_id: str
    status: str  # dstack RunStatus value, e.g. "running", "terminated"
    is_finished: bool
    service_url: Optional[str] = None
    price_per_hour: Optional[float] = None
    instance_id: Optional[str] = None


class ControllerAPI(Protocol):
    """The narrow dstack surface the controller mutates through."""

    def submit(self, configuration: dict, run_name: str) -> RunInfo: ...

    def get(self, run_name: str) -> Optional[RunInfo]: ...

    def stop(self, run_names: Sequence[str], *, abort: bool = False) -> None: ...

    def logs(self, run_name: str) -> str: ...

    def offers(self, filters: Mapping[str, Any]) -> list[dict]: ...

    def http_request(
        self,
        service_url: str,
        *,
        method: str,
        path: str,
        body: Optional[bytes],
        headers: Mapping[str, str],
    ) -> dict: ...


@dataclass(frozen=True)
class ControllerPolicy:
    """Budgets and constraints enforced on every mutation."""

    build_name: str
    allowed_fleets: tuple[str, ...]
    declared_workload: Mapping[str, Any] = field(default_factory=dict)
    max_runs: int = 3
    max_concurrent: int = 1
    total_budget_seconds: float = 2 * 60 * 60
    stop_timeout_seconds: float = 10 * 60
    poll_interval_seconds: float = 2.0
    cost_budget: Optional[float] = None
    max_price: Optional[float] = None
    backends: Optional[tuple[str, ...]] = None
    spot_policy: Optional[str] = None

    def to_public_data(self) -> dict[str, Any]:
        return {
            "build_name": self.build_name,
            "allowed_fleets": list(self.allowed_fleets),
            "declared_workload": dict(self.declared_workload),
            "max_runs": self.max_runs,
            "max_concurrent": self.max_concurrent,
            "total_budget_seconds": self.total_budget_seconds,
            "cost_budget": self.cost_budget,
            "max_price": self.max_price,
            "backends": list(self.backends) if self.backends else None,
            "spot_policy": self.spot_policy,
        }


@dataclass
class RunRecord:
    name: str
    run_id: str
    purpose: str
    expected_workload: dict[str, Any]
    config_digest: str
    artifact_paths: list[str]
    submitted_at: float
    status: str
    price_per_hour: Optional[float] = None
    instance_id: Optional[str] = None
    stopped_at: Optional[float] = None
    handoff: Optional[dict[str, Any]] = None
    accrued_cost: float = 0.0

    def to_data(self) -> dict[str, Any]:
        return asdict(self)


def workload_weaker_than(
    declared: Mapping[str, Any], candidate: Mapping[str, Any]
) -> Optional[str]:
    """Return the reason the candidate workload is weaker, or None."""
    for key, declared_value in declared.items():
        if declared_value is None:
            continue
        candidate_value = candidate.get(key)
        if candidate_value is None:
            return f"workload field {key!r} is missing (declared {declared_value!r})"
        if isinstance(declared_value, (int, float)) and not isinstance(declared_value, bool):
            try:
                if float(candidate_value) < float(declared_value):
                    return (
                        f"workload field {key!r} is weaker than declared: "
                        f"{candidate_value!r} < {declared_value!r}"
                    )
            except (TypeError, ValueError):
                return f"workload field {key!r} is not comparable to {declared_value!r}"
        elif candidate_value != declared_value:
            return (
                f"workload field {key!r} differs from declared: "
                f"{candidate_value!r} != {declared_value!r}"
            )
    return None


class EndpointController:
    """Session-scoped mutation controller owned by the parent CLI process."""

    def __init__(
        self,
        *,
        policy: ControllerPolicy,
        api: ControllerAPI,
        recorder: EndpointSessionRecorder,
        artifacts_dir: Path,
        workspace_dir: Optional[Path] = None,
        submissions_path: Optional[Path] = None,
        finalize: Optional[Callable[[str, dict], dict]] = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._policy = policy
        self._api = api
        self._recorder = recorder
        self._artifacts_dir = artifacts_dir
        self._workspace_dir = workspace_dir
        self._submissions_path = submissions_path
        self._finalize = finalize
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._started_at = clock()
        self._phase = "planning"
        self._runs: dict[str, RunRecord] = {}
        self._verified_run: Optional[str] = None
        self._finalized_result: Optional[dict[str, Any]] = None
        self._pending_handoff: Optional[dict[str, Any]] = None
        self._diagnostics: list[dict[str, Any]] = []
        self._persist()

    # -- typed mutations ---------------------------------------------------

    def submit_candidate(
        self,
        *,
        configuration_yaml: str,
        purpose: str,
        expected_workload: Mapping[str, Any],
    ) -> dict[str, Any]:
        with self._lock:
            self._check_not_terminal()
            self._check_deadline()
            self._accrue_costs()
            self._check_cost_budget()
            if len(self._runs) >= self._policy.max_runs:
                raise self._refuse(
                    f"run budget exhausted: {self._policy.max_runs} submissions were "
                    "already used in this session",
                    failure_class="budget",
                )
            active = self._active_run_names()
            if len(active) >= self._policy.max_concurrent:
                raise self._refuse(
                    "another prototype is still active "
                    f"({', '.join(active)}); stop it before submitting again",
                    failure_class="budget",
                )
            configuration = self._parse_configuration(configuration_yaml)
            self._check_constraints(configuration)
            weaker = workload_weaker_than(self._policy.declared_workload, expected_workload)
            if weaker is not None:
                raise self._refuse(weaker, failure_class="validation")
            config_digest = hashlib.sha256(configuration_yaml.encode()).hexdigest()
            for record in self._runs.values():
                if record.config_digest == config_digest and record.status != "failed-submit":
                    raise self._refuse(
                        f"duplicate submission: run {record.name} already used an "
                        "identical configuration; change the configuration or record "
                        "new evidence before retrying",
                        failure_class="validation",
                    )
            run_name = f"{self._policy.build_name}-{len(self._runs) + 1}"
            artifact_paths = self._package_artifacts(run_name, configuration_yaml, configuration)
            try:
                info = self._api.submit(configuration, run_name)
            except ControllerError:
                raise
            except Exception as e:
                self._record_diagnostic(
                    f"submission of {run_name} failed: {e}", failure_class="capacity"
                )
                self._persist()
                raise ControllerError(
                    f"submission of {run_name} failed: {e}", failure_class="capacity"
                ) from e
            record = RunRecord(
                name=info.name,
                run_id=info.run_id,
                purpose=purpose,
                expected_workload=dict(expected_workload),
                config_digest=config_digest,
                artifact_paths=[str(path) for path in artifact_paths],
                submitted_at=self._clock(),
                status=info.status,
                price_per_hour=info.price_per_hour,
                instance_id=info.instance_id,
                handoff=self._pending_handoff,
            )
            self._pending_handoff = None
            self._runs[record.name] = record
            self._phase = "prototyping"
            self._append_submission(record)
            self._recorder.record(
                "run_submitted",
                run_name=record.name,
                run_id=record.run_id,
                purpose=purpose,
                price_per_hour=record.price_per_hour,
            )
            self._persist()
            return {"run_name": record.name, "run_id": record.run_id, "status": record.status}

    def stop_and_handoff(
        self, *, run_name: str, handoff_requirements: Optional[Mapping[str, Any]] = None
    ) -> dict[str, Any]:
        with self._lock:
            self._check_not_terminal()
            record = self._require_run(run_name)
            if self._verified_run == run_name and self._finalized_result is None:
                raise self._refuse(
                    "the verified final service must stay running until its preset is finalized",
                    failure_class="validation",
                )
            self._api.stop([run_name], abort=False)
            deadline = self._clock() + self._policy.stop_timeout_seconds
            while True:
                info = self._api.get(run_name)
                if info is None or info.is_finished:
                    break
                if self._clock() >= deadline:
                    self._record_diagnostic(
                        f"run {run_name} did not reach a terminal state within "
                        f"{self._policy.stop_timeout_seconds:.0f}s of the stop request",
                        failure_class="controller",
                    )
                    self._persist()
                    raise ControllerError(
                        f"failed stop: {run_name} is still not terminal; do not submit "
                        "another run until it stops",
                        failure_class="controller",
                    )
                self._accrue_costs()
                self._sleep(self._policy.poll_interval_seconds)
            record.status = info.status if info is not None else "terminated"
            record.stopped_at = self._clock()
            if handoff_requirements:
                self._pending_handoff = {
                    "from_run": run_name,
                    "from_instance_id": record.instance_id,
                    "requirements": dict(handoff_requirements),
                }
            self._recorder.record(
                "run_stopped",
                run_name=run_name,
                terminal_status=record.status,
                handoff_requested=bool(handoff_requirements),
            )
            self._persist()
            return {"run_name": run_name, "status": record.status}

    def verify_final(
        self, *, run_name: str, declared_workload: Mapping[str, Any]
    ) -> dict[str, Any]:
        with self._lock:
            self._check_not_terminal()
            record = self._require_run(run_name)
            self._check_deadline()
            info = self._api.get(run_name)
            if info is None:
                self._record_diagnostic(
                    f"final service {run_name} no longer exists", failure_class="controller"
                )
                self._persist()
                raise ControllerError(
                    f"final service {run_name} no longer exists",
                    failure_class="controller",
                )
            record.status = info.status
            record.instance_id = info.instance_id or record.instance_id
            self._check_handoff_result(record, info)
            if info.is_finished or info.status != "running" or info.service_url is None:
                self._record_diagnostic(
                    f"final service {run_name} is not running "
                    f"(status={info.status}, url={'set' if info.service_url else 'missing'})",
                    failure_class="runtime",
                )
                self._persist()
                raise ControllerError(
                    f"final service {run_name} is not a running service",
                    failure_class="runtime",
                )
            weaker = workload_weaker_than(self._policy.declared_workload, declared_workload)
            if weaker is not None:
                raise self._refuse(weaker, failure_class="validation")
            self._verified_run = run_name
            self._phase = "verified"
            self._recorder.record("final_verified", run_name=run_name)
            self._persist()
            return {"run_name": run_name, "status": info.status, "service_url": info.service_url}

    def finalize_preset(
        self, *, run_name: str, report_metadata: Mapping[str, Any]
    ) -> dict[str, Any]:
        with self._lock:
            self._check_not_terminal()
            if self._verified_run != run_name:
                raise self._refuse(
                    f"run {run_name} has not passed verify_final",
                    failure_class="validation",
                )
            record = self._require_run(run_name)
            info = self._api.get(run_name)
            if info is None or info.is_finished or info.status != "running":
                self._record_diagnostic(
                    f"final service {run_name} disappeared before the preset was saved "
                    f"(status={info.status if info else 'missing'})",
                    failure_class="runtime",
                )
                self._persist()
                raise ControllerError(
                    f"final service {run_name} is no longer running; the final service "
                    "must stay running until its preset is verified and saved",
                    failure_class="runtime",
                )
            missing = [path for path in record.artifact_paths if not Path(path).exists()]
            if missing:
                self._record_diagnostic(
                    f"packaged artifacts are missing for {run_name}: {missing}",
                    failure_class="packaging",
                )
                self._persist()
                raise ControllerError(
                    f"packaged artifacts are missing for {run_name}",
                    failure_class="packaging",
                )
            if self._finalize is None:
                raise self._refuse(
                    "finalization is not available in this session",
                    failure_class="controller",
                )
            try:
                result = self._finalize(run_name, dict(report_metadata))
            except ControllerError as e:
                self._record_diagnostic(str(e), failure_class=e.failure_class)
                self._persist()
                raise
            except Exception as e:
                self._record_diagnostic(
                    f"finalization of {run_name} failed: {e}", failure_class="validation"
                )
                self._persist()
                raise ControllerError(
                    f"finalization of {run_name} failed: {e}", failure_class="validation"
                ) from e
            self._finalized_result = dict(result)
            self._phase = "finalized"
            self._recorder.record(
                "preset_finalized", run_name=run_name, preset_id=result.get("preset_id")
            )
            self._persist()
            return dict(result)

    # -- read-only helpers -------------------------------------------------

    def get_endpoint_context(self) -> dict[str, Any]:
        with self._lock:
            return {
                "policy": self._policy.to_public_data(),
                "state": self._state_data(),
            }

    def get_run_status(self, *, run_name: str) -> dict[str, Any]:
        with self._lock:
            record = self._require_run(run_name)
            info = self._api.get(run_name)
            if info is not None:
                record.status = info.status
                self._check_handoff_result(record, info)
                self._accrue_costs()
                self._persist()
            return {
                "run_name": run_name,
                "status": info.status if info else "missing",
                "is_finished": info.is_finished if info else True,
                "service_url": info.service_url if info else None,
            }

    def get_run_logs(self, *, run_name: str) -> dict[str, Any]:
        with self._lock:
            self._require_run(run_name)
        return {"run_name": run_name, "logs": self._api.logs(run_name)}

    def list_offers(self, *, filters: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
        return {"offers": self._api.offers(dict(filters or {}))}

    def request_service_http(
        self,
        *,
        run_name: str,
        method: str,
        path: str,
        body_base64: Optional[str] = None,
        headers: Optional[Mapping[str, str]] = None,
    ) -> dict[str, Any]:
        """Brokered HTTP to a session run's service; credentials stay parental."""
        import base64

        with self._lock:
            self._require_run(run_name)
            info = self._api.get(run_name)
        if info is None or info.service_url is None:
            raise ControllerError(
                f"run {run_name} has no routable service URL", failure_class="runtime"
            )
        body = base64.b64decode(body_base64) if body_base64 else None
        safe_headers = {
            key: value
            for key, value in (headers or {}).items()
            if key.lower() not in {"authorization", "cookie"}
        }
        return self._api.http_request(
            info.service_url, method=method, path=path, body=body, headers=safe_headers
        )

    # -- lifecycle ---------------------------------------------------------

    def cleanup(self, *, keep_final: bool = False) -> dict[str, Any]:
        """Stop every session run (optionally keeping the finalized service)."""
        with self._lock:
            kept: Optional[str] = None
            if keep_final and self._finalized_result is not None:
                kept = self._verified_run
            to_stop = [name for name in self._active_run_names() if name != kept]
            errors: list[str] = []
            if to_stop:
                try:
                    self._api.stop(to_stop, abort=False)
                except Exception as e:
                    errors.append(f"stop request failed: {e}")
                deadline = self._clock() + self._policy.stop_timeout_seconds
                pending = set(to_stop)
                while pending and not errors:
                    for name in list(pending):
                        info = self._api.get(name)
                        if info is None or info.is_finished:
                            self._runs[name].status = info.status if info else "terminated"
                            pending.discard(name)
                    if not pending:
                        break
                    if self._clock() >= deadline:
                        errors.append(f"runs did not stop in time: {sorted(pending)}")
                        break
                    self._sleep(self._policy.poll_interval_seconds)
            for error in errors:
                self._record_diagnostic(f"cleanup: {error}", failure_class="controller")
            self._recorder.record(
                "cleanup_completed",
                stopped=sorted(to_stop),
                kept_final=kept,
                errors=errors,
            )
            self._persist()
            if errors:
                raise ControllerError(
                    "cleanup failed: " + "; ".join(errors), failure_class="controller"
                )
            return {"stopped": sorted(to_stop), "kept_final": kept}

    def fail_session(self, reason: str, *, failure_class: str = "controller") -> None:
        with self._lock:
            self._phase = "failed"
            self._record_diagnostic(reason, failure_class=failure_class)
            self._persist()

    @property
    def finalized_result(self) -> Optional[dict[str, Any]]:
        with self._lock:
            return dict(self._finalized_result) if self._finalized_result else None

    @property
    def verified_run(self) -> Optional[str]:
        with self._lock:
            return self._verified_run

    def submitted_run_names(self) -> list[str]:
        with self._lock:
            return list(self._runs)

    def estimated_cost(self) -> float:
        with self._lock:
            self._accrue_costs()
            return round(sum(record.accrued_cost for record in self._runs.values()), 6)

    # -- internals ---------------------------------------------------------

    def _refuse(self, message: str, *, failure_class: str) -> ControllerError:
        self._recorder.record("operation_refused", reason=message, failure_class=failure_class)
        return ControllerError(message, failure_class=failure_class)

    def _check_not_terminal(self) -> None:
        if self._phase == "finalized":
            raise ControllerError(
                "the preset is already finalized; no further mutations are allowed",
                failure_class="validation",
            )
        if self._phase == "failed":
            raise ControllerError(
                "the session has failed; no further mutations are allowed",
                failure_class="controller",
            )

    def _check_deadline(self) -> None:
        elapsed = self._clock() - self._started_at
        if elapsed >= self._policy.total_budget_seconds:
            self._record_diagnostic(
                f"session deadline expired after {elapsed:.0f}s "
                f"(budget {self._policy.total_budget_seconds:.0f}s)",
                failure_class="budget",
            )
            self._persist()
            raise ControllerError(
                "session deadline expired; no further billable mutations are allowed",
                failure_class="budget",
            )

    def _accrue_costs(self) -> None:
        now = self._clock()
        for record in self._runs.values():
            if record.price_per_hour is None:
                continue
            end = record.stopped_at if record.stopped_at is not None else now
            record.accrued_cost = round(
                max(0.0, end - record.submitted_at) / 3600 * record.price_per_hour, 6
            )

    def _check_cost_budget(self) -> None:
        if self._policy.cost_budget is None:
            return
        total = sum(record.accrued_cost for record in self._runs.values())
        if total >= self._policy.cost_budget:
            self._record_diagnostic(
                f"estimated cost {total:.2f} reached the budget {self._policy.cost_budget:.2f}",
                failure_class="budget",
            )
            self._persist()
            raise ControllerError(
                "estimated cost budget exhausted; no further submissions are allowed",
                failure_class="budget",
            )

    def _active_run_names(self) -> list[str]:
        return [
            record.name
            for record in self._runs.values()
            if record.stopped_at is None
            and record.status not in {"terminated", "failed", "done", "aborted"}
        ]

    def _require_run(self, run_name: str) -> RunRecord:
        record = self._runs.get(run_name)
        if record is None:
            raise ControllerError(
                f"run {run_name} does not belong to this session",
                failure_class="validation",
            )
        return record

    def _parse_configuration(self, configuration_yaml: str) -> dict:
        try:
            configuration = yaml.safe_load(configuration_yaml)
        except yaml.YAMLError as e:
            raise self._refuse(
                f"candidate configuration is not valid YAML: {e}", failure_class="validation"
            ) from e
        if not isinstance(configuration, dict):
            raise self._refuse(
                "candidate configuration must be a YAML mapping", failure_class="validation"
            )
        if configuration.get("type") not in {"task", "service"}:
            raise self._refuse(
                "candidate configuration must be a dstack task or service",
                failure_class="validation",
            )
        return configuration

    def _check_constraints(self, configuration: dict) -> None:
        fleets = configuration.get("fleets")
        if not fleets or not isinstance(fleets, list):
            raise self._refuse(
                "candidate configuration must pin `fleets` to the allowed fleets",
                failure_class="validation",
            )
        disallowed = [fleet for fleet in fleets if fleet not in self._policy.allowed_fleets]
        if disallowed:
            raise self._refuse(
                f"fleets {disallowed} are not allowed "
                f"(allowed: {', '.join(self._policy.allowed_fleets)})",
                failure_class="validation",
            )
        if self._policy.max_price is not None:
            max_price = configuration.get("max_price")
            if max_price is None or float(max_price) > self._policy.max_price:
                raise self._refuse(
                    f"candidate max_price {max_price!r} exceeds or omits the endpoint "
                    f"limit {self._policy.max_price}",
                    failure_class="validation",
                )
        if self._policy.backends is not None:
            backends = configuration.get("backends") or []
            disallowed = [backend for backend in backends if backend not in self._policy.backends]
            if not backends or disallowed:
                raise self._refuse(
                    f"candidate backends {backends!r} must be a non-empty subset of "
                    f"{list(self._policy.backends)}",
                    failure_class="validation",
                )
        if self._policy.spot_policy is not None:
            spot = configuration.get("spot_policy")
            if spot != self._policy.spot_policy:
                raise self._refuse(
                    f"candidate spot_policy {spot!r} conflicts with the endpoint "
                    f"constraint {self._policy.spot_policy!r}",
                    failure_class="validation",
                )

    def _package_artifacts(
        self, run_name: str, configuration_yaml: str, configuration: dict
    ) -> list[Path]:
        """Durably copy the candidate config and referenced files before submit."""
        import shutil

        run_dir = self._artifacts_dir / run_name
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            config_path = run_dir / "configuration.yml"
            config_path.write_text(configuration_yaml, encoding="utf-8")
            paths = [config_path]
            for mapping in configuration.get("files") or []:
                local = mapping.get("local_path") if isinstance(mapping, dict) else mapping
                if not isinstance(local, str):
                    continue
                source = Path(local)
                if not source.is_absolute() and self._workspace_dir is not None:
                    source = self._workspace_dir / source
                if not source.exists():
                    raise ControllerError(
                        f"referenced file {local!r} does not exist; every referenced "
                        "file must be packageable before submission",
                        failure_class="packaging",
                    )
                destination = run_dir / source.name
                if source.is_dir():
                    shutil.copytree(source, destination, dirs_exist_ok=True)
                else:
                    shutil.copy2(source, destination)
                paths.append(destination)
            return paths
        except ControllerError:
            raise
        except OSError as e:
            raise ControllerError(
                f"could not package candidate artifacts for {run_name}: {e}",
                failure_class="packaging",
            ) from e

    def _check_handoff_result(self, record: RunRecord, info: RunInfo) -> None:
        handoff = record.handoff
        if not handoff or handoff.get("result") is not None:
            return
        if info.instance_id is None:
            return
        reused = handoff.get("from_instance_id") is not None and info.instance_id == handoff.get(
            "from_instance_id"
        )
        handoff["result"] = "reused" if reused else "lost"
        record.instance_id = info.instance_id
        self._recorder.record(
            "handoff_result",
            run_name=record.name,
            from_run=handoff.get("from_run"),
            result=handoff["result"],
        )

    def _append_submission(self, record: RunRecord) -> None:
        if self._submissions_path is None:
            return
        line = json.dumps(
            {
                "event": "submit",
                "name": record.name,
                "run_id": record.run_id,
                "status": record.status,
                "reason": record.purpose,
            },
            ensure_ascii=False,
        )
        try:
            with self._submissions_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass

    def _record_diagnostic(self, reason: str, *, failure_class: str) -> None:
        assert failure_class in FAILURE_CLASSES
        diagnostic = {
            "timestamp": time.time(),
            "phase": self._phase,
            "failure_class": failure_class,
            "reason": reason,
            "runs": {name: record.status for name, record in self._runs.items()},
        }
        self._diagnostics.append(diagnostic)
        self._recorder.record("diagnostic", **diagnostic)
        try:
            path = self._recorder.session.path / _DIAGNOSTICS_FILENAME
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(diagnostic, ensure_ascii=False, default=str) + "\n")
        except OSError:
            pass

    def _state_data(self) -> dict[str, Any]:
        self._accrue_costs()
        return {
            "phase": self._phase,
            "elapsed_seconds": round(self._clock() - self._started_at, 3),
            "total_budget_seconds": self._policy.total_budget_seconds,
            "max_runs": self._policy.max_runs,
            "runs_used": len(self._runs),
            "active_runs": self._active_run_names(),
            "estimated_cost": round(sum(record.accrued_cost for record in self._runs.values()), 6),
            "cost_budget": self._policy.cost_budget,
            "verified_run": self._verified_run,
            "finalized": self._finalized_result is not None,
            "runs": {name: record.to_data() for name, record in self._runs.items()},
            "diagnostics": list(self._diagnostics),
        }

    def _persist(self) -> None:
        path = self._recorder.session.path / _STATE_FILENAME
        temporary = path.with_suffix(".tmp")
        try:
            temporary.write_text(
                json.dumps(self._state_data(), indent=2, sort_keys=True, default=str) + "\n",
                encoding="utf-8",
            )
            temporary.replace(path)
        except OSError:
            pass


__all__ = [
    "FAILURE_CLASSES",
    "ControllerAPI",
    "ControllerError",
    "ControllerPolicy",
    "EndpointController",
    "RunInfo",
    "RunRecord",
    "workload_weaker_than",
]
