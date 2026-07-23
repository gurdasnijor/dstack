"""Structured phase events and baseline instrumentation for preset creation.

Every preset-creation session appends phase events to ``events.jsonl`` in the
agent session directory and, on completion, writes ``baseline.json`` — the
wall-clock, run-count, and outcome record used to measure the endpoint agent
workflow. This module only observes; it never changes mutation behavior.

Cost baselines require billable runs and are recorded as ``null`` with an
explicit reason until run price data is captured by the controller.
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

from dstack._internal.cli.services.endpoints.agent import (
    EndpointAgentSession,
    redact,
)

_EVENTS_FILENAME = "events.jsonl"
_BASELINE_FILENAME = "baseline.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass
class EndpointSessionRecorder:
    """Append-only, redacting phase-event recorder for one creation session."""

    session: EndpointAgentSession
    redacted_values: Sequence[str] = ()
    _started: float = field(default_factory=time.monotonic, init=False, repr=False)
    _phase_started: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _events_enabled: bool = field(default=True, init=False, repr=False)

    @property
    def events_path(self) -> Path:
        return self.session.path / _EVENTS_FILENAME

    def record(self, event: str, **data: Any) -> None:
        record = {"timestamp": _utc_now(), "event": event, **data}
        line = redact(
            json.dumps(record, ensure_ascii=False, sort_keys=True, default=str),
            self.redacted_values,
        )
        if not self._events_enabled:
            return
        try:
            with self.events_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
        except OSError:
            # Instrumentation must never break preset creation.
            self._events_enabled = False

    def phase_started(self, phase: str, **data: Any) -> None:
        self._phase_started[phase] = time.monotonic()
        self.record("phase_started", phase=phase, **data)

    def phase_completed(self, phase: str, **data: Any) -> None:
        duration: Optional[float] = None
        started = self._phase_started.pop(phase, None)
        if started is not None:
            duration = round(time.monotonic() - started, 3)
        self.record("phase_completed", phase=phase, duration_seconds=duration, **data)

    def phase_failed(self, phase: str, error: str, **data: Any) -> None:
        duration: Optional[float] = None
        started = self._phase_started.pop(phase, None)
        if started is not None:
            duration = round(time.monotonic() - started, 3)
        self.record("phase_failed", phase=phase, duration_seconds=duration, error=error, **data)

    def fail_open_phases(self, error: str) -> None:
        for phase in list(self._phase_started):
            self.phase_failed(phase, error=error)

    def elapsed_seconds(self) -> float:
        return round(time.monotonic() - self._started, 3)


def load_events(session_path: Path) -> list[dict[str, Any]]:
    path = session_path / _EVENTS_FILENAME
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            events.append(value)
    return events


def write_baseline_summary(
    recorder: EndpointSessionRecorder,
    *,
    submitted_run_names: Sequence[str],
    succeeded: bool,
    failure_summary: Optional[str] = None,
) -> Optional[Path]:
    """Summarize the session for workflow baseline measurement."""
    events = load_events(recorder.session.path)
    phases: dict[str, Any] = {}
    failures = 0
    for event in events:
        name = event.get("event")
        phase = event.get("phase")
        if name == "phase_completed" and isinstance(phase, str):
            phases[phase] = event.get("duration_seconds")
        elif name == "phase_failed":
            failures += 1
            if isinstance(phase, str):
                phases.setdefault(phase, event.get("duration_seconds"))
    baseline = {
        "baseline_version": 1,
        "succeeded": succeeded,
        "failure_summary": redact(failure_summary, recorder.redacted_values)
        if failure_summary
        else None,
        "total_seconds": recorder.elapsed_seconds(),
        "phase_seconds": phases,
        "runs_submitted": len(submitted_run_names),
        "run_names": list(submitted_run_names),
        "phase_failures": failures,
        "estimated_cost": None,
        "estimated_cost_unavailable_reason": (
            "run price capture requires the controller mutation boundary; "
            "no billable run prices were recorded in this session"
        ),
    }
    path = recorder.session.path / _BASELINE_FILENAME
    try:
        path.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except OSError:
        return None
    return path


__all__ = [
    "EndpointSessionRecorder",
    "load_events",
    "write_baseline_summary",
]
