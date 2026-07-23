import enum
import json
from pathlib import Path
from typing import Any, Sequence

from dstack._internal.cli.models.endpoints import EndpointConfiguration
from dstack._internal.core.models.profiles import ProfileParams

_SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent / "resources" / "system_prompt.md"


def get_endpoint_agent_system_prompt() -> str:
    return _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()


def format_endpoint_constraints(
    configuration: EndpointConfiguration,
    endpoint_env: dict[str, str],
    *,
    allowed_fleets: Sequence[str],
) -> str:
    lines = [
        "Fixed endpoint constraints:",
        "- Do not submit any task or service that conflicts with these values.",
    ]
    for field in ProfileParams.__fields__:
        if field == "fleets":
            continue
        value = getattr(configuration, field)
        if value is not None:
            lines.append(f"- {field}: {_format_constraint_value(value)}")
    if allowed_fleets:
        lines.append(f"- fleets: {', '.join(allowed_fleets)}")
    if configuration.gateway is not None:
        lines.append(f"- gateway: {_format_constraint_value(configuration.gateway)}")
    lines.append("- endpoint_env_keys: " + (", ".join(endpoint_env) if endpoint_env else "none"))
    return "\n".join(lines)


def format_controlled_mutation_surface() -> str:
    """The prompt section that replaces direct dstack CLI usage in controlled mode."""
    return """# Controlled Mutation Surface

This session runs with the controlled mutation surface. It overrides every
instruction elsewhere in this prompt that tells you to run `dstack` CLI
commands: the `dstack` CLI is not installed, and no dstack token, service
bearer token, or endpoint secret value is present in this shell. Do not look
for them.

All run lifecycle operations go through the `endpoint` command, which talks to
the parent CLI controller. The controller owns authoritative session state,
enforces the endpoint constraints, run/concurrency/time/cost budgets, and run
naming, and refuses invalid transitions with a JSON error explaining why. A
refusal is a policy decision — record it in progress and change your plan;
never look for another way to mutate runs.

- `endpoint context` — endpoint constraints, budgets, and current session
  state (phase, runs used, active runs, estimated cost).
- `endpoint offers --filters FILTERS.json` — evaluate real offers; FILTERS
  must contain {"configuration": <candidate task/service mapping>}.
- `endpoint submit --file CONFIG.yml --purpose TEXT --workload WORKLOAD.json`
  — submit a candidate task or service. The controller assigns the run name
  (returned in the response); never invent run names. `WORKLOAD.json` states
  the workload this candidate is expected to validate and must not be weaker
  than the endpoint's declared workload.
- `endpoint status RUN_NAME` / `endpoint logs RUN_NAME` — authoritative run
  status and logs. Poll status with bounded loops as instructed elsewhere.
- `endpoint stop-handoff RUN_NAME [--requirements REQS.json]` — stop a run
  and wait for its terminal state before anything else reuses its instance or
  caches. Always stop through this operation, and wait for it to return.
- `endpoint http RUN_NAME METHOD PATH [--body FILE] [--header K:V]` — send a
  real model request to the run's service URL. The controller injects service
  authentication; you never see the token. The response body is base64 in
  `body_base64`.
- `endpoint verify RUN_NAME --workload WORKLOAD.json` — declare the final
  service and the workload it was validated with. Required before finalize.
- `endpoint finalize RUN_NAME --report REPORT.json` — REPORT.json is the
  exact `final_report.json` object. The controller re-verifies the run is
  still running, checks packaged artifacts, builds and saves the preset, and
  returns the preset id. The final service must stay running until this
  succeeds.

On success you must call `endpoint verify` and then `endpoint finalize`
before writing the final report; a success report without controller
finalization is treated as a failure. Continue to write `final_report.json`
and submit it through `StructuredOutput` exactly as specified, and keep
using `progress` for decisions. `submissions.jsonl` is maintained by the
controller in this mode; do not write it yourself."""


def format_model_inspection(inspection_data: dict[str, Any]) -> str:
    """Render the compact deterministic-inspection evidence block for the agent."""
    compact = {key: value for key, value in inspection_data.items() if key != "ir"}
    return "\n".join(
        [
            "Deterministic model inspection (evidence object):",
            "```json",
            json.dumps(compact, indent=2, sort_keys=True),
            "```",
            "",
            "The full Model Shape IR and raw metadata snapshot are in "
            "`inspection.json` in this workspace.",
            "Interpret this object exactly as described in the "
            "`/dstack-prototyping` skill section "
            '"Interpreting the inspection evidence object": take the '
            "highest-confidence candidate through service-first validation, do "
            "not research alternate frameworks until it fails or the evidence "
            "names a material ambiguity, record why any lower-ranked candidate "
            "is tried, never override a negative result without new evidence, "
            "and always run the listed smoke test — metadata classification is "
            "not runtime validation.",
        ]
    )


def _format_constraint_value(value: Any) -> str:
    data = _constraint_value_to_data(value)
    if isinstance(data, (bool, dict, list)):
        return json.dumps(data, sort_keys=True)
    return str(data)


def _constraint_value_to_data(value: Any) -> Any:
    if isinstance(value, enum.Enum):
        return value.value
    if hasattr(value, "format"):
        return value.format()
    if hasattr(value, "json"):
        return json.loads(value.json(exclude_none=True))
    if isinstance(value, dict):
        return {key: _constraint_value_to_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_constraint_value_to_data(item) for item in value]
    return value
