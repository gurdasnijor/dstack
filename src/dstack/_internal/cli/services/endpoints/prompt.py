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
