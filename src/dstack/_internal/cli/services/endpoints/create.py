import asyncio
import json
import os
import secrets
import uuid
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from pydantic import ValidationError

from dstack._internal.cli.models.endpoint_agent import AgentFinalReport
from dstack._internal.cli.models.endpoint_presets import EndpointPreset
from dstack._internal.cli.models.endpoints import EndpointConfiguration
from dstack._internal.cli.services.endpoints.agent import (
    EndpointAgentSession,
    EndpointAgentWorkspace,
    build_endpoint_agent_env,
    contains_redacted_value,
    create_endpoint_agent_session,
    endpoint_agent_workspace,
    get_claude_auth,
    get_redacted_values,
    get_sensitive_inherited_env_values,
    install_workspace_command,
    print_endpoint_progress,
    redact,
    run_endpoint_agent,
)
from dstack._internal.cli.services.endpoints.controller import (
    ControllerError,
    ControllerPolicy,
    EndpointController,
)
from dstack._internal.cli.services.endpoints.controller_api import DstackControllerAPI
from dstack._internal.cli.services.endpoints.inspect.service import (
    InspectionResult,
    inspect_endpoint_model,
)
from dstack._internal.cli.services.endpoints.presets import endpoint_preset_to_data
from dstack._internal.cli.services.endpoints.prompt import (
    format_controlled_mutation_surface,
    format_endpoint_constraints,
    format_model_inspection,
    get_endpoint_agent_system_prompt,
)
from dstack._internal.cli.services.endpoints.rpc import (
    CONTROLLER_SOCKET_ENV,
    CONTROLLER_TOKEN_ENV,
    EndpointControllerServer,
    get_controller_client_script,
)
from dstack._internal.cli.services.endpoints.session import (
    EndpointSessionRecorder,
    write_baseline_summary,
)
from dstack._internal.cli.services.endpoints.store import EndpointPresetStore
from dstack._internal.cli.services.endpoints.verify import (
    build_verified_endpoint_preset,
    load_endpoint_agent_report,
)
from dstack._internal.cli.utils.common import console, warn
from dstack._internal.core.errors import CLIError, ConfigurationError
from dstack._internal.core.models.envs import EnvSentinel
from dstack._internal.core.models.fleets import FleetStatus
from dstack.api import Client

_RUN_STOP_TIMEOUT_SECONDS = 10 * 60


@dataclass(frozen=True)
class EndpointPresetCreateResult:
    preset: EndpointPreset
    path: Path
    final_run_id: uuid.UUID
    final_run_name: str


def create_endpoint_preset(
    *,
    api: Client,
    configuration: EndpointConfiguration,
    store: EndpointPresetStore,
    keep_service: bool = False,
    build_name: Optional[str] = None,
    debug: bool = False,
) -> EndpointPresetCreateResult:
    agent_session = create_endpoint_agent_session(configuration, debug=debug)
    try:
        resolved_configuration = _resolve_endpoint_env(configuration)
        result = asyncio.run(
            _create_endpoint_preset(
                api=api,
                configuration=resolved_configuration,
                source_configuration=configuration,
                store=store,
                keep_service=keep_service,
                build_name=build_name,
                agent_session=agent_session,
            )
        )
    except BaseException:
        _finish_agent_session(agent_session)
        raise
    _finish_agent_session(agent_session, result.preset.id)
    return result


async def _create_endpoint_preset(
    *,
    api: Client,
    configuration: EndpointConfiguration,
    store: EndpointPresetStore,
    source_configuration: Optional[EndpointConfiguration] = None,
    keep_service: bool = False,
    build_name: Optional[str] = None,
    agent_session: EndpointAgentSession,
) -> EndpointPresetCreateResult:
    source_configuration = source_configuration or configuration
    build_name = build_name or _get_build_name(configuration.name)
    allowed_fleets = _get_allowed_fleets(api, configuration)
    if not allowed_fleets:
        raise CLIError("The project has no active fleets available for preset creation")
    auth = get_claude_auth()

    endpoint_env = configuration.env.as_dict()
    token = getattr(api.client, "_token", None)
    if not isinstance(token, str) or not token:
        raise CLIError("The configured dstack client has no authentication token")
    redacted_values = get_redacted_values(
        [
            token,
            auth.api_key or "",
            *endpoint_env.values(),
            *get_sensitive_inherited_env_values(),
        ]
    )
    recorder = EndpointSessionRecorder(agent_session, redacted_values=redacted_values)
    recorder.record(
        "session_started",
        model=configuration.model.api_model_name,
        build_name=build_name,
        allowed_fleets=list(allowed_fleets),
    )
    controlled = not _legacy_agent_shell_enabled()
    recorder.record("mutation_surface", controlled=controlled)
    report: Optional[AgentFinalReport] = None
    preset: Optional[EndpointPreset] = None
    preset_path: Optional[Path] = None
    creation_succeeded = False
    creation_error: Optional[str] = None
    cleanup_error: Optional[str] = None
    with endpoint_agent_workspace(install_dstack_cli=not controlled) as workspace:
        env = build_endpoint_agent_env(
            api=api,
            endpoint_env=endpoint_env,
            auth=auth,
            workspace=workspace,
            token=token,
            controlled=controlled,
        )
        controller: Optional[EndpointController] = None
        server: Optional[EndpointControllerServer] = None
        finalized: dict[str, object] = {}
        if controlled:
            controller = EndpointController(
                policy=_build_controller_policy(
                    configuration, build_name=build_name, allowed_fleets=allowed_fleets
                ),
                api=DstackControllerAPI(api, endpoint_env=endpoint_env, service_token=token),
                recorder=recorder,
                artifacts_dir=agent_session.path / "artifacts",
                workspace_dir=workspace.path,
                submissions_path=workspace.submissions_path,
                finalize=_make_finalizer(
                    api=api,
                    source_configuration=source_configuration,
                    store=store,
                    redacted_values=redacted_values,
                    holder=finalized,
                ),
            )
            server = EndpointControllerServer(
                controller,
                socket_path=workspace.path.parent / "ctl.sock",
                redacted_values=redacted_values,
            )
            env[CONTROLLER_SOCKET_ENV] = str(server.socket_path)
            env[CONTROLLER_TOKEN_ENV] = server.token
            install_workspace_command(workspace, "endpoint", get_controller_client_script())
        recorder.phase_started("inspection")
        inspection_data = _run_inspection_stage(
            configuration=configuration,
            workspace=workspace,
            agent_session=agent_session,
        )
        recorder.phase_completed(
            "inspection",
            classification=(inspection_data or {}).get("classification"),
            candidates=[
                candidate.get("runtime")
                for candidate in (inspection_data or {}).get("candidates", [])
            ],
        )
        prompt = _build_prompt(
            configuration=configuration,
            build_name=build_name,
            allowed_fleets=allowed_fleets,
            inspection_data=inspection_data,
            controlled=controlled,
        )
        if agent_session.debug:
            agent_session.write_prompt(prompt)
        print_endpoint_progress(
            f"Starting endpoint preset creation for {configuration.model.api_model_name}. "
            f"Allowed fleets: {', '.join(allowed_fleets)}.",
            agent_session=agent_session,
        )
        try:
            recorder.phase_started("agent")
            if server is not None:
                async with server:
                    process_output = await run_endpoint_agent(
                        prompt=prompt,
                        env=env,
                        workspace=workspace,
                        auth=auth,
                        redacted_values=redacted_values,
                        agent_session=agent_session,
                    )
            else:
                process_output = await run_endpoint_agent(
                    prompt=prompt,
                    env=env,
                    workspace=workspace,
                    auth=auth,
                    redacted_values=redacted_values,
                    agent_session=agent_session,
                )
            recorder.phase_completed(
                "agent",
                submitted_runs=_load_submitted_run_names(workspace.submissions_path),
            )
            recorder.phase_started("verification")
            if controller is not None:
                report, preset, preset_path = _require_controller_finalization(
                    controller=controller,
                    process_output=process_output,
                    holder=finalized,
                    redacted_values=redacted_values,
                )
            else:
                report = load_endpoint_agent_report(
                    output=process_output,
                    workspace=workspace,
                    redacted_values=redacted_values,
                )
                run = api.client.runs.get(api.project, report.run_name)
                preset = build_verified_endpoint_preset(
                    run=run,
                    endpoint_configuration=source_configuration,
                    report=report,
                )
                if contains_redacted_value(endpoint_preset_to_data(preset), redacted_values):
                    raise CLIError("Generated endpoint preset contains a secret value")
                preset_path = store.save(preset)
            recorder.phase_completed(
                "verification",
                run_name=report.run_name,
                preset_id=preset.id,
            )
            print_endpoint_progress(
                f"Saved endpoint preset {preset.id} for {preset.base} at {preset_path}.",
                agent_session=agent_session,
            )
            creation_succeeded = True
        except BaseException as e:
            creation_error = str(e) or type(e).__name__
            recorder.fail_open_phases(creation_error)
            raise
        finally:
            keep_final_service = keep_service and creation_succeeded
            recorder.phase_started("cleanup")
            try:
                if controller is not None:
                    controller.cleanup(keep_final=keep_final_service)
                else:
                    await _cleanup_runs(
                        api=api,
                        build_name=build_name,
                        workspace=workspace,
                        final_run_name=report.run_name if report is not None else None,
                        keep_final_service=keep_final_service,
                        agent_session=agent_session,
                    )
                recorder.phase_completed("cleanup", kept_final_service=keep_final_service)
            except Exception as e:
                cleanup_error = str(e)
                recorder.phase_failed("cleanup", error=cleanup_error)
                if keep_final_service:
                    with suppress(Exception):
                        if controller is not None:
                            controller.cleanup(keep_final=False)
                        else:
                            await _cleanup_runs(
                                api=api,
                                build_name=build_name,
                                workspace=workspace,
                                final_run_name=report.run_name if report is not None else None,
                                agent_session=agent_session,
                            )
            submitted_run_names = _load_submitted_run_names(workspace.submissions_path)
            if report is not None and report.run_name:
                submitted_run_names.append(report.run_name)
            write_baseline_summary(
                recorder,
                submitted_run_names=list(dict.fromkeys(submitted_run_names)),
                succeeded=creation_succeeded,
                failure_summary=creation_error or cleanup_error,
            )

    if cleanup_error is not None:
        raise CLIError(f"Failed to clean up preset creation runs: {cleanup_error}")
    assert preset is not None
    assert preset_path is not None
    assert report is not None
    assert report.run_id is not None
    assert report.run_name is not None
    return EndpointPresetCreateResult(
        preset=preset,
        path=preset_path,
        final_run_id=report.run_id,
        final_run_name=report.run_name,
    )


def _resolve_endpoint_env(configuration: EndpointConfiguration) -> EndpointConfiguration:
    configuration = configuration.copy(deep=True)
    for key, value in configuration.env.items():
        if not isinstance(value, EnvSentinel):
            continue
        try:
            configuration.env[key] = value.from_env(os.environ)
        except ValueError as e:
            raise ConfigurationError(str(e)) from e
    return configuration


def _finish_agent_session(
    session: EndpointAgentSession,
    preset_id: Optional[str] = None,
) -> None:
    try:
        path = session.finish(preset_id)
    except OSError as e:
        path = session.path
        warn(f"Could not finalize agent output. Files remain at {path}: {e}")
    console.print(f"Agent log saved to [code]{path / 'agent.log'}[/]")


def _get_build_name(endpoint_name: Optional[str]) -> str:
    if endpoint_name is None:
        raise CLIError("Endpoint name is required. Set `name` in the configuration or use --name")
    suffix = secrets.token_hex(3)
    # Leave room for the numeric submission suffix while retaining a recognizable prefix.
    prefix = endpoint_name[:28].rstrip("-")
    return f"{prefix}-{suffix}"


def _get_allowed_fleets(api: Client, configuration: EndpointConfiguration) -> tuple[str, ...]:
    if configuration.fleets is not None:
        return tuple(
            fleet.format() if hasattr(fleet, "format") else str(fleet)
            for fleet in configuration.fleets
        )
    fleets = api.client.fleets.list(api.project, include_imported=True)
    return tuple(
        fleet.name if fleet.project_name == api.project else f"{fleet.project_name}/{fleet.name}"
        for fleet in fleets
        if fleet.status == FleetStatus.ACTIVE
    )


def _build_prompt(
    *,
    configuration: EndpointConfiguration,
    build_name: str,
    allowed_fleets: Sequence[str],
    inspection_data: Optional[dict] = None,
    controlled: bool = False,
) -> str:
    context_lines = [f"- service_model_name: {configuration.model.api_model_name}"]
    context_lines.append(f"- model_source: {configuration.model.source_type}")
    context_lines.append(f"- requested_modality: {configuration.model.requested_modality}")
    if configuration.model.requested_revision is not None:
        context_lines.append(f"- model_revision: {configuration.model.requested_revision}")
    if configuration.model.allows_variant_selection:
        context_lines.append(f"- base_model: {configuration.model.api_model_name}")
    else:
        context_lines.append(f"- model_locator: {configuration.model.exact_repo}")
    if configuration.context_length is not None:
        context_lines.append(f"- context_length: {configuration.context_length}")
    inspection_block = ""
    if inspection_data is not None:
        inspection_block = "\n" + format_model_inspection(inspection_data) + "\n"
    mutation_block = ""
    if controlled:
        mutation_block = "\n" + format_controlled_mutation_surface() + "\n"
    return f"""{get_endpoint_agent_system_prompt()}
{mutation_block}
Endpoint context:
- endpoint_name: {build_name}
{chr(10).join(context_lines)}
{inspection_block}
{
        format_endpoint_constraints(
            configuration,
            configuration.env.as_dict(),
            allowed_fleets=allowed_fleets,
        )
    }
"""


_HF_TOKEN_ENV_NAMES = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN")


def _get_hf_token(configuration: EndpointConfiguration) -> Optional[str]:
    endpoint_env = configuration.env.as_dict()
    for name in _HF_TOKEN_ENV_NAMES:
        value = endpoint_env.get(name) or os.getenv(name)
        if isinstance(value, str) and value:
            return value
    return None


def _run_inspection_stage(
    *,
    configuration: EndpointConfiguration,
    workspace: EndpointAgentWorkspace,
    agent_session: EndpointAgentSession,
) -> Optional[dict]:
    """Run deterministic model inspection and persist its snapshot and evidence.

    Returns the compact evidence object for the agent prompt, or ``None`` when
    inspection does not apply or failed (the agent then follows the research
    path with an explicit note).
    """
    result: InspectionResult = inspect_endpoint_model(
        configuration,
        token=_get_hf_token(configuration),
    )
    if result.skipped_reason is not None:
        print_endpoint_progress(
            f"Deterministic model inspection skipped: {result.skipped_reason}",
            agent_session=agent_session,
        )
        return None
    if result.error is not None or result.inspection is None:
        print_endpoint_progress(
            f"Deterministic model inspection unavailable: "
            f"{result.error or 'no inspection produced'}. "
            "Continuing on the agent research path.",
            agent_session=agent_session,
        )
        return None
    assert result.snapshot is not None
    inspection_data = result.inspection.to_data()
    try:
        _write_json(agent_session.path / "inspection-snapshot.json", result.snapshot.to_data())
        _write_json(agent_session.path / "inspection.json", inspection_data)
        _write_json(workspace.path / "inspection.json", inspection_data)
    except OSError as e:
        warn(f"Could not persist model inspection output: {e}")
    candidates = ", ".join(
        f"{candidate.runtime} ({candidate.evidence_level.value})"
        for candidate in result.inspection.candidates
    )
    print_endpoint_progress(
        f"Deterministic inspection pinned {result.inspection.model} at "
        f"{result.inspection.revision[:12]}: modality={result.inspection.modality}, "
        f"classification={result.inspection.classification}, "
        f"candidates=[{candidates or 'none'}].",
        agent_session=agent_session,
    )
    return inspection_data


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


_LEGACY_SHELL_ENV = "DSTACK_ENDPOINT_LEGACY_AGENT_SHELL"


def _legacy_agent_shell_enabled() -> bool:
    """Development flag: give the agent the legacy unrestricted dstack shell."""
    return os.getenv(_LEGACY_SHELL_ENV, "").strip().lower() in {"1", "true", "yes"}


def _env_number(name: str, default: Optional[float]) -> Optional[float]:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError as e:
        raise CLIError(f"{name} must be a number, got {value!r}") from e


def _build_controller_policy(
    configuration: EndpointConfiguration,
    *,
    build_name: str,
    allowed_fleets: Sequence[str],
) -> ControllerPolicy:
    declared_workload: dict = {}
    if configuration.model.requested_modality != "auto":
        declared_workload["modality"] = configuration.model.requested_modality
    if configuration.context_length is not None:
        declared_workload["context_length"] = configuration.context_length
    backends = configuration.backends
    spot_policy = configuration.spot_policy
    max_runs = _env_number("DSTACK_ENDPOINT_MAX_RUNS", 3)
    max_concurrent = _env_number("DSTACK_ENDPOINT_MAX_CONCURRENT", 1)
    total_budget = _env_number("DSTACK_ENDPOINT_TOTAL_BUDGET_SECONDS", 2 * 60 * 60)
    assert max_runs is not None and max_concurrent is not None and total_budget is not None
    return ControllerPolicy(
        build_name=build_name,
        allowed_fleets=tuple(allowed_fleets),
        declared_workload=declared_workload,
        max_runs=int(max_runs),
        max_concurrent=int(max_concurrent),
        total_budget_seconds=total_budget,
        cost_budget=_env_number("DSTACK_ENDPOINT_COST_BUDGET", None),
        max_price=configuration.max_price,
        backends=tuple(str(backend.value) for backend in backends) if backends else None,
        spot_policy=spot_policy.value if spot_policy is not None else None,
    )


def _make_finalizer(
    *,
    api: Client,
    source_configuration: EndpointConfiguration,
    store: EndpointPresetStore,
    redacted_values: tuple[str, ...],
    holder: dict,
):
    """Build the controller's finalize callback: verify the run, save the preset."""

    def finalize(run_name: str, report_metadata: dict) -> dict:
        try:
            report = AgentFinalReport.parse_obj(report_metadata)
        except ValidationError as e:
            raise ControllerError(
                f"final report is invalid: {e}", failure_class="validation"
            ) from e
        if not report.success:
            raise ControllerError(
                "final report does not claim success", failure_class="validation"
            )
        if report.run_name != run_name:
            raise ControllerError(
                "final report identifies a different run", failure_class="validation"
            )
        run = api.client.runs.get(api.project, report.run_name)
        try:
            preset = build_verified_endpoint_preset(
                run=run,
                endpoint_configuration=source_configuration,
                report=report,
            )
        except CLIError as e:
            raise ControllerError(str(e), failure_class="validation") from e
        if contains_redacted_value(endpoint_preset_to_data(preset), redacted_values):
            raise ControllerError(
                "generated endpoint preset contains a secret value",
                failure_class="validation",
            )
        try:
            path = store.save(preset)
        except CLIError as e:
            raise ControllerError(str(e), failure_class="packaging") from e
        holder["preset"] = preset
        holder["report"] = report
        holder["path"] = path
        return {
            "preset_id": preset.id,
            "path": str(path),
            "run_name": report.run_name,
            "run_id": str(report.run_id),
        }

    return finalize


def _require_controller_finalization(
    *,
    controller: EndpointController,
    process_output,
    holder: dict,
    redacted_values: tuple[str, ...],
) -> tuple[AgentFinalReport, EndpointPreset, Path]:
    """In controlled mode, success exists only if the controller finalized."""
    result = controller.finalized_result
    if result is None:
        summary = None
        if isinstance(process_output.report_data, dict):
            summary = process_output.report_data.get("failure_summary")
        message = (
            summary
            or process_output.error
            or "The agent exited without finalizing a preset through the controller"
        )
        controller.fail_session(str(message), failure_class="controller")
        raise CLIError(redact(str(message), redacted_values))
    report = holder.get("report")
    preset = holder.get("preset")
    path = holder.get("path")
    assert isinstance(report, AgentFinalReport)
    assert isinstance(preset, EndpointPreset)
    assert isinstance(path, Path)
    return report, preset, path


async def _cleanup_runs(
    *,
    api: Client,
    build_name: str,
    workspace: EndpointAgentWorkspace,
    final_run_name: Optional[str],
    agent_session: EndpointAgentSession,
    keep_final_service: bool = False,
) -> None:
    run_names = _load_submitted_run_names(workspace.submissions_path)
    if final_run_name is not None:
        run_names.append(final_run_name)
    run_names = list(dict.fromkeys(run_names))
    expected_prefix = f"{build_name}-"
    run_names = [name for name in run_names if name.startswith(expected_prefix)]
    if keep_final_service:
        run_names = [name for name in run_names if name != final_run_name]
    active_names = []
    for name in run_names:
        run = api.runs.get(name)
        if run is not None and not run.status.is_finished():
            active_names.append(name)
    if not active_names:
        return
    print_endpoint_progress(
        f"Stopping preset creation runs: {', '.join(active_names)}.",
        agent_session=agent_session,
    )
    api.client.runs.stop(api.project, active_names, abort=False)
    deadline = asyncio.get_running_loop().time() + _RUN_STOP_TIMEOUT_SECONDS
    pending = set(active_names)
    while pending:
        if asyncio.get_running_loop().time() >= deadline:
            raise CLIError(f"Timed out waiting for runs to stop: {', '.join(sorted(pending))}")
        for name in list(pending):
            run = api.runs.get(name)
            if run is None or run.status.is_finished():
                pending.remove(name)
        if pending:
            await asyncio.sleep(2)
    print_endpoint_progress("All preset creation runs stopped.", agent_session=agent_session)


def _load_submitted_run_names(path: Path) -> list[str]:
    if not path.exists():
        return []
    names = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("name"), str):
            name = value["name"].strip()
            if name:
                names.append(name)
    return names
