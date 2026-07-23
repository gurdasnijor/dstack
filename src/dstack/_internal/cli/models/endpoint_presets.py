import re
from datetime import datetime
from typing import Annotated, Any, Literal, Optional

from pydantic import (
    Field,
    PositiveFloat,
    PositiveInt,
    parse_obj_as,
    root_validator,
    validator,
)

from dstack._internal.core.models.common import CoreModel
from dstack._internal.core.models.configurations import ServiceConfiguration
from dstack._internal.core.models.profiles import ProfileParams
from dstack._internal.core.models.resources import CPUSpec, ResourcesSpec


class EndpointBenchmarkWorkload(CoreModel):
    api: str
    request_path: Optional[str] = None
    num_requests: PositiveInt
    concurrency: PositiveInt
    input_tokens: Optional[PositiveInt] = None
    output_tokens: Optional[Annotated[int, Field(ge=2)]] = None
    width: Optional[PositiveInt] = None
    height: Optional[PositiveInt] = None
    num_inference_steps: Optional[PositiveInt] = None
    outputs_per_request: Optional[PositiveInt] = None
    output_unit: Optional[str] = None
    parameters: dict[str, Any] = Field(default_factory=dict)

    @validator("api", "request_path", "output_unit")
    def validate_optional_non_empty(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not value.strip():
            raise ValueError("value must be non-empty")
        return value


class EndpointBenchmarkLatency(CoreModel):
    mean: Annotated[float, Field(ge=0)]
    p50: Annotated[float, Field(ge=0)]
    p99: Annotated[float, Field(ge=0)]


class EndpointBenchmarkMetrics(CoreModel):
    successful_requests: Annotated[int, Field(ge=0)]
    failed_requests: Annotated[int, Field(ge=0)]
    duration_seconds: PositiveFloat
    latency_ms: Optional[EndpointBenchmarkLatency] = None
    total_outputs: Optional[Annotated[int, Field(ge=0)]] = None
    total_output_bytes: Optional[Annotated[int, Field(ge=0)]] = None
    total_input_tokens: Optional[Annotated[int, Field(ge=0)]] = None
    total_output_tokens: Optional[Annotated[int, Field(ge=0)]] = None
    ttft_ms: Optional[EndpointBenchmarkLatency] = None
    tpot_ms: Optional[EndpointBenchmarkLatency] = None


class EndpointBenchmarkTarget(CoreModel):
    type: Literal["gateway", "server-proxy"]


class EndpointBenchmarkClient(CoreModel):
    type: Literal["local"]


class EndpointBenchmark(CoreModel):
    tool: str
    tool_version: str
    command: str
    workload: EndpointBenchmarkWorkload
    metrics: EndpointBenchmarkMetrics
    target: Optional[EndpointBenchmarkTarget] = None
    client: Optional[EndpointBenchmarkClient] = None

    @validator("tool", "tool_version", "command")
    def validate_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must be non-empty")
        return value

    @validator("command")
    def validate_command_has_no_bearer_token(cls, value: str) -> str:
        for match in re.finditer(r"(?i)\bbearer\s+([^\s\"']+)", value):
            token = match.group(1)
            if token.startswith("$") or "redacted" in token.lower() or set(token) == {"*"}:
                continue
            raise ValueError("command must not contain a bearer token value")
        return value

    @root_validator(skip_on_failure=True)
    def validate_metrics(cls, values: dict) -> dict:
        metrics = values.get("metrics")
        workload = values.get("workload")
        if metrics.failed_requests != 0:
            raise ValueError("benchmark must not include failed requests")
        if metrics.successful_requests != workload.num_requests:
            raise ValueError("benchmark request count must match workload.num_requests")
        if workload.api in {"chat_completions", "completions"}:
            required_workload = ("input_tokens", "output_tokens")
            required_metrics = ("total_input_tokens", "total_output_tokens", "ttft_ms", "tpot_ms")
            if any(getattr(workload, field) is None for field in required_workload):
                raise ValueError("token benchmark must include input_tokens and output_tokens")
            if any(getattr(metrics, field) is None for field in required_metrics):
                raise ValueError("token benchmark must include token totals and TTFT/TPOT latency")
        else:
            if workload.request_path is None:
                raise ValueError("non-token benchmark must include request_path")
            if metrics.latency_ms is None:
                raise ValueError("non-token benchmark must include end-to-end latency")
            if workload.outputs_per_request is not None:
                expected_outputs = workload.num_requests * workload.outputs_per_request
                if metrics.total_outputs != expected_outputs:
                    raise ValueError(
                        "benchmark output count must match num_requests * outputs_per_request"
                    )
        return values


class EndpointPresetValidationReplica(CoreModel):
    resources: list[ResourcesSpec]
    """Exact resources for each running replica in this service replica group."""


class EndpointPresetValidation(CoreModel):
    replicas: list[EndpointPresetValidationReplica]
    """Ordered to match `ServiceConfiguration.replica_groups`."""
    benchmark: EndpointBenchmark


class EndpointPreset(CoreModel):
    base: str
    """Base model used for local preset lookup."""
    id: str
    model: str
    """Exact model locator loaded by the service command."""
    api_model_name: Optional[str] = None
    """Client-facing model name, independent of dstack's chat model registration."""
    source: str = "unknown"
    """Model source type, for example `huggingface`, `url`, `path`, or `custom`."""
    revision: Optional[str] = None
    """Immutable model revision when the source exposes one."""
    modality: str = "text-generation"
    """Verified endpoint modality."""
    context_length: Optional[PositiveInt] = None
    """Token context length, when the modality has token context."""
    created_at: datetime
    service: ServiceConfiguration
    validations: list[EndpointPresetValidation]

    @validator("base", "id", "model", "source", "modality")
    def validate_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must be non-empty")
        return value

    @root_validator
    def validate_preset(cls, values: dict) -> dict:
        service = values.get("service")
        validations = values.get("validations")
        if service is None or validations is None:
            return values
        api_model_name = values.get("api_model_name")
        if api_model_name is None:
            api_model_name = (
                service.model.name if service.model is not None else values.get("base")
            )
            values["api_model_name"] = api_model_name
        if not isinstance(api_model_name, str) or not api_model_name.strip():
            raise ValueError("preset api_model_name must be non-empty")
        if service.model is not None and service.model.name != api_model_name:
            raise ValueError("preset service model name must match api_model_name")
        if service.model is None and not service.probes:
            raise ValueError("non-chat preset service must specify an explicit health probe")
        if any(group.resources is None for group in service.replica_groups):
            raise ValueError("preset service must specify resources")
        if service.name is not None or service.gateway is not None:
            raise ValueError("preset service must not specify name or gateway")
        if any(getattr(service, field) is not None for field in ProfileParams.__fields__):
            raise ValueError("preset service must not specify placement constraints")
        if not validations:
            raise ValueError("preset must include validation evidence")
        for validation in validations:
            if len(validation.replicas) != len(service.replica_groups):
                raise ValueError(
                    "preset validation replicas must match service replica group order"
                )
            if validation.benchmark.target is None or validation.benchmark.client is None:
                raise ValueError("preset benchmark must specify target and client")
            for replica_group in validation.replicas:
                if not replica_group.resources:
                    raise ValueError("preset validation replicas must specify resources")
                for resources in replica_group.resources:
                    _validate_exact_resources(resources)
        return values


class EndpointPresetListOutput(CoreModel):
    presets: list[EndpointPreset]


def _validate_exact_resources(resources: ResourcesSpec) -> None:
    cpu = parse_obj_as(CPUSpec, resources.cpu)
    if not _is_exact(cpu.count) or not _is_exact(resources.memory):
        raise ValueError("preset validation resources must be exact")
    if resources.disk is None or not _is_exact(resources.disk.size):
        raise ValueError("preset validation resources must be exact")
    gpu = resources.gpu
    if gpu is None or not _is_exact(gpu.count):
        raise ValueError("preset validation resources must be exact")
    if gpu.count.min == 0:
        return
    if gpu.name is None or len(gpu.name) != 1 or not _is_exact(gpu.memory):
        raise ValueError("preset validation resources must be exact")


def _is_exact(value) -> bool:
    return (
        value is not None
        and value.min is not None
        and value.max is not None
        and value.min == value.max
    )
