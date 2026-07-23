import uuid
from typing import Optional

from pydantic import PositiveInt, root_validator

from dstack._internal.cli.models.endpoint_presets import EndpointBenchmark
from dstack._internal.core.models.common import CoreModel

_LATENCY_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "mean": {"type": "number", "minimum": 0},
        "p50": {"type": "number", "minimum": 0},
        "p99": {"type": "number", "minimum": 0},
    },
    "required": ["mean", "p50", "p99"],
    "additionalProperties": False,
}

_BENCHMARK_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "tool": {"type": "string", "minLength": 1},
        "tool_version": {"type": "string", "minLength": 1},
        "command": {"type": "string", "minLength": 1},
        "workload": {
            "type": "object",
            "properties": {
                "api": {
                    "type": "string",
                    "minLength": 1,
                },
                "request_path": {"type": "string", "minLength": 1},
                "num_requests": {"type": "integer", "minimum": 1},
                "input_tokens": {"type": "integer", "minimum": 1},
                "output_tokens": {"type": "integer", "minimum": 2},
                "concurrency": {"type": "integer", "minimum": 1},
                "width": {"type": "integer", "minimum": 1},
                "height": {"type": "integer", "minimum": 1},
                "num_inference_steps": {"type": "integer", "minimum": 1},
                "outputs_per_request": {"type": "integer", "minimum": 1},
                "output_unit": {"type": "string", "minLength": 1},
                "parameters": {"type": "object"},
            },
            "required": [
                "api",
                "num_requests",
                "concurrency",
            ],
            "additionalProperties": False,
        },
        "metrics": {
            "type": "object",
            "properties": {
                "successful_requests": {"type": "integer", "minimum": 0},
                "failed_requests": {"type": "integer", "minimum": 0},
                "duration_seconds": {"type": "number", "exclusiveMinimum": 0},
                "latency_ms": _LATENCY_JSON_SCHEMA,
                "total_outputs": {"type": "integer", "minimum": 0},
                "total_output_bytes": {"type": "integer", "minimum": 0},
                "total_input_tokens": {"type": "integer", "minimum": 0},
                "total_output_tokens": {"type": "integer", "minimum": 0},
                "ttft_ms": _LATENCY_JSON_SCHEMA,
                "tpot_ms": _LATENCY_JSON_SCHEMA,
            },
            "required": [
                "successful_requests",
                "failed_requests",
                "duration_seconds",
            ],
            "additionalProperties": False,
        },
    },
    "required": ["tool", "tool_version", "command", "workload", "metrics"],
    "additionalProperties": False,
}

AGENT_FINAL_REPORT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "success": {"type": "boolean"},
        "run_id": {"type": "string"},
        "run_name": {"type": "string"},
        "service_yaml": {"type": "string"},
        "base": {"type": "string"},
        "model": {"type": "string"},
        "api_model_name": {"type": "string"},
        "source": {"type": "string"},
        "revision": {"type": "string"},
        "modality": {"type": "string"},
        "context_length": {"type": "integer", "minimum": 1},
        "benchmark": _BENCHMARK_JSON_SCHEMA,
        "failure_summary": {"type": "string"},
    },
    "required": ["success"],
    "additionalProperties": False,
}


class AgentFinalReport(CoreModel):
    success: bool
    run_id: Optional[uuid.UUID] = None
    run_name: Optional[str] = None
    service_yaml: Optional[str] = None
    base: Optional[str] = None
    model: Optional[str] = None
    api_model_name: Optional[str] = None
    source: Optional[str] = None
    revision: Optional[str] = None
    modality: Optional[str] = None
    context_length: Optional[PositiveInt] = None
    benchmark: Optional[EndpointBenchmark] = None
    failure_summary: Optional[str] = None

    @root_validator
    def validate_report(cls, values: dict) -> dict:
        if values.get("success"):
            required = (
                "run_id",
                "run_name",
                "service_yaml",
                "base",
                "model",
                "api_model_name",
                "source",
                "modality",
                "benchmark",
            )
            missing = [field for field in required if values.get(field) in (None, "")]
            if missing:
                raise ValueError("successful agent report must include " + ", ".join(missing))
            if values.get("source") == "auto" or values.get("modality") == "auto":
                raise ValueError("successful agent report must resolve source and modality")
            if values.get("source") == "huggingface" and values.get("revision") is None:
                raise ValueError("successful Hugging Face report must include revision")
            benchmark = values.get("benchmark")
            if (
                benchmark is not None
                and benchmark.workload.api in {"chat_completions", "completions"}
                and values.get("context_length") is None
            ):
                raise ValueError("successful token endpoint report must include context_length")
        elif not values.get("failure_summary"):
            raise ValueError("failed agent report must include failure_summary")
        return values
