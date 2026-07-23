from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from dstack._internal.cli.models.endpoint_agent import AgentFinalReport
from dstack._internal.cli.models.endpoints import EndpointConfiguration
from dstack._internal.cli.services.endpoints.verify import (
    build_verified_endpoint_preset,
)
from dstack._internal.core.errors import CLIError
from dstack._internal.core.models.envs import EnvSentinel
from dstack._internal.core.models.profiles import ProfileParams
from tests._internal.cli.endpoint_presets import (
    get_running_image_service_run,
    get_running_service_run,
    get_successful_endpoint_report,
    get_successful_image_report,
)

pytestmark = pytest.mark.windows


class TestBuildVerifiedEndpointPreset:
    def test_successful_report_requires_benchmark(self):
        run = get_running_service_run()
        data = get_successful_endpoint_report(run).dict()
        data.pop("benchmark")

        with pytest.raises(ValidationError, match="benchmark"):
            AgentFinalReport.parse_obj(data)

    def test_builds_portable_self_contained_preset(self):
        run = get_running_service_run()
        created_at = datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc)

        with patch(
            "dstack._internal.cli.services.endpoints.presets.get_current_datetime",
            return_value=created_at,
        ):
            preset = build_verified_endpoint_preset(
                run=run,
                endpoint_configuration=EndpointConfiguration(
                    name="qwen-build",
                    model={"base": "Qwen/Qwen3.5-27B"},
                    context_length=8192,
                    gateway="benchmark-gateway",
                    env=["LICENSE", "TOKENIZERS_PARALLELISM=false"],
                ),
                report=get_successful_endpoint_report(run),
            )

        assert preset.base == "Qwen/Qwen3.5-27B"
        assert preset.model == "community/Qwen3.5-27B-GPTQ-Int4"
        assert preset.context_length == 32768
        assert preset.created_at == created_at
        assert preset.service.name is None
        assert preset.service.gateway is None
        assert all(getattr(preset.service, field) is None for field in ProfileParams.__fields__)
        assert isinstance(preset.service.env["LICENSE"], EnvSentinel)
        assert preset.service.env["TOKENIZERS_PARALLELISM"] == "false"
        assert preset.service.resources.gpu.vendor.value == "nvidia"
        validation = preset.validations[0]
        assert validation.replicas[0].resources[0].gpu.name == ["A6000"]
        assert validation.benchmark.target.type == "server-proxy"
        assert validation.benchmark.client.type == "local"

    def test_rejects_variant_for_exact_model_request(self):
        run = get_running_service_run()
        report = get_successful_endpoint_report(run).copy(update={"model": "other/model"})

        with pytest.raises(CLIError, match="changed an exact model request"):
            build_verified_endpoint_preset(
                run=run,
                endpoint_configuration=EndpointConfiguration(
                    name="qwen-build",
                    model={
                        "repo": "community/Qwen3.5-27B-GPTQ-Int4",
                        "name": "Qwen/Qwen3.5-27B",
                    },
                ),
                report=report,
            )

    def test_rejects_reported_revision_not_used_by_service(self):
        run = get_running_service_run()
        run.run_spec.configuration.commands = ["vllm serve community/Qwen3.5-27B-GPTQ-Int4"]

        with pytest.raises(CLIError, match="does not pin the reported model revision"):
            build_verified_endpoint_preset(
                run=run,
                endpoint_configuration=EndpointConfiguration(
                    name="qwen-build",
                    model={"base": "Qwen/Qwen3.5-27B"},
                ),
                report=get_successful_endpoint_report(run),
            )

    def test_builds_non_chat_image_preset_with_explicit_probe(self):
        run = get_running_image_service_run()

        preset = build_verified_endpoint_preset(
            run=run,
            endpoint_configuration=EndpointConfiguration(
                name="juggernaut-build",
                model={
                    "repo": "eniora/Juggernaut_XL_Ragnarok",
                    "source": "huggingface",
                    "modality": "image-generation",
                },
                gateway=False,
                env=["HF_TOKEN"],
            ),
            report=get_successful_image_report(run),
        )

        assert preset.api_model_name == "eniora/Juggernaut_XL_Ragnarok"
        assert preset.modality == "image-generation"
        assert preset.source == "huggingface"
        assert preset.revision == "fe71bb49af337c43faf10a6b50b0dd1d10b23015"
        assert preset.context_length is None
        assert preset.service.model is None
        assert preset.service.probes[0].url == "/health"
        assert preset.validations[0].benchmark.workload.api == "images_generations"

    def test_tightens_gpu_floor_to_successfully_validated_memory(self):
        run = get_running_image_service_run()
        validated_gpu = (
            run.jobs[0].job_submissions[0].job_runtime_data.offer.instance.resources.gpus[0]
        )
        validated_gpu.memory_mib = 40 * 1024

        preset = build_verified_endpoint_preset(
            run=run,
            endpoint_configuration=EndpointConfiguration(
                name="juggernaut-build",
                model={
                    "repo": "eniora/Juggernaut_XL_Ragnarok",
                    "source": "huggingface",
                    "modality": "image-generation",
                },
                env=["HF_TOKEN"],
            ),
            report=get_successful_image_report(run),
        )

        assert preset.service.resources.gpu.memory.min == 40
        assert preset.service.resources.gpu.memory.max is None
        assert preset.validations[0].replicas[0].resources[0].gpu.memory.min == 40

    def test_rejects_non_chat_service_without_probe(self):
        run = get_running_image_service_run()
        run.run_spec.configuration.probes = None

        with pytest.raises(CLIError, match="no explicit health probe"):
            build_verified_endpoint_preset(
                run=run,
                endpoint_configuration=EndpointConfiguration(
                    name="juggernaut-build",
                    model="eniora/Juggernaut_XL_Ragnarok",
                ),
                report=get_successful_image_report(run),
            )
