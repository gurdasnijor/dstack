"""
Regression test: applying a saved endpoint preset must not depend on the agent
workspace or any trial-local state that existed when the preset was created.
The saved artifact (preset YAML + staged assets) must be fully self-contained.
"""

import shutil
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from dstack._internal.cli.models.endpoint_presets import (
    EndpointPreset,
    EndpointPresetValidation,
    EndpointPresetValidationReplica,
)
from dstack._internal.cli.models.endpoints import EndpointConfiguration
from dstack._internal.cli.services.endpoints.apply import apply_endpoint_preset
from dstack._internal.cli.services.endpoints.store import EndpointPresetStore
from dstack._internal.core.models.configurations import ServiceConfiguration
from dstack._internal.core.models.instances import InstanceAvailability
from dstack._internal.core.models.resources import ResourcesSpec
from tests._internal.cli.endpoint_presets import get_endpoint_benchmark

pytestmark = pytest.mark.windows

GENERATED_SERVER_CODE = "print('generated model server')\n"


def _make_preset(workspace: Path) -> EndpointPreset:
    resources = ResourcesSpec.parse_obj(
        {
            "cpu": "16",
            "memory": "64GB",
            "disk": "200GB",
            "gpu": {"name": "A6000", "memory": "48GB", "count": 1},
        }
    )
    service = ServiceConfiguration.parse_obj(
        {
            "image": "vllm/vllm-openai:v0.11.0",
            "commands": ["python /app/server.py"],
            "port": 8000,
            "model": "Qwen/Qwen3.5-27B",
            "resources": {"gpu": "nvidia:40GB..48GB:1"},
            "env": ["HF_TOKEN"],
            # A generated file in the (temporary) agent workspace, as the
            # preset-creation flow produces for custom runtimes.
            "files": [f"{workspace / 'server.py'}:/app/server.py"],
        }
    )
    return EndpointPreset(
        base="Qwen/Qwen3.5-27B",
        id="selfcontained1",
        model="community/Qwen3.5-27B-GPTQ-Int4",
        context_length=32768,
        created_at=datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc),
        service=service,
        validations=[
            EndpointPresetValidation(
                replicas=[EndpointPresetValidationReplica(resources=[resources])],
                benchmark=get_endpoint_benchmark(),
            )
        ],
    )


class TestPresetSelfContainment:
    def test_apply_succeeds_after_agent_workspace_deleted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        # 1. Create: an agent workspace with a generated file, saved as a preset.
        workspace = tmp_path / "agent-workspace"
        workspace.mkdir()
        (workspace / "server.py").write_text(GENERATED_SERVER_CODE)
        store_root = tmp_path / "presets"
        EndpointPresetStore(root=store_root).save(_make_preset(workspace))

        # 2. Delete all trial-local state: the agent workspace is gone.
        shutil.rmtree(workspace)

        # 3. Apply from the saved artifact only, with a fresh store instance
        # (as a new CLI process would) and a fake API/configurator.
        store = EndpointPresetStore(root=store_root)
        run_plan = SimpleNamespace(
            job_plans=[
                SimpleNamespace(
                    offers=[SimpleNamespace(availability=InstanceAvailability.AVAILABLE)]
                )
            ]
        )
        repo = Mock()
        service_args = SimpleNamespace(profile=None)
        configurator = Mock()
        configurator.get_parser.return_value.parse_args.return_value = service_args
        configurator.get_plan.return_value = (run_plan, repo)
        monkeypatch.setattr(
            "dstack._internal.cli.services.endpoints.apply.ServiceConfigurator",
            lambda api_client: configurator,
        )

        apply_endpoint_preset(
            api=Mock(),
            configuration=EndpointConfiguration(
                name="qwen",
                model={"base": "Qwen/Qwen3.5-27B"},
                env={"HF_TOKEN": "runtime-token"},
            ),
            configuration_path="endpoint.dstack.yml",
            preset_id=None,
            profile_name=None,
            command_args=SimpleNamespace(),
            store=store,
        )

        # 4. The plan was built and applied without the workspace.
        configurator.apply_plan.assert_called_once()
        planned_service = configurator.get_plan.call_args.kwargs["conf"]

        # Every referenced file resolves inside the preset store, exists, and
        # has the content captured at creation time.
        assert planned_service.files, "preset service lost its file mappings"
        for mapping in planned_service.files:
            local_path = Path(mapping.local_path)
            assert local_path.is_absolute()
            assert store_root in local_path.parents, f"{local_path} is outside the preset store"
            assert str(workspace) not in str(local_path)
            assert local_path.read_text() == GENERATED_SERVER_CODE
        assert planned_service.files[0].path == "/app/server.py"

        # Runtime env supplied at apply time is merged in.
        assert planned_service.env["HF_TOKEN"] == "runtime-token"

    def test_saved_preset_yaml_has_no_workspace_paths(self, tmp_path: Path):
        workspace = tmp_path / "agent-workspace"
        workspace.mkdir()
        (workspace / "server.py").write_text(GENERATED_SERVER_CODE)
        store_root = tmp_path / "presets"
        path = EndpointPresetStore(root=store_root).save(_make_preset(workspace))

        # The stored artifact must not reference the workspace by path.
        assert str(workspace) not in path.read_text()

        # The staged asset is a real copy, not a link into the workspace.
        shutil.rmtree(workspace)
        preset = EndpointPresetStore(root=store_root).get("selfcontained1")
        assert preset is not None
        asset_path = Path(preset.service.files[0].local_path)
        assert asset_path.exists()
        assert asset_path.read_text() == GENERATED_SERVER_CODE
