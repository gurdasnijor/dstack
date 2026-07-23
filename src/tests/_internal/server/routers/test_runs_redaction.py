"""
Regression tests: submitted secrets and environment values must never leave the
server through the Runs API. See `dstack._internal.core.services.redaction`.
"""

import json

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from dstack._internal.core.models.common import RegistryAuth
from dstack._internal.core.models.configurations import ServiceConfiguration
from dstack._internal.core.models.resources import Range
from dstack._internal.core.models.users import GlobalRole, ProjectRole
from dstack._internal.core.services.redaction import SENSITIVE_VALUE_PLACEHOLDER
from dstack._internal.server.services.projects import add_project_member
from dstack._internal.server.services.runs import get_run_spec
from dstack._internal.server.services.runs.spec import validate_run_spec_and_set_defaults
from dstack._internal.server.testing.common import (
    create_job,
    create_project,
    create_repo,
    create_run,
    create_user,
    get_auth_headers,
)
from dstack._internal.server.testing.common import (
    get_run_spec as get_test_run_spec,
)

pytestmark = pytest.mark.usefixtures("image_config_mock")

SECRET_ENV_VALUE = "hf_livesecrettokenvalue0000000000000"
SECRET_REGISTRY_PASSWORD = "registry-password-value"
SECRET_TEMPLATE = "${{ secrets.HF_TOKEN }}"


def _service_configuration(env: dict) -> ServiceConfiguration:
    return ServiceConfiguration(
        type="service",
        image="my/image:latest",
        registry_auth=RegistryAuth(username="registry-user", password=SECRET_REGISTRY_PASSWORD),
        commands=["serve"],
        port=80,
        replicas=Range(min=1, max=1),
        env=env,
    )


async def _create_project_with_run(session: AsyncSession, env: dict, run_name: str = "test-svc"):
    user = await create_user(session=session, global_role=GlobalRole.USER)
    project = await create_project(session=session, owner=user)
    await add_project_member(
        session=session, project=project, user=user, project_role=ProjectRole.USER
    )
    repo = await create_repo(session=session, project_id=project.id)
    run_spec = get_test_run_spec(
        run_name=run_name,
        repo_id=repo.name,
        configuration=_service_configuration(env),
    )
    validate_run_spec_and_set_defaults(user, run_spec)
    run_model = await create_run(
        session=session,
        project=project,
        repo=repo,
        user=user,
        run_name=run_spec.run_name,
        run_spec=run_spec,
    )
    await create_job(session=session, run=run_model)
    return user, project, repo, run_spec, run_model


@pytest.mark.usefixtures("test_db")
class TestGetRunRedaction:
    @pytest.mark.asyncio
    async def test_get_run_redacts_env_and_registry_auth(
        self, session: AsyncSession, client: AsyncClient
    ):
        user, project, _, _, _ = await _create_project_with_run(
            session,
            env={"HF_TOKEN": SECRET_ENV_VALUE, "HF_REF": SECRET_TEMPLATE},
        )
        response = await client.post(
            f"/api/project/{project.name}/runs/get",
            headers=get_auth_headers(user.token),
            json={"run_name": "test-svc"},
        )
        assert response.status_code == 200, response.json()
        assert SECRET_ENV_VALUE not in response.text
        assert SECRET_REGISTRY_PASSWORD not in response.text
        body = response.json()
        run_conf = body["run_spec"]["configuration"]
        assert run_conf["env"] == {
            "HF_TOKEN": SENSITIVE_VALUE_PLACEHOLDER,
            "HF_REF": SECRET_TEMPLATE,
        }
        assert run_conf["registry_auth"]["username"] == "registry-user"
        assert run_conf["registry_auth"]["password"] == SENSITIVE_VALUE_PLACEHOLDER
        job_spec = body["jobs"][0]["job_spec"]
        assert job_spec["env"]["HF_TOKEN"] == SENSITIVE_VALUE_PLACEHOLDER
        assert job_spec["env"]["HF_REF"] == SECRET_TEMPLATE
        assert job_spec["registry_auth"]["password"] == SENSITIVE_VALUE_PLACEHOLDER

    @pytest.mark.asyncio
    async def test_list_runs_redacts_env(self, session: AsyncSession, client: AsyncClient):
        user, _, _, _, _ = await _create_project_with_run(
            session, env={"HF_TOKEN": SECRET_ENV_VALUE}
        )
        response = await client.post(
            "/api/runs/list",
            headers=get_auth_headers(user.token),
            json={},
        )
        assert response.status_code == 200, response.json()
        assert SECRET_ENV_VALUE not in response.text
        assert SECRET_REGISTRY_PASSWORD not in response.text
        body = response.json()
        assert body[0]["run_spec"]["configuration"]["env"]["HF_TOKEN"] == (
            SENSITIVE_VALUE_PLACEHOLDER
        )
        assert body[0]["jobs"][0]["job_spec"]["env"]["HF_TOKEN"] == SENSITIVE_VALUE_PLACEHOLDER


@pytest.mark.usefixtures("test_db")
class TestGetPlanRedaction:
    @pytest.mark.asyncio
    async def test_get_plan_restores_unchanged_and_redacts_changed_env(
        self, session: AsyncSession, client: AsyncClient
    ):
        user, project, _, run_spec, _ = await _create_project_with_run(
            session,
            env={"KEEP": "unchanged-value", "ROTATED": "old-secret-value"},
        )
        new_run_spec = run_spec.copy(deep=True)
        new_run_spec.configuration.env["ROTATED"] = "new-secret-value"
        new_run_spec.configuration.registry_auth = RegistryAuth(
            username="registry-user", password="new-registry-password"
        )
        response = await client.post(
            f"/api/project/{project.name}/runs/get_plan",
            headers=get_auth_headers(user.token),
            json={"run_spec": json.loads(new_run_spec.json())},
        )
        assert response.status_code == 200, response.json()
        # Stored values the requester does not know must not be revealed.
        # (The requester's own submitted values are echoed in run_spec /
        # effective_run_spec; that is not a leak.)
        assert "old-secret-value" not in response.text
        assert SECRET_REGISTRY_PASSWORD not in response.text
        body = response.json()
        current_conf = body["current_resource"]["run_spec"]["configuration"]
        current_env = current_conf["env"]
        # Values the requester provably knows (equal to its own submission) stay
        # exact so client-side "no changes" detection keeps working.
        assert current_env["KEEP"] == "unchanged-value"
        assert current_env["ROTATED"] == SENSITIVE_VALUE_PLACEHOLDER
        assert current_conf["registry_auth"]["password"] == SENSITIVE_VALUE_PLACEHOLDER
        # Job plans must be redacted too.
        job_plan_env = body["job_plans"][0]["job_spec"]["env"]
        assert job_plan_env["KEEP"] == SENSITIVE_VALUE_PLACEHOLDER
        assert job_plan_env["ROTATED"] == SENSITIVE_VALUE_PLACEHOLDER

    @pytest.mark.asyncio
    async def test_get_plan_identical_spec_restores_all_env_values(
        self, session: AsyncSession, client: AsyncClient
    ):
        user, project, _, run_spec, _ = await _create_project_with_run(
            session, env={"HF_TOKEN": SECRET_ENV_VALUE}
        )
        response = await client.post(
            f"/api/project/{project.name}/runs/get_plan",
            headers=get_auth_headers(user.token),
            json={"run_spec": json.loads(run_spec.json())},
        )
        assert response.status_code == 200, response.json()
        body = response.json()
        # An unchanged spec must compare equal (diff is None client-side), so
        # the requester's own values are preserved in current_resource.
        current = body["current_resource"]["run_spec"]["configuration"]
        assert current["env"]["HF_TOKEN"] == SECRET_ENV_VALUE
        assert current["registry_auth"]["password"] == SECRET_REGISTRY_PASSWORD


@pytest.mark.usefixtures("test_db")
class TestApplyPlanRedaction:
    @pytest.mark.asyncio
    async def test_in_place_env_update_with_redacted_current_resource(
        self, session: AsyncSession, client: AsyncClient
    ):
        user, project, _, run_spec, run_model = await _create_project_with_run(
            session, env={"HF_TOKEN": "old-secret-value"}
        )
        new_run_spec = run_spec.copy(deep=True)
        new_run_spec.configuration.env["HF_TOKEN"] = "new-secret-value"
        plan_response = await client.post(
            f"/api/project/{project.name}/runs/get_plan",
            headers=get_auth_headers(user.token),
            json={"run_spec": json.loads(new_run_spec.json())},
        )
        assert plan_response.status_code == 200, plan_response.json()
        plan_body = plan_response.json()
        assert plan_body["action"] == "update"
        # Apply exactly what a client would echo back: the redacted plan output.
        response = await client.post(
            f"/api/project/{project.name}/runs/apply",
            headers=get_auth_headers(user.token),
            json={
                "plan": {
                    "run_spec": json.loads(new_run_spec.json()),
                    "current_resource": plan_body["current_resource"],
                },
                "force": False,
            },
        )
        assert response.status_code == 200, response.json()
        assert "old-secret-value" not in response.text
        assert "new-secret-value" not in response.text
        # The server must store the real submitted value, not the placeholder.
        await session.refresh(run_model)
        stored_spec = get_run_spec(run_model)
        assert stored_spec.configuration.env["HF_TOKEN"] == "new-secret-value"

    @pytest.mark.asyncio
    async def test_run_model_keeps_cleartext_values(self, session: AsyncSession):
        # Redaction happens at API serialization time only; the stored spec and
        # the runner-facing serialization keep the real values.
        _, project, _, _, run_model = await _create_project_with_run(
            session, env={"HF_TOKEN": SECRET_ENV_VALUE}
        )
        stored_spec = get_run_spec(run_model)
        assert stored_spec.configuration.env["HF_TOKEN"] == SECRET_ENV_VALUE

        from dstack._internal.server.services.runs import run_model_to_run

        await session.refresh(run_model, attribute_names=["jobs", "project", "user"])
        sensitive_run = run_model_to_run(run_model, include_sensitive=True)
        assert sensitive_run.run_spec.configuration.env["HF_TOKEN"] == SECRET_ENV_VALUE
        assert sensitive_run.jobs[0].job_spec.env["HF_TOKEN"] == SECRET_ENV_VALUE
