"""
Regression tests for the registered-but-unroutable window: a service job could
be RUNNING while `registered=False`, making the run externally `running` and
its advertised URL unroutable (the in-server proxy only routes to registered
replicas, see `ServerProxyRepo.get_service`). A service run must become
`running` only once the proxy can route to at least one replica.
"""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from dstack._internal.core.models.configurations import (
    ServiceConfiguration,
    TaskConfiguration,
)
from dstack._internal.core.models.runs import JobStatus, RunStatus
from dstack._internal.server.background.pipeline_tasks.runs import RunWorker
from dstack._internal.server.services.proxy.repo import ServerProxyRepo
from dstack._internal.server.services.runs import run_model_to_run
from dstack._internal.server.testing.common import (
    create_job,
    create_project,
    create_repo,
    create_run,
    create_user,
    get_run_spec,
)
from tests._internal.server.background.pipeline_tasks.test_runs.helpers import (
    lock_run,
    run_to_pipeline_item,
)


async def _create_service_run(
    session: AsyncSession,
    *,
    run_status: RunStatus,
    job_registered: bool,
    job_status: JobStatus = JobStatus.RUNNING,
):
    project = await create_project(session=session)
    user = await create_user(session=session)
    repo = await create_repo(session=session, project_id=project.id)
    run_spec = get_run_spec(
        repo_id=repo.name,
        run_name="service-run",
        configuration=ServiceConfiguration(port=8080, commands=["serve"]),
    )
    run = await create_run(
        session=session,
        project=project,
        repo=repo,
        user=user,
        run_name="service-run",
        run_spec=run_spec,
        status=run_status,
    )
    job = await create_job(
        session=session,
        run=run,
        status=job_status,
        registered=job_registered,
    )
    return project, run, job


@pytest.mark.asyncio
@pytest.mark.parametrize("test_db", ["sqlite", "postgres"], indirect=True)
@pytest.mark.usefixtures("image_config_mock")
class TestServiceReadiness:
    async def test_running_unregistered_service_stays_provisioning(
        self, test_db, session: AsyncSession, worker: RunWorker
    ) -> None:
        """The incident window: processes run, proxy cannot route. The run must
        not be advertised as `running`."""
        project, run, job = await _create_service_run(
            session, run_status=RunStatus.PROVISIONING, job_registered=False
        )
        lock_run(run)
        await session.commit()

        await worker.process(run_to_pipeline_item(run))

        await session.refresh(run)
        assert run.status == RunStatus.PROVISIONING

        # The proxy indeed cannot route to this service yet.
        repo = ServerProxyRepo(session=session)
        assert await repo.get_service(project.name, "service-run") is None

    async def test_running_registered_service_becomes_running(
        self, test_db, session: AsyncSession, worker: RunWorker
    ) -> None:
        _, run, _ = await _create_service_run(
            session, run_status=RunStatus.PROVISIONING, job_registered=True
        )
        lock_run(run)
        await session.commit()

        await worker.process(run_to_pipeline_item(run))

        await session.refresh(run)
        assert run.status == RunStatus.RUNNING

    async def test_previously_running_service_downgrades_when_unroutable(
        self, test_db, session: AsyncSession, worker: RunWorker
    ) -> None:
        """A run must not stay externally `running` when no replica is routable
        (e.g. a redeployed submission that has not re-registered)."""
        _, run, _ = await _create_service_run(
            session, run_status=RunStatus.RUNNING, job_registered=False
        )
        lock_run(run)
        await session.commit()

        await worker.process(run_to_pipeline_item(run))

        await session.refresh(run)
        assert run.status == RunStatus.PROVISIONING

    async def test_task_running_does_not_require_registration(
        self, test_db, session: AsyncSession, worker: RunWorker
    ) -> None:
        """Registration is a service concept; tasks become `running` as before."""
        project = await create_project(session=session)
        user = await create_user(session=session)
        repo = await create_repo(session=session, project_id=project.id)
        run_spec = get_run_spec(
            repo_id=repo.name,
            run_name="task-run",
            configuration=TaskConfiguration(commands=["work"]),
        )
        run = await create_run(
            session=session,
            project=project,
            repo=repo,
            user=user,
            run_name="task-run",
            run_spec=run_spec,
            status=RunStatus.PROVISIONING,
        )
        await create_job(session=session, run=run, status=JobStatus.RUNNING, registered=False)
        lock_run(run)
        await session.commit()

        await worker.process(run_to_pipeline_item(run))

        await session.refresh(run)
        assert run.status == RunStatus.RUNNING


@pytest.mark.asyncio
@pytest.mark.usefixtures("test_db", "image_config_mock")
class TestServiceReadinessStatusMessage:
    async def test_status_message_exposes_pending_registration(
        self, session: AsyncSession
    ) -> None:
        _, run, _ = await _create_service_run(
            session, run_status=RunStatus.PROVISIONING, job_registered=False
        )
        await session.refresh(run, attribute_names=["jobs", "project", "user"])
        api_run = run_model_to_run(run, return_in_api=True)
        assert api_run.status_message == "registering"

    async def test_status_message_running_once_registered(self, session: AsyncSession) -> None:
        _, run, _ = await _create_service_run(
            session, run_status=RunStatus.RUNNING, job_registered=True
        )
        await session.refresh(run, attribute_names=["jobs", "project", "user"])
        api_run = run_model_to_run(run, return_in_api=True)
        assert api_run.status_message == "running"
