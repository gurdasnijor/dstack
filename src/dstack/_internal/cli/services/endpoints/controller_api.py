"""The production `ControllerAPI` adapter over the dstack client.

Only the parent CLI process holds the dstack client and its token; the agent
reaches these operations exclusively through the controller RPC.
"""

import itertools
from typing import Any, Mapping, Optional, Sequence

import pydantic
import requests

from dstack._internal.cli.services.endpoints.controller import ControllerError, RunInfo
from dstack._internal.core.errors import ConfigurationError
from dstack._internal.core.models.configurations import AnyRunConfiguration
from dstack._internal.core.models.envs import EnvSentinel
from dstack._internal.core.models.runs import Run as RunModel
from dstack.api import Client

_MAX_LOG_BYTES = 256 * 1024
_HTTP_TIMEOUT_SECONDS = 15 * 60
_MAX_HTTP_RESPONSE_BYTES = 64 * 1024 * 1024


def _run_info_from_model(run: RunModel) -> RunInfo:
    price: Optional[float] = None
    instance_id: Optional[str] = None
    for job in run.jobs:
        for submission in job.job_submissions:
            runtime_data = submission.job_runtime_data
            if runtime_data is not None and runtime_data.offer is not None:
                price = runtime_data.offer.price
            provisioning = submission.job_provisioning_data
            if provisioning is not None:
                instance_id = getattr(provisioning, "instance_id", None) or instance_id
    service_url = None
    if run.service is not None:
        service_url = run.service.url
    return RunInfo(
        name=run.run_spec.run_name or "",
        run_id=str(run.id),
        status=run.status.value,
        is_finished=run.status.is_finished(),
        service_url=service_url,
        price_per_hour=price,
        instance_id=instance_id,
    )


class DstackControllerAPI:
    """Implements the controller's API protocol against a live dstack server."""

    def __init__(
        self,
        api: Client,
        *,
        endpoint_env: Mapping[str, str],
        service_token: str,
        external_base_url: Optional[str] = None,
    ) -> None:
        self._api = api
        self._endpoint_env = dict(endpoint_env)
        self._service_token = service_token
        self._external_base_url = external_base_url or api.client.base_url

    def submit(self, configuration: dict, run_name: str) -> RunInfo:
        try:
            parsed = pydantic.parse_obj_as(AnyRunConfiguration, configuration)  # type: ignore[arg-type]
        except pydantic.ValidationError as e:
            raise ControllerError(
                f"candidate configuration is invalid: {e}", failure_class="validation"
            ) from e
        parsed.name = run_name
        for key, value in parsed.env.items():
            if isinstance(value, EnvSentinel) and key in self._endpoint_env:
                parsed.env[key] = self._endpoint_env[key]
        try:
            self._api.runs.apply_configuration(configuration=parsed, reserve_ports=False)
        except ConfigurationError as e:
            raise ControllerError(
                f"candidate configuration was rejected: {e}", failure_class="validation"
            ) from e
        model = self._api.client.runs.get(self._api.project, run_name)
        if model is None:
            raise ControllerError(
                f"run {run_name} was submitted but cannot be fetched from the server",
                failure_class="controller",
            )
        return _run_info_from_model(model)

    def get(self, run_name: str) -> Optional[RunInfo]:
        model = self._api.client.runs.get(self._api.project, run_name)
        if model is None:
            return None
        return _run_info_from_model(model)

    def stop(self, run_names: Sequence[str], *, abort: bool = False) -> None:
        self._api.client.runs.stop(self._api.project, list(run_names), abort=abort)

    def logs(self, run_name: str) -> str:
        run = self._api.runs.get(run_name)
        if run is None:
            return ""
        collected = bytearray()
        for chunk in itertools.islice(run.logs(), 4096):
            collected.extend(chunk)
            if len(collected) >= _MAX_LOG_BYTES:
                break
        return collected[-_MAX_LOG_BYTES:].decode(errors="replace")

    def offers(self, filters: Mapping[str, Any]) -> list[dict]:
        configuration = filters.get("configuration")
        if not isinstance(configuration, dict):
            raise ControllerError(
                "list_offers requires {'configuration': <task/service mapping>} to "
                "evaluate offers against a concrete resource request",
                failure_class="validation",
            )
        try:
            parsed = pydantic.parse_obj_as(AnyRunConfiguration, configuration)  # type: ignore[arg-type]
        except pydantic.ValidationError as e:
            raise ControllerError(
                f"offer configuration is invalid: {e}", failure_class="validation"
            ) from e
        plan = self._api.runs.get_run_plan(configuration=parsed)
        offers: list[dict] = []
        for job_plan in plan.job_plans:
            for offer in job_plan.offers:
                offers.append(
                    {
                        "backend": str(offer.backend.value),
                        "region": offer.region,
                        "instance": offer.instance.name,
                        "price": offer.price,
                        "availability": offer.availability.value,
                        "resources": offer.instance.resources.pretty_format(),
                    }
                )
        return offers

    def http_request(
        self,
        service_url: str,
        *,
        method: str,
        path: str,
        body: Optional[bytes],
        headers: Mapping[str, str],
    ) -> dict:
        import base64

        url = service_url if not path else service_url.rstrip("/") + "/" + path.lstrip("/")
        if url.startswith("/"):
            url = self._external_base_url.rstrip("/") + url
        request_headers = dict(headers)
        request_headers["Authorization"] = f"Bearer {self._service_token}"
        response = requests.request(
            method=method.upper(),
            url=url,
            data=body,
            headers=request_headers,
            timeout=_HTTP_TIMEOUT_SECONDS,
            stream=True,
        )
        content = response.raw.read(_MAX_HTTP_RESPONSE_BYTES, decode_content=True)
        return {
            "status": response.status_code,
            "headers": {
                key: value
                for key, value in response.headers.items()
                if key.lower() in {"content-type", "content-length"}
            },
            "body_base64": base64.b64encode(content).decode(),
        }


__all__ = ["DstackControllerAPI"]
