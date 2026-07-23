import base64
import json
import time
from typing import AsyncIterator, Optional

import httpx
from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import Response, StreamingResponse
from typing_extensions import Annotated

from dstack._internal.proxy.lib.deps import ProxyAuth, get_proxy_repo, get_service_connection_pool
from dstack._internal.proxy.lib.errors import ProxyError, UnexpectedProxyError
from dstack._internal.proxy.lib.models import EndpointModel, Service
from dstack._internal.proxy.lib.repo import BaseProxyRepo
from dstack._internal.proxy.lib.schemas.model_proxy import (
    ChatCompletionsChoice,
    ChatCompletionsChunk,
    ChatCompletionsChunkChoice,
    ChatCompletionsRequest,
    ChatCompletionsResponse,
    ChatCompletionsUsage,
    ChatMessage,
    Model,
    ModelsResponse,
)
from dstack._internal.proxy.lib.services.model_proxy.model_proxy import get_chat_client
from dstack._internal.proxy.lib.services.service_connection import (
    ServiceConnectionPool,
    get_service_replica_client,
)

router = APIRouter(dependencies=[Depends(ProxyAuth(auto_enforce=True))])


@router.get("/{project_name}/models")
async def get_models(
    project_name: str, repo: Annotated[BaseProxyRepo, Depends(get_proxy_repo)]
) -> ModelsResponse:
    models = await repo.list_models(project_name)
    data = [
        Model(
            id=m.name,
            created=int(m.created_at.timestamp()),
            owned_by=project_name,
            base=m.base,
            model=m.model,
            source=m.source,
            revision=m.revision,
            modality=m.modality,
            context_length=m.context_length,
            api=m.api,
            request_path=m.request_path,
            output_unit=m.output_unit,
        )
        for m in models
    ]
    return ModelsResponse(data=data)


@router.post("/{project_name}/chat/completions", response_model=ChatCompletionsResponse)
async def post_chat_completions(
    project_name: str,
    body: ChatCompletionsRequest,
    repo: Annotated[BaseProxyRepo, Depends(get_proxy_repo)],
    service_conn_pool: Annotated[ServiceConnectionPool, Depends(get_service_connection_pool)],
):
    model = await repo.get_model(project_name, body.model)
    model, service = await _resolve_model_service(project_name, body.model, model, repo)
    http_client = await get_service_replica_client(service, repo, service_conn_pool)
    if model.format_spec is None:
        return await _post_endpoint_chat(model, body, http_client, project_name)
    client = get_chat_client(model, http_client)
    if not body.stream:
        return await client.generate(body)
    else:
        return StreamingResponse(
            await StreamingAdaptor(client.stream(body)).get_stream(),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no"},
        )


@router.post("/{project_name}/images/generations")
async def post_image_generations(
    project_name: str,
    request: Request,
    repo: Annotated[BaseProxyRepo, Depends(get_proxy_repo)],
    service_conn_pool: Annotated[ServiceConnectionPool, Depends(get_service_connection_pool)],
) -> Response:
    body = await _json_body(request)
    model_name = _required_model(body)
    model = await repo.get_model(project_name, model_name)
    model, service = await _resolve_model_service(project_name, model_name, model, repo)
    _require_api(model, {"images_generations"})
    return await _forward_json(model, service, body, repo, service_conn_pool)


@router.post("/{project_name}/videos")
@router.post("/{project_name}/video/generations")
async def post_videos(
    project_name: str,
    request: Request,
    repo: Annotated[BaseProxyRepo, Depends(get_proxy_repo)],
    service_conn_pool: Annotated[ServiceConnectionPool, Depends(get_service_connection_pool)],
) -> Response:
    body = await _json_body(request)
    model_name = _required_model(body)
    model = await repo.get_model(project_name, model_name)
    model, service = await _resolve_model_service(project_name, model_name, model, repo)
    _require_api(model, {"videos", "video_generations"})
    response = await _request_endpoint(model, service, "POST", body, repo, service_conn_pool)
    return _video_create_response(model, response)


@router.get("/{project_name}/videos/{video_id}")
async def get_video(
    project_name: str,
    video_id: str,
    repo: Annotated[BaseProxyRepo, Depends(get_proxy_repo)],
    service_conn_pool: Annotated[ServiceConnectionPool, Depends(get_service_connection_pool)],
) -> Response:
    return await _get_video_resource(project_name, video_id, "", repo, service_conn_pool)


@router.get("/{project_name}/videos/{video_id}/content")
async def get_video_content(
    project_name: str,
    video_id: str,
    repo: Annotated[BaseProxyRepo, Depends(get_proxy_repo)],
    service_conn_pool: Annotated[ServiceConnectionPool, Depends(get_service_connection_pool)],
) -> Response:
    return await _get_video_resource(project_name, video_id, "/content", repo, service_conn_pool)


async def _post_endpoint_chat(
    model: EndpointModel,
    body: ChatCompletionsRequest,
    http_client: httpx.AsyncClient,
    project_name: str,
):
    if model.api in {"chat_completions", "completions"}:
        return await _forward_chat(model, body, http_client)
    if model.api == "images_generations":
        content = await _generate_image_content(model, body, http_client)
    elif model.api in {"videos", "video_generations"}:
        content = await _generate_video_content(model, body, http_client, project_name)
    else:
        raise ProxyError(
            f"Model {model.name} uses unsupported API {model.api!r}",
            status.HTTP_422_UNPROCESSABLE_ENTITY,
        )
    response = _projected_chat_response(model.name, content)
    if not body.stream:
        return response
    return StreamingResponse(
        _single_response_stream(response),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no"},
    )


async def _forward_chat(
    model: EndpointModel,
    body: ChatCompletionsRequest,
    http_client: httpx.AsyncClient,
):
    if not body.stream:
        response = await _post_json(
            http_client,
            _request_path(model),
            body.dict(exclude_unset=True),
        )
        return Response(
            content=response.content,
            status_code=response.status_code,
            media_type=response.headers.get("content-type"),
        )
    response = await _open_stream(
        http_client,
        _request_path(model),
        body.dict(exclude_unset=True),
    )
    return StreamingResponse(
        response.aiter_raw(),
        status_code=response.status_code,
        media_type=response.headers.get("content-type", "text/event-stream"),
        headers={"X-Accel-Buffering": "no"},
        background=response.aclose,
    )


async def _open_stream(
    client: httpx.AsyncClient,
    path: str,
    body: dict,
) -> httpx.Response:
    try:
        request = client.build_request("POST", path, json=body)
        response = await client.send(request, stream=True)
    except httpx.RequestError as e:
        raise ProxyError(f"Error requesting model: {e!r}", status.HTTP_502_BAD_GATEWAY)
    if response.status_code >= 400:
        content = await response.aread()
        await response.aclose()
        raise ProxyError(
            content.decode(errors="replace"),
            response.status_code,
        )
    return response


async def _generate_image_content(
    model: EndpointModel,
    body: ChatCompletionsRequest,
    http_client: httpx.AsyncClient,
) -> str:
    payload = {
        "model": model.name,
        "prompt": _last_user_prompt(body),
        "n": 1,
        "response_format": "b64_json",
    }
    response = await _post_json(http_client, _request_path(model), payload)
    data = _response_json(response).get("data", [])
    if not data:
        raise ProxyError("Image endpoint returned no images", status.HTTP_502_BAD_GATEWAY)
    image = data[0]
    if image.get("url"):
        return f"![Generated image]({image['url']})"
    if image.get("b64_json"):
        return f"![Generated image](data:image/png;base64,{image['b64_json']})"
    raise ProxyError("Image endpoint returned no usable image", status.HTTP_502_BAD_GATEWAY)


async def _generate_video_content(
    model: EndpointModel,
    body: ChatCompletionsRequest,
    http_client: httpx.AsyncClient,
    project_name: str,
) -> str:
    payload = {"model": model.name, "prompt": _last_user_prompt(body)}
    response = await _post_json(http_client, _request_path(model), payload)
    result = _response_json(response)
    url = result.get("url") or result.get("video_url")
    if url is None and result.get("data"):
        url = result["data"][0].get("url")
    if url is not None:
        return f"[Generated video]({url})"
    if result.get("id"):
        video_id = _encode_video_id(model.name, result["id"])
        return (
            f"Video generation started with status `{result.get('status', 'processing')}`. "
            f"[Download video when ready]"
            f"(/proxy/models/{project_name}/videos/{video_id}/content)"
        )
    raise ProxyError("Video endpoint returned no usable video", status.HTTP_502_BAD_GATEWAY)


def _projected_chat_response(model_name: str, content: str) -> ChatCompletionsResponse:
    return ChatCompletionsResponse(
        id=f"chatcmpl-dstack-{int(time.time() * 1000)}",
        choices=[
            ChatCompletionsChoice(
                finish_reason="stop",
                index=0,
                message=ChatMessage(role="assistant", content=content),
            )
        ],
        created=int(time.time()),
        model=model_name,
        usage=ChatCompletionsUsage(completion_tokens=0, prompt_tokens=0, total_tokens=0),
    )


async def _single_response_stream(
    response: ChatCompletionsResponse,
) -> AsyncIterator[bytes]:
    chunk = ChatCompletionsChunk(
        id=response.id,
        choices=[
            ChatCompletionsChunkChoice(
                finish_reason="stop",
                index=0,
                delta={
                    "role": "assistant",
                    "content": response.choices[0].message.content,
                },
            )
        ],
        created=response.created,
        model=response.model,
    )
    yield f"data:{chunk.json()}\n\n".encode()
    yield b"data: [DONE]\n\n"


async def _get_video_resource(
    project_name: str,
    video_id: str,
    suffix: str,
    repo: BaseProxyRepo,
    service_conn_pool: ServiceConnectionPool,
) -> Response:
    model_name, upstream_id = _decode_video_id(video_id)
    model = await repo.get_model(project_name, model_name)
    model, service = await _resolve_model_service(project_name, model_name, model, repo)
    _require_api(model, {"videos", "video_generations"})
    http_client = await get_service_replica_client(service, repo, service_conn_pool)
    response = await _request(
        http_client,
        "GET",
        f"{_request_path(model).rstrip('/')}/{upstream_id}{suffix}",
    )
    if not suffix:
        return _video_resource_response(video_id, response)
    return _http_response(response)


async def _forward_json(
    model: EndpointModel,
    service: Service,
    body: dict,
    repo: BaseProxyRepo,
    service_conn_pool: ServiceConnectionPool,
) -> Response:
    response = await _request_endpoint(model, service, "POST", body, repo, service_conn_pool)
    return _http_response(response)


async def _request_endpoint(
    model: EndpointModel,
    service: Service,
    method: str,
    body: dict,
    repo: BaseProxyRepo,
    service_conn_pool: ServiceConnectionPool,
) -> httpx.Response:
    http_client = await get_service_replica_client(service, repo, service_conn_pool)
    return await _request(http_client, method, _request_path(model), json=body)


async def _post_json(client: httpx.AsyncClient, path: str, body: dict) -> httpx.Response:
    return await _request(client, "POST", path, json=body)


async def _request(client: httpx.AsyncClient, method: str, path: str, **kwargs) -> httpx.Response:
    try:
        response = await client.request(method, path, **kwargs)
    except httpx.RequestError as e:
        raise ProxyError(f"Error requesting model: {e!r}", status.HTTP_502_BAD_GATEWAY)
    if response.status_code >= 400:
        raise ProxyError(
            response.content.decode(errors="replace"),
            response.status_code,
        )
    return response


def _http_response(response: httpx.Response) -> Response:
    headers = {}
    for name in ("content-disposition", "cache-control"):
        if name in response.headers:
            headers[name] = response.headers[name]
    return Response(
        content=response.content,
        status_code=response.status_code,
        media_type=response.headers.get("content-type"),
        headers=headers,
    )


def _video_create_response(model: EndpointModel, response: httpx.Response) -> Response:
    result = _response_json(response)
    if result.get("id"):
        result["id"] = _encode_video_id(model.name, result["id"])
    return Response(
        content=json.dumps(result),
        status_code=response.status_code,
        media_type="application/json",
    )


def _video_resource_response(video_id: str, response: httpx.Response) -> Response:
    result = _response_json(response)
    if result.get("id"):
        result["id"] = video_id
    return Response(
        content=json.dumps(result),
        status_code=response.status_code,
        media_type="application/json",
    )


async def _resolve_model_service(
    project_name: str,
    model_name: str,
    model: Optional[EndpointModel],
    repo: BaseProxyRepo,
) -> tuple[EndpointModel, Service]:
    if model is None:
        raise ProxyError(
            f"Model {model_name} not found in project {project_name}",
            status.HTTP_404_NOT_FOUND,
        )
    service = await repo.get_service(project_name, model.run_name)
    if service is None or not service.replicas:
        raise UnexpectedProxyError(
            f"Model {model.name} in project {project_name} references run {model.run_name}"
            " that does not exist or has no replicas"
        )
    return model, service


def _request_path(model: EndpointModel) -> str:
    if model.request_path:
        return model.request_path
    if model.format_spec is not None and model.format_spec.format == "openai":
        return f"{model.format_spec.prefix.rstrip('/')}/chat/completions"
    raise ProxyError(
        f"Model {model.name} does not declare a request path",
        status.HTTP_422_UNPROCESSABLE_ENTITY,
    )


def _require_api(model: EndpointModel, supported: set[str]) -> None:
    if model.api not in supported:
        raise ProxyError(
            f"Model {model.name} uses {model.api!r}, expected one of {sorted(supported)}",
            status.HTTP_422_UNPROCESSABLE_ENTITY,
        )


async def _json_body(request: Request) -> dict:
    try:
        body = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise ProxyError("Expected a JSON request body", status.HTTP_400_BAD_REQUEST)
    if not isinstance(body, dict):
        raise ProxyError("Expected a JSON object", status.HTTP_400_BAD_REQUEST)
    return body


def _required_model(body: dict) -> str:
    model = body.get("model")
    if not isinstance(model, str) or not model:
        raise ProxyError("Request must specify model", status.HTTP_422_UNPROCESSABLE_ENTITY)
    return model


def _last_user_prompt(body: ChatCompletionsRequest) -> str:
    for message in reversed(body.messages):
        if message.role != "user":
            continue
        if isinstance(message.content, str):
            return message.content
        if isinstance(message.content, list):
            texts = [
                part.get("text", "")
                for part in message.content
                if isinstance(part, dict) and part.get("type") == "text"
            ]
            prompt = "\n".join(text for text in texts if text)
            if prompt:
                return prompt
    raise ProxyError("Request has no user text prompt", status.HTTP_422_UNPROCESSABLE_ENTITY)


def _response_json(response: httpx.Response) -> dict:
    try:
        result = response.json()
    except json.JSONDecodeError:
        raise ProxyError("Endpoint returned invalid JSON", status.HTTP_502_BAD_GATEWAY)
    if not isinstance(result, dict):
        raise ProxyError("Endpoint returned a non-object response", status.HTTP_502_BAD_GATEWAY)
    return result


def _encode_video_id(model_name: str, upstream_id: str) -> str:
    data = json.dumps({"model": model_name, "id": upstream_id}, separators=(",", ":")).encode()
    return "dstack_" + base64.urlsafe_b64encode(data).decode().rstrip("=")


def _decode_video_id(video_id: str) -> tuple[str, str]:
    if not video_id.startswith("dstack_"):
        raise ProxyError("Invalid dstack video ID", status.HTTP_404_NOT_FOUND)
    encoded = video_id.removeprefix("dstack_")
    try:
        payload = json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))
        model_name = payload["model"]
        upstream_id = payload["id"]
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        raise ProxyError("Invalid dstack video ID", status.HTTP_404_NOT_FOUND)
    if not isinstance(model_name, str) or not isinstance(upstream_id, str):
        raise ProxyError("Invalid dstack video ID", status.HTTP_404_NOT_FOUND)
    return model_name, upstream_id


class StreamingAdaptor:
    """
    Converts a stream of ChatCompletionsChunk to an SSE stream.
    Also pre-fetches the first chunk **before** starting streaming to downstream,
    so that upstream request errors can propagate to the downstream client.
    """

    def __init__(self, stream: AsyncIterator[ChatCompletionsChunk]) -> None:
        self._stream = stream

    async def get_stream(self) -> AsyncIterator[bytes]:
        try:
            first_chunk = await self._stream.__anext__()
        except StopAsyncIteration:
            first_chunk = None
        return self._adaptor(first_chunk)

    async def _adaptor(self, first_chunk: Optional[ChatCompletionsChunk]) -> AsyncIterator[bytes]:
        if first_chunk is not None:
            yield self._encode_chunk(first_chunk)

            try:
                async for chunk in self._stream:
                    yield self._encode_chunk(chunk)
            except ProxyError as e:
                # No standard way to report errors while streaming,
                # but we'll at least send them as comments
                yield f": {e.detail!r}\n\n".encode()  # !r to avoid line breaks
                return

        yield "data: [DONE]\n\n".encode()

    @staticmethod
    def _encode_chunk(chunk: ChatCompletionsChunk) -> bytes:
        return f"data:{chunk.json()}\n\n".encode()
