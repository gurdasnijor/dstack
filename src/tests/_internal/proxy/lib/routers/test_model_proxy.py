import base64
import json
from datetime import datetime
from typing import AsyncIterator, Generator
from unittest.mock import AsyncMock, patch

import httpx
import openai
import pytest
from fastapi import FastAPI

from dstack._internal.proxy.gateway.repo.repo import GatewayProxyRepo
from dstack._internal.proxy.lib.auth import BaseProxyAuthProvider
from dstack._internal.proxy.lib.models import ChatModel, EndpointModel, OpenAIChatModelFormat
from dstack._internal.proxy.lib.repo import BaseProxyRepo
from dstack._internal.proxy.lib.routers.model_proxy import assets_router, router
from dstack._internal.proxy.lib.schemas.model_proxy import (
    ChatCompletionsChoice,
    ChatCompletionsChunk,
    ChatCompletionsChunkChoice,
    ChatCompletionsRequest,
    ChatCompletionsResponse,
    ChatCompletionsUsage,
    ChatMessage,
)
from dstack._internal.proxy.lib.services.model_proxy.clients.base import ChatCompletionsClient
from dstack._internal.proxy.lib.testing.auth import ProxyTestAuthProvider
from dstack._internal.proxy.lib.testing.common import (
    ProxyTestDependencyInjector,
    make_project,
    make_service,
)

SAMPLE_RESPONSE = "Hello there, how may I assist you today?"


class ChatClientStub(ChatCompletionsClient):
    async def generate(self, request: ChatCompletionsRequest) -> ChatCompletionsResponse:
        return ChatCompletionsResponse(
            id="chatcmpl-123",
            choices=[
                ChatCompletionsChoice(
                    finish_reason="stop",
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=SAMPLE_RESPONSE,
                    ),
                )
            ],
            created=int(datetime.now().timestamp()),
            model=request.model,
            usage=ChatCompletionsUsage(
                completion_tokens=12,
                prompt_tokens=9,
                total_tokens=21,
            ),
        )

    async def stream(self, request: ChatCompletionsRequest) -> AsyncIterator[ChatCompletionsChunk]:
        for i, word in enumerate(SAMPLE_RESPONSE.split(" ")):
            if i > 0:
                word = " " + word
            yield ChatCompletionsChunk(
                id="chatcmpl-123",
                choices=[
                    ChatCompletionsChunkChoice(
                        finish_reason=None,
                        index=0,
                        delta=dict(
                            role="assistant",
                            content=word,
                        ),
                    )
                ],
                created=int(datetime.now().timestamp()),
                model=request.model,
            )


def make_model(
    project_name: str, name: str, run_name: str, created_at: datetime = datetime.fromtimestamp(0)
) -> ChatModel:
    return ChatModel(
        project_name=project_name,
        name=name,
        created_at=created_at,
        run_name=run_name,
        format_spec=OpenAIChatModelFormat(format="openai", prefix="/v1"),
    )


def make_endpoint_model(
    project_name: str,
    name: str,
    run_name: str,
    *,
    modality: str,
    api: str,
    request_path: str,
    base: str = "test/base-model",
    model: str = "test/runtime-model",
    source: str = "huggingface",
    revision: str = "abc123",
) -> EndpointModel:
    return EndpointModel(
        project_name=project_name,
        name=name,
        created_at=datetime.fromtimestamp(0),
        run_name=run_name,
        base=base,
        model=model,
        source=source,
        revision=revision,
        modality=modality,
        api=api,
        request_path=request_path,
    )


def make_http_client(repo: BaseProxyRepo, auth: BaseProxyAuthProvider) -> httpx.AsyncClient:
    app = FastAPI()
    app.state.proxy_dependency_injector = ProxyTestDependencyInjector(repo=repo, auth=auth)
    app.include_router(assets_router, prefix="/proxy/models")
    app.include_router(router, prefix="/proxy/models")
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app))


def make_openai_client(
    repo: BaseProxyRepo,
    auth: BaseProxyAuthProvider,
    project_name: str,
    auth_token: str = "token",
) -> openai.AsyncOpenAI:
    http_client = make_http_client(repo, auth)
    return openai.AsyncOpenAI(
        api_key=auth_token,
        base_url=f"http://test-host/proxy/models/{project_name}",
        http_client=http_client,
    )


@pytest.fixture
def mock_chat_client() -> Generator[None, None, None]:
    with (
        patch(
            "dstack._internal.proxy.lib.services.service_connection.ServiceConnectionPool.get_or_add"
        ),
        patch("dstack._internal.proxy.lib.routers.model_proxy.get_chat_client") as get_client_mock,
    ):
        get_client_mock.return_value = ChatClientStub()
        yield


@pytest.mark.asyncio
async def test_list_models() -> None:
    auth = ProxyTestAuthProvider({"test-proj": {"token"}})
    repo = GatewayProxyRepo()
    await repo.set_project(make_project("test-proj"))
    await repo.set_service(make_service("test-proj", "test-service-1"))
    await repo.set_service(make_service("test-proj", "test-service-2"))
    await repo.set_model(
        make_model(
            "test-proj", "test-model-1", "test-service-1", created_at=datetime.fromtimestamp(123)
        ),
    )
    await repo.set_model(
        make_model(
            "test-proj", "test-model-2", "test-service-2", created_at=datetime.fromtimestamp(321)
        ),
    )

    client = make_openai_client(repo, auth, "test-proj", auth_token="token")
    models = [model async for model in client.models.list()]

    assert models[0].id == "test-model-1"
    assert models[0].created == 123
    assert models[0].owned_by == "test-proj"
    assert models[1].id == "test-model-2"
    assert models[1].created == 321
    assert models[1].owned_by == "test-proj"


@pytest.mark.asyncio
async def test_list_models_includes_endpoint_capabilities() -> None:
    auth = ProxyTestAuthProvider({"test-proj": {"token"}})
    repo = GatewayProxyRepo()
    await repo.set_project(make_project("test-proj"))
    await repo.set_service(make_service("test-proj", "image-service"))
    await repo.set_model(
        make_endpoint_model(
            "test-proj",
            "test-image-model",
            "image-service",
            modality="image-generation",
            api="images_generations",
            request_path="/v1/images/generations",
        )
    )

    client = make_http_client(repo, auth)
    response = await client.get(
        "http://test-host/proxy/models/test-proj/models",
        headers={"Authorization": "Bearer token"},
    )

    assert response.status_code == 200
    assert response.json()["data"] == [
        {
            "id": "test-image-model",
            "object": "model",
            "created": 0,
            "owned_by": "test-proj",
            "base": "test/base-model",
            "model": "test/runtime-model",
            "source": "huggingface",
            "revision": "abc123",
            "modality": "image-generation",
            "context_length": None,
            "api": "images_generations",
            "request_path": "/v1/images/generations",
            "output_unit": None,
        }
    ]


@pytest.mark.asyncio
async def test_list_models_empty() -> None:
    auth = ProxyTestAuthProvider({"test-proj": {"token"}, "test-proj-empty": {"token"}})
    repo = GatewayProxyRepo()
    await repo.set_project(make_project("test-proj"))
    await repo.set_project(make_project("test-proj-empty"))
    await repo.set_service(make_service("test-proj", "test-service"))
    await repo.set_model(make_model("test-proj", "test-model", "test-service"))

    client = make_openai_client(repo, auth, "test-proj-empty", auth_token="token")
    models = [model async for model in client.models.list()]
    assert not models


@pytest.mark.asyncio
async def test_chat_completions(mock_chat_client) -> None:
    auth = ProxyTestAuthProvider({"test-proj": {"token"}})
    repo = GatewayProxyRepo()
    await repo.set_project(make_project("test-proj"))
    await repo.set_service(make_service("test-proj", "test-service"))
    await repo.set_model(make_model("test-proj", "test-model", "test-service"))
    client = make_openai_client(repo, auth, "test-proj", auth_token="token")
    completion = await client.chat.completions.create(
        model="test-model",
        messages=[{"role": "user", "content": "Hi"}],
    )
    assert completion.choices[0].message.content == SAMPLE_RESPONSE


@pytest.mark.asyncio
async def test_chat_completions_stream(mock_chat_client) -> None:
    auth = ProxyTestAuthProvider({"test-proj": {"token"}})
    repo = GatewayProxyRepo()
    await repo.set_project(make_project("test-proj"))
    await repo.set_service(make_service("test-proj", "test-service"))
    await repo.set_model(make_model("test-proj", "test-model", "test-service"))
    client = make_openai_client(repo, auth, "test-proj", auth_token="token")
    response = await client.chat.completions.create(
        model="test-model",
        messages=[{"role": "user", "content": "Hi"}],
        stream=True,
    )
    completion = ""
    async for chunk in response:
        completion += chunk.choices[0].delta.content
    assert completion == SAMPLE_RESPONSE


@pytest.mark.asyncio
async def test_image_generation_native_and_chat_projection(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DSTACK_PROXY_ASSETS_DIR", str(tmp_path))
    auth = ProxyTestAuthProvider({"test-proj": {"token"}})
    repo = GatewayProxyRepo()
    await repo.set_project(make_project("test-proj"))
    await repo.set_service(make_service("test-proj", "image-service"))
    await repo.set_model(
        make_endpoint_model(
            "test-proj",
            "test-image-model",
            "image-service",
            modality="image-generation",
            api="images_generations",
            request_path="/v1/images/generations",
        )
    )
    upstream_requests = []

    image_content = b"\x89PNG\r\n\x1a\nimage"
    encoded_image = base64.b64encode(image_content).decode()

    async def upstream(request: httpx.Request) -> httpx.Response:
        upstream_requests.append(request)
        return httpx.Response(
            200,
            json={"created": 1, "data": [{"b64_json": encoded_image}]},
        )

    upstream_client = httpx.AsyncClient(
        transport=httpx.MockTransport(upstream),
        base_url="http://image-service",
    )
    client = make_http_client(repo, auth)
    with patch(
        "dstack._internal.proxy.lib.routers.model_proxy.get_service_replica_client",
        new=AsyncMock(return_value=upstream_client),
    ):
        native_response = await client.post(
            "http://test-host/proxy/models/test-proj/images/generations",
            headers={"Authorization": "Bearer token"},
            json={
                "model": "test-image-model",
                "prompt": "A copper teapot",
                "size": "1024x1024",
            },
        )
        chat_response = await client.post(
            "http://test-host/proxy/models/test-proj/chat/completions",
            headers={"Authorization": "Bearer token"},
            json={
                "model": "test-image-model",
                "messages": [{"role": "user", "content": "A glass forest"}],
            },
        )
    await upstream_client.aclose()

    assert native_response.status_code == 200
    assert native_response.json()["data"][0]["b64_json"] == encoded_image
    assert upstream_requests[0].url.path == "/v1/images/generations"
    assert upstream_requests[0].content == (
        b'{"model":"test-image-model","prompt":"A copper teapot","size":"1024x1024"}'
    )
    assert upstream_requests[1].url.path == "/v1/images/generations"
    assert chat_response.status_code == 200
    content = chat_response.json()["choices"][0]["message"]["content"]
    assert content.startswith(
        "![Generated image](http://test-host/proxy/models/test-proj/assets/dstack_asset_"
    )
    assert content.endswith(")")
    asset_response = await client.get(
        content.removeprefix("![Generated image](").removesuffix(")")
    )
    assert asset_response.status_code == 200
    assert asset_response.content == image_content
    assert asset_response.headers["content-type"] == "image/png"
    assert asset_response.headers["access-control-allow-origin"] == "*"
    assert asset_response.headers["access-control-allow-private-network"] == "true"
    assert asset_response.headers["access-control-expose-headers"] == (
        "Content-Length, Content-Type"
    )
    assert asset_response.headers["x-content-type-options"] == "nosniff"
    preflight_response = await client.options(
        content.removeprefix("![Generated image](").removesuffix(")"),
        headers={
            "Origin": "https://ai.example",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Private-Network": "true",
        },
    )
    assert preflight_response.status_code == 204
    assert preflight_response.headers["access-control-allow-methods"] == "GET, OPTIONS"
    assert preflight_response.headers["access-control-allow-private-network"] == "true"


@pytest.mark.asyncio
async def test_image_chat_projection_uses_reference_image_edit(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DSTACK_PROXY_ASSETS_DIR", str(tmp_path))
    auth = ProxyTestAuthProvider({"test-proj": {"token"}})
    repo = GatewayProxyRepo()
    await repo.set_project(make_project("test-proj"))
    await repo.set_service(make_service("test-proj", "image-service"))
    await repo.set_model(
        make_endpoint_model(
            "test-proj",
            "test-image-model",
            "image-service",
            modality="image-generation",
            api="images_generations",
            request_path="/v1/images/generations",
        )
    )
    reference_content = b"\x89PNG\r\n\x1a\nreference"
    generated_content = b"\x89PNG\r\n\x1a\ngenerated"
    upstream_requests = []

    async def upstream(request: httpx.Request) -> httpx.Response:
        upstream_requests.append(request)
        return httpx.Response(
            200,
            json={
                "created": 1,
                "data": [{"b64_json": base64.b64encode(generated_content).decode()}],
            },
        )

    upstream_client = httpx.AsyncClient(
        transport=httpx.MockTransport(upstream),
        base_url="http://image-service",
    )
    client = make_http_client(repo, auth)
    with patch(
        "dstack._internal.proxy.lib.routers.model_proxy.get_service_replica_client",
        new=AsyncMock(return_value=upstream_client),
    ):
        response = await client.post(
            "http://test-host/proxy/models/test-proj/chat/completions",
            headers={"Authorization": "Bearer token"},
            json={
                "model": "test-image-model",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Paint this as a watercolor"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": (
                                        "data:image/png;base64,"
                                        + base64.b64encode(reference_content).decode()
                                    )
                                },
                            },
                        ],
                    }
                ],
            },
        )
    await upstream_client.aclose()

    assert response.status_code == 200
    assert len(upstream_requests) == 1
    upstream_request = upstream_requests[0]
    assert upstream_request.url.path == "/v1/images/edits"
    assert upstream_request.headers["content-type"].startswith("multipart/form-data; boundary=")
    assert b'name="model"' in upstream_request.content
    assert b"test-image-model" in upstream_request.content
    assert b'name="prompt"' in upstream_request.content
    assert b"Paint this as a watercolor" in upstream_request.content
    assert b'name="size"' in upstream_request.content
    assert b"auto" in upstream_request.content
    assert b'name="image"; filename="reference.png"' in upstream_request.content
    assert b"Content-Type: image/png" in upstream_request.content
    assert reference_content in upstream_request.content

    content = response.json()["choices"][0]["message"]["content"]
    asset_response = await client.get(
        content.removeprefix("![Generated image](").removesuffix(")")
    )
    assert asset_response.status_code == 200
    assert asset_response.content == generated_content


@pytest.mark.asyncio
async def test_image_chat_projection_rejects_non_data_reference_url() -> None:
    auth = ProxyTestAuthProvider({"test-proj": {"token"}})
    repo = GatewayProxyRepo()
    await repo.set_project(make_project("test-proj"))
    await repo.set_service(make_service("test-proj", "image-service"))
    await repo.set_model(
        make_endpoint_model(
            "test-proj",
            "test-image-model",
            "image-service",
            modality="image-generation",
            api="images_generations",
            request_path="/v1/images/generations",
        )
    )
    client = make_http_client(repo, auth)
    upstream_client = httpx.AsyncClient(base_url="http://image-service")

    with patch(
        "dstack._internal.proxy.lib.routers.model_proxy.get_service_replica_client",
        new=AsyncMock(return_value=upstream_client),
    ):
        response = await client.post(
            "http://test-host/proxy/models/test-proj/chat/completions",
            headers={"Authorization": "Bearer token"},
            json={
                "model": "test-image-model",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Paint this as a watercolor"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://private.example/reference.png"},
                            },
                        ],
                    }
                ],
            },
        )
    await upstream_client.aclose()

    assert response.status_code == 422
    assert response.json()["detail"] == "Reference images must be base64 data image URLs"


@pytest.mark.asyncio
async def test_image_chat_projection_splits_large_stream_events(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DSTACK_PROXY_ASSETS_DIR", str(tmp_path))
    auth = ProxyTestAuthProvider({"test-proj": {"token"}})
    repo = GatewayProxyRepo()
    await repo.set_project(make_project("test-proj"))
    await repo.set_service(make_service("test-proj", "image-service"))
    await repo.set_model(
        make_endpoint_model(
            "test-proj",
            "test-image-model",
            "image-service",
            modality="image-generation",
            api="images_generations",
            request_path="/v1/images/generations",
        )
    )
    image_content = b"\x89PNG\r\n\x1a\n" + b"a" * (256 * 1024)
    encoded_image = base64.b64encode(image_content).decode()

    async def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"created": 1, "data": [{"b64_json": encoded_image}]},
        )

    upstream_client = httpx.AsyncClient(
        transport=httpx.MockTransport(upstream),
        base_url="http://image-service",
    )
    client = make_http_client(repo, auth)
    with patch(
        "dstack._internal.proxy.lib.routers.model_proxy.get_service_replica_client",
        new=AsyncMock(return_value=upstream_client),
    ):
        response = await client.post(
            "http://test-host/proxy/models/test-proj/chat/completions",
            headers={"Authorization": "Bearer token"},
            json={
                "model": "test-image-model",
                "messages": [{"role": "user", "content": "A glass forest"}],
                "stream": True,
            },
        )
    await upstream_client.aclose()

    events = [
        line.removeprefix("data:")
        for line in response.text.splitlines()
        if line.startswith("data:") and line != "data: [DONE]"
    ]
    assert max(map(len, events)) < 128 * 1024
    content = "".join(
        json.loads(event)["choices"][0]["delta"].get("content", "") for event in events
    )
    assert content.startswith(
        "![Generated image](http://test-host/proxy/models/test-proj/assets/dstack_asset_"
    )
    asset_response = await client.get(
        content.removeprefix("![Generated image](").removesuffix(")")
    )
    assert asset_response.content == image_content


@pytest.mark.asyncio
async def test_image_chat_projection_uses_configured_public_base_url(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("DSTACK_PROXY_ASSETS_DIR", str(tmp_path))
    auth = ProxyTestAuthProvider({"test-proj": {"token"}})
    repo = GatewayProxyRepo()
    await repo.set_project(make_project("test-proj"))
    await repo.set_service(make_service("test-proj", "image-service"))
    await repo.set_model(
        make_endpoint_model(
            "test-proj",
            "test-image-model",
            "image-service",
            modality="image-generation",
            api="images_generations",
            request_path="/v1/images/generations",
        )
    )

    async def upstream(request: httpx.Request) -> httpx.Response:
        image_content = b"\x89PNG\r\n\x1a\nimage"
        return httpx.Response(
            200,
            json={
                "created": 1,
                "data": [{"b64_json": base64.b64encode(image_content).decode()}],
            },
        )

    upstream_client = httpx.AsyncClient(
        transport=httpx.MockTransport(upstream),
        base_url="http://image-service",
    )
    client = make_http_client(repo, auth)
    with patch(
        "dstack._internal.proxy.lib.routers.model_proxy.get_service_replica_client",
        new=AsyncMock(return_value=upstream_client),
    ):
        response = await client.post(
            "http://test-host/proxy/models/test-proj/chat/completions",
            headers={
                "Authorization": "Bearer token",
                "X-Dstack-Public-Base-URL": "https://dstack.example/proxy/models/test-proj",
            },
            json={
                "model": "test-image-model",
                "messages": [{"role": "user", "content": "A glass forest"}],
            },
        )
    await upstream_client.aclose()

    content = response.json()["choices"][0]["message"]["content"]
    assert content.startswith(
        "![Generated image](https://dstack.example/proxy/models/test-proj/assets/dstack_asset_"
    )


@pytest.mark.asyncio
async def test_background_task_for_media_model_uses_ready_text_model() -> None:
    auth = ProxyTestAuthProvider({"test-proj": {"token"}})
    repo = GatewayProxyRepo()
    await repo.set_project(make_project("test-proj"))
    await repo.set_service(make_service("test-proj", "image-service"))
    await repo.set_service(make_service("test-proj", "text-service"))
    await repo.set_model(
        make_endpoint_model(
            "test-proj",
            "test-image-model",
            "image-service",
            modality="image-generation",
            api="images_generations",
            request_path="/v1/images/generations",
        )
    )
    await repo.set_model(
        make_endpoint_model(
            "test-proj",
            "test-text-model",
            "text-service",
            modality="text-generation",
            api="chat_completions",
            request_path="/v1/chat/completions",
        )
    )
    upstream_requests = []

    async def upstream(request: httpx.Request) -> httpx.Response:
        upstream_requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-task",
                "object": "chat.completion",
                "created": 1,
                "model": "test-text-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": '{"title":"Glass Forest"}',
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            },
        )

    upstream_client = httpx.AsyncClient(
        transport=httpx.MockTransport(upstream),
        base_url="http://text-service",
    )
    client = make_http_client(repo, auth)
    with patch(
        "dstack._internal.proxy.lib.routers.model_proxy.get_service_replica_client",
        new=AsyncMock(return_value=upstream_client),
    ):
        response = await client.post(
            "http://test-host/proxy/models/test-proj/chat/completions",
            headers={
                "Authorization": "Bearer token",
                "X-OpenWebUI-Task": "title_generation",
            },
            json={
                "model": "test-image-model",
                "messages": [{"role": "user", "content": "Generate a title"}],
            },
        )
    await upstream_client.aclose()

    assert response.status_code == 200
    assert response.json()["model"] == "test-text-model"
    assert upstream_requests[0].url.path == "/v1/chat/completions"
    assert json.loads(upstream_requests[0].content)["model"] == "test-text-model"


@pytest.mark.asyncio
async def test_endpoint_chat_stream_is_forwarded_without_buffering() -> None:
    auth = ProxyTestAuthProvider({"test-proj": {"token"}})
    repo = GatewayProxyRepo()
    await repo.set_project(make_project("test-proj"))
    await repo.set_service(make_service("test-proj", "chat-service"))
    await repo.set_model(
        make_endpoint_model(
            "test-proj",
            "test-chat-model",
            "chat-service",
            modality="text-generation",
            api="chat_completions",
            request_path="/v1/chat/completions",
        )
    )

    class UpstreamStream(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield (
                b'data: {"id":"1","object":"chat.completion.chunk","created":1,'
                b'"model":"test-chat-model","choices":[{"index":0,"delta":'
                b'{"content":"hello"},"finish_reason":null}]}\n\n'
            )
            yield b"data: [DONE]\n\n"

    async def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=UpstreamStream(),
        )

    upstream_client = httpx.AsyncClient(
        transport=httpx.MockTransport(upstream),
        base_url="http://chat-service",
    )
    client = make_openai_client(repo, auth, "test-proj")
    with patch(
        "dstack._internal.proxy.lib.routers.model_proxy.get_service_replica_client",
        new=AsyncMock(return_value=upstream_client),
    ):
        stream = await client.chat.completions.create(
            model="test-chat-model",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )
        content = ""
        async for chunk in stream:
            content += chunk.choices[0].delta.content or ""
    await upstream_client.aclose()

    assert content == "hello"


@pytest.mark.asyncio
async def test_video_generation_wraps_upstream_id_for_status_and_content() -> None:
    auth = ProxyTestAuthProvider({"test-proj": {"token"}})
    repo = GatewayProxyRepo()
    await repo.set_project(make_project("test-proj"))
    await repo.set_service(make_service("test-proj", "video-service"))
    await repo.set_model(
        make_endpoint_model(
            "test-proj",
            "test-video-model",
            "video-service",
            modality="video-generation",
            api="videos",
            request_path="/v1/videos",
        )
    )
    upstream_paths = []

    async def upstream(request: httpx.Request) -> httpx.Response:
        upstream_paths.append(request.url.path)
        if request.method == "POST":
            return httpx.Response(200, json={"id": "upstream-job", "status": "queued"})
        if request.url.path.endswith("/content"):
            return httpx.Response(200, content=b"video", headers={"content-type": "video/mp4"})
        return httpx.Response(200, json={"id": "upstream-job", "status": "completed"})

    upstream_client = httpx.AsyncClient(
        transport=httpx.MockTransport(upstream),
        base_url="http://video-service",
    )
    client = make_http_client(repo, auth)
    with patch(
        "dstack._internal.proxy.lib.routers.model_proxy.get_service_replica_client",
        new=AsyncMock(return_value=upstream_client),
    ):
        create_response = await client.post(
            "http://test-host/proxy/models/test-proj/videos",
            headers={"Authorization": "Bearer token"},
            json={"model": "test-video-model", "prompt": "Clouds over a canyon"},
        )
        video_id = create_response.json()["id"]
        status_response = await client.get(
            f"http://test-host/proxy/models/test-proj/videos/{video_id}",
            headers={"Authorization": "Bearer token"},
        )
        content_response = await client.get(
            f"http://test-host/proxy/models/test-proj/videos/{video_id}/content",
            headers={"Authorization": "Bearer token"},
        )
    await upstream_client.aclose()

    assert video_id.startswith("dstack_")
    assert status_response.json() == {"id": video_id, "status": "completed"}
    assert content_response.content == b"video"
    assert content_response.headers["content-type"] == "video/mp4"
    assert upstream_paths == [
        "/v1/videos",
        "/v1/videos/upstream-job",
        "/v1/videos/upstream-job/content",
    ]


@pytest.mark.asyncio
async def test_synchronous_video_is_exposed_as_completed_video_resource(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("DSTACK_PROXY_ASSETS_DIR", str(tmp_path))
    auth = ProxyTestAuthProvider({"test-proj": {"token"}})
    repo = GatewayProxyRepo()
    await repo.set_project(make_project("test-proj"))
    await repo.set_service(make_service("test-proj", "video-service"))
    await repo.set_model(
        make_endpoint_model(
            "test-proj",
            "test-video-model",
            "video-service",
            modality="video-generation",
            api="video_generations",
            request_path="/v1/videos/sync",
        )
    )

    async def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"generated-video",
            headers={"content-type": "video/mp4"},
        )

    upstream_client = httpx.AsyncClient(
        transport=httpx.MockTransport(upstream),
        base_url="http://video-service",
    )
    client = make_http_client(repo, auth)
    headers = {"Authorization": "Bearer token"}
    with patch(
        "dstack._internal.proxy.lib.routers.model_proxy.get_service_replica_client",
        new=AsyncMock(return_value=upstream_client),
    ):
        create_response = await client.post(
            "http://test-host/proxy/models/test-proj/videos",
            headers=headers,
            json={
                "model": "test-video-model",
                "prompt": "A glass forest",
                "num_frames": 25,
            },
        )
        video = create_response.json()
        status_response = await client.get(
            f"http://test-host/proxy/models/test-proj/videos/{video['id']}",
            headers=headers,
        )
        content_response = await client.get(
            f"http://test-host/proxy/models/test-proj/videos/{video['id']}/content",
            headers=headers,
        )
    await upstream_client.aclose()

    assert create_response.status_code == 200
    assert video["status"] == "completed"
    assert video["model"] == "test-video-model"
    assert video["id"].startswith("dstack_cached_")
    assert status_response.json() == video
    assert content_response.content == b"generated-video"
    assert content_response.headers["content-type"] == "video/mp4"


@pytest.mark.asyncio
async def test_synchronous_video_has_chat_projection(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DSTACK_PROXY_ASSETS_DIR", str(tmp_path))
    auth = ProxyTestAuthProvider({"test-proj": {"token"}})
    repo = GatewayProxyRepo()
    await repo.set_project(make_project("test-proj"))
    await repo.set_service(make_service("test-proj", "video-service"))
    await repo.set_model(
        make_endpoint_model(
            "test-proj",
            "test-video-model",
            "video-service",
            modality="video-generation",
            api="video_generations",
            request_path="/v1/videos/sync",
        )
    )

    async def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"generated-video",
            headers={"content-type": "video/mp4"},
        )

    upstream_client = httpx.AsyncClient(
        transport=httpx.MockTransport(upstream),
        base_url="http://video-service",
    )
    client = make_http_client(repo, auth)
    headers = {"Authorization": "Bearer token"}
    with patch(
        "dstack._internal.proxy.lib.routers.model_proxy.get_service_replica_client",
        new=AsyncMock(return_value=upstream_client),
    ):
        chat_response = await client.post(
            "http://test-host/proxy/models/test-proj/chat/completions",
            headers={
                **headers,
                "X-Dstack-Public-Base-URL": "https://dstack.example/proxy/models/test-proj",
            },
            json={
                "model": "test-video-model",
                "messages": [{"role": "user", "content": "A glass forest"}],
            },
        )
        content = chat_response.json()["choices"][0]["message"]["content"]
    await upstream_client.aclose()

    assert chat_response.status_code == 200
    assert content.startswith(
        "[Generated video](https://dstack.example/proxy/models/test-proj/assets/dstack_asset_"
    )
    asset_id = content.removesuffix(")").rsplit("/", 1)[1]
    asset_response = await client.get(f"http://test-host/proxy/models/test-proj/assets/{asset_id}")
    assert asset_response.status_code == 200
    assert asset_response.content == b"generated-video"
    assert asset_response.headers["content-type"] == "video/mp4"
    wrong_project_response = await client.get(
        f"http://test-host/proxy/models/other-project/assets/{asset_id}"
    )
    assert wrong_project_response.status_code == 404


@pytest.mark.asyncio
async def test_chat_completions_model_not_found() -> None:
    auth = ProxyTestAuthProvider({"test-proj": {"token"}})
    repo = GatewayProxyRepo()
    await repo.set_project(make_project("test-proj"))
    client = make_openai_client(repo, auth, "test-proj", auth_token="token")
    with pytest.raises(openai.NotFoundError):
        await client.chat.completions.create(
            model="unknown-model",
            messages=[{"role": "user", "content": "Hi"}],
        )


@pytest.mark.asyncio
async def test_unauthorized_openai_sdk() -> None:
    auth = ProxyTestAuthProvider({"test-proj": {"correct-token"}})
    repo = GatewayProxyRepo()
    await repo.set_project(make_project("test-proj"))
    client = make_openai_client(repo, auth, "test-proj", auth_token="invalid-token")

    with pytest.raises(openai.PermissionDeniedError):
        await client.models.list()
    with pytest.raises(openai.PermissionDeniedError):
        await client.chat.completions.create(
            model="test-model",
            messages=[{"role": "user", "content": "Hi"}],
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers",
    [
        {"Authorization": "Bearer invalid-token"},
        {"Authorization": "Bearer "},
        {"Authorization": "Bearer"},
        {"Authorization": ""},
        None,
    ],
)
async def test_unauthorized_http(headers) -> None:
    auth = ProxyTestAuthProvider({"test-proj": {"correct-token"}})
    repo = GatewayProxyRepo()
    await repo.set_project(make_project("test-proj"))
    client = make_http_client(repo, auth)

    resp = await client.get("http://test-host/proxy/models/test-proj/models", headers=headers)
    assert resp.status_code == 403

    resp = await client.post(
        "http://test-host/proxy/models/test-proj/chat/completions",
        json={"model": "test-model", "messages": [{"role": "user", "content": "Hi"}]},
        headers=headers,
    )
    assert resp.status_code == 403
