import httpx

from dstack._internal.proxy.lib.errors import UnexpectedProxyError
from dstack._internal.proxy.lib.models import ChatModel
from dstack._internal.proxy.lib.services.model_proxy.clients.base import ChatCompletionsClient
from dstack._internal.proxy.lib.services.model_proxy.clients.openai import OpenAIChatCompletions
from dstack._internal.proxy.lib.services.model_proxy.clients.tgi import TGIChatCompletions


def get_chat_client(model: ChatModel, http_client: httpx.AsyncClient) -> ChatCompletionsClient:
    format_spec = model.format_spec
    if format_spec is None:
        raise UnexpectedProxyError(f"Model {model.name} does not declare a chat format")
    if format_spec.format == "tgi":
        return TGIChatCompletions(
            http_client=http_client,
            chat_template=format_spec.chat_template,
            eos_token=format_spec.eos_token,
        )
    elif format_spec.format == "openai":
        return OpenAIChatCompletions(
            http_client=http_client,
            prefix=format_spec.prefix,
        )
    else:
        raise UnexpectedProxyError(f"Unsupported model format {format_spec.format}")
