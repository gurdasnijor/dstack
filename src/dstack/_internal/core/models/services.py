"""
Data structures related to `type: service` runs.
"""

import base64
import binascii
from typing import Optional, Union

from pydantic import Field, ValidationError
from typing_extensions import Annotated, Literal

from dstack._internal.core.models.common import CoreModel

ENDPOINT_METADATA_OPTION_KEY = "endpoint"
ENDPOINT_METADATA_TAG_FIELDS = {
    "_dstack_endpoint_base": "base",
    "_dstack_endpoint_model": "model",
    "_dstack_endpoint_api_model_name": "api_model_name",
    "_dstack_endpoint_source": "source",
    "_dstack_endpoint_revision": "revision",
    "_dstack_endpoint_modality": "modality",
    "_dstack_endpoint_context_length": "context_length",
    "_dstack_endpoint_api": "api",
    "_dstack_endpoint_request_path": "request_path",
    "_dstack_endpoint_output_unit": "output_unit",
}


class ServiceEndpointMetadata(CoreModel):
    """Validated endpoint capability metadata carried with an applied service."""

    base: str
    model: str
    api_model_name: str
    source: str
    revision: Optional[str] = None
    modality: str
    context_length: Optional[int] = None
    api: str
    request_path: Optional[str] = None
    output_unit: Optional[str] = None


def endpoint_metadata_to_tags(metadata: ServiceEndpointMetadata) -> dict[str, str]:
    data = metadata.dict(exclude_none=True)
    tags = {}
    for tag, field in ENDPOINT_METADATA_TAG_FIELDS.items():
        if field not in data:
            continue
        tags[tag] = _encode_endpoint_tag_value(str(data[field]))
    return tags


def endpoint_metadata_from_tags(
    tags: Optional[dict[str, str]],
) -> Optional[ServiceEndpointMetadata]:
    if not tags:
        return None
    data = {}
    try:
        for tag, field in ENDPOINT_METADATA_TAG_FIELDS.items():
            if tag in tags:
                data[field] = _decode_endpoint_tag_value(tags[tag])
        if "context_length" in data:
            data["context_length"] = int(data["context_length"])
        return ServiceEndpointMetadata.parse_obj(data)
    except (ValueError, UnicodeDecodeError, binascii.Error, ValidationError):
        return None


def _encode_endpoint_tag_value(value: str) -> str:
    encoded = base64.urlsafe_b64encode(value.encode()).decode().rstrip("=")
    if len(encoded) > 256:
        raise ValueError("Endpoint metadata value is too long to carry in run tags")
    return encoded


def _decode_endpoint_tag_value(value: str) -> str:
    padding = "=" * (-len(value) % 4)
    return base64.b64decode(value + padding, altchars=b"-_", validate=True).decode()


class BaseChatModel(CoreModel):
    type: Annotated[Literal["chat"], Field(description="The type of the model")] = "chat"
    name: Annotated[str, Field(description="The name of the model")]
    format: Annotated[
        str, Field(description="The serving format. Supported values include `openai` and `tgi`")
    ]


class TGIChatModel(BaseChatModel):
    """
    Mapping of the model for the OpenAI-compatible endpoint.

    Attributes:
        type (str): The type of the model, e.g. "chat"
        name (str): The name of the model. This name will be used both to load model configuration from the HuggingFace Hub and in the OpenAI-compatible endpoint.
        format (str): The format of the model, e.g. "tgi" if the model is served with HuggingFace's Text Generation Inference.
        chat_template (Optional[str]): The custom prompt template for the model. If not specified, the default prompt template from the HuggingFace Hub configuration will be used.
        eos_token (Optional[str]): The custom end of sentence token. If not specified, the default end of sentence token from the HuggingFace Hub configuration will be used.
    """

    format: Annotated[
        Literal["tgi"], Field(description="The serving format. Must be set to `tgi`")
    ]
    chat_template: Annotated[
        Optional[str],
        Field(
            description=(
                "The custom prompt template for the model."
                " If not specified, the default prompt template"
                " from the HuggingFace Hub configuration will be used"
            )
        ),
    ] = None  # will be set before registering the service
    eos_token: Annotated[
        Optional[str],
        Field(
            description=(
                "The custom end of sentence token."
                " If not specified, the default end of sentence token"
                " from the HuggingFace Hub configuration will be used"
            )
        ),
    ] = None


class OpenAIChatModel(BaseChatModel):
    """
    Mapping of the model for the OpenAI-compatible endpoint.

    Attributes:
        type (str): The type of the model, e.g. "chat"
        name (str): The name of the model. This name will be used both to load model configuration from the HuggingFace Hub and in the OpenAI-compatible endpoint.
        format (str): The format of the model, i.e. "openai".
        prefix (str): The `base_url` prefix: `http://hostname/{prefix}/chat/completions`. Defaults to `/v1`.
    """

    format: Annotated[
        Literal["openai"], Field(description="The serving format. Must be set to `openai`")
    ]
    prefix: Annotated[str, Field(description="The `base_url` prefix (after hostname)")] = "/v1"


ChatModel = Annotated[Union[TGIChatModel, OpenAIChatModel], Field(discriminator="format")]
AnyModel = Union[ChatModel]  # embeddings and etc.
