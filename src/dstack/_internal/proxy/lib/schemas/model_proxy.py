from typing import Any, Dict, List, Literal, Optional, Union

from dstack._internal.core.models.common import CoreModel


class ChatMessage(CoreModel):
    role: str  # TODO(egor-s) types
    content: Any


class ChatCompletionsRequest(CoreModel):
    messages: List[ChatMessage]
    model: str
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Dict[str, float] = {}
    max_tokens: Optional[int] = None
    n: int = 1
    presence_penalty: float = 0.0
    response_format: Optional[Dict] = None
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    tools: List[Any] = []
    tool_choice: Union[Literal["none", "auto"], Dict] = {}
    user: Optional[str] = None
    # dstack extensions: optional diffusion controls for image-generation models.
    # Forwarded to the backend image endpoints only when explicitly set, so the
    # model keeps its own defaults otherwise. These let clients (e.g. OpenWebUI
    # advanced/custom parameters) tune generation and, in particular, how much a
    # reference image conditions image-to-image edits. `seed` (above) is
    # forwarded the same way. Text/chat models ignore these fields.
    strength: Optional[float] = None
    guidance_scale: Optional[float] = None
    num_inference_steps: Optional[int] = None
    negative_prompt: Optional[str] = None


class ChatCompletionsChoice(CoreModel):
    finish_reason: str
    index: int
    message: ChatMessage


class ChatCompletionsChunkChoice(CoreModel):
    delta: object
    logprobs: object = {}
    finish_reason: Optional[str]
    index: int


class ChatCompletionsUsage(CoreModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


class ChatCompletionsResponse(CoreModel):
    id: str
    choices: List[ChatCompletionsChoice]
    created: int
    model: str
    system_fingerprint: str = ""
    object: Literal["chat.completion"] = "chat.completion"
    usage: ChatCompletionsUsage


class ChatCompletionsChunk(CoreModel):
    id: Optional[str] = None
    choices: List[ChatCompletionsChunkChoice]
    created: Optional[int] = None
    model: str
    system_fingerprint: Optional[str] = ""
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"


class Model(CoreModel):
    object: Literal["model"] = "model"
    id: str
    created: int
    owned_by: str
    base: Optional[str] = None
    model: Optional[str] = None
    source: Optional[str] = None
    revision: Optional[str] = None
    modality: Optional[str] = None
    context_length: Optional[int] = None
    api: Optional[str] = None
    request_path: Optional[str] = None
    output_unit: Optional[str] = None


class ModelsResponse(CoreModel):
    object: Literal["list"] = "list"
    data: List[Model]
