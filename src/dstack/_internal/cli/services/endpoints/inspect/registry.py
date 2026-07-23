"""Versioned support registries for the runtimes dstack can actually deploy.

Each registry is an explicit, versioned snapshot of a runtime's documented
support surface. Registry membership is evidence that a model architecture or
pipeline class is supported by the pinned runtime version; absence from a
registry is *not* a negative result — it only means the fact is unverified and
must be established by research or a smoke test.

These tables are hand-captured from the documentation sources recorded on each
entry. When a runtime image pin changes, re-capture the table and bump the
registry version; never edit entries without updating ``captured_at``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from dstack._internal.cli.services.endpoints.inspect.runtime import (
    CURRENT_VLLM_IMAGE_DIGEST,
    NATIVE_OMNI_PIPELINES,
)


@dataclass(frozen=True)
class RuntimeSupportRegistry:
    """A versioned snapshot of one runtime's documented support surface."""

    runtime: str
    version: str
    source: str
    captured_at: str
    architectures: frozenset[str] = field(default_factory=frozenset)
    pipelines: frozenset[str] = field(default_factory=frozenset)
    gguf: bool = False
    notes: tuple[str, ...] = ()

    def supports_architecture(self, architecture: str) -> bool:
        return architecture in self.architectures


@dataclass(frozen=True)
class ExactRecipe:
    """A versioned recipe validated end to end for a specific model."""

    model: str
    runtime: str
    evidence: str
    verified_at: str
    launch_hints: tuple[str, ...] = ()
    revision: Optional[str] = None
    notes: tuple[str, ...] = ()


VLLM_REGISTRY = RuntimeSupportRegistry(
    runtime="vllm",
    version=CURRENT_VLLM_IMAGE_DIGEST,
    source="https://docs.vllm.ai/en/stable/models/supported_models/",
    captured_at="2026-07-22",
    architectures=frozenset(
        {
            "BaichuanForCausalLM",
            "ChatGLMModel",
            "DeepseekV2ForCausalLM",
            "DeepseekV3ForCausalLM",
            "FalconForCausalLM",
            "Gemma2ForCausalLM",
            "Gemma3ForCausalLM",
            "GlmForCausalLM",
            "Glm4ForCausalLM",
            "GPTNeoXForCausalLM",
            "GptOssForCausalLM",
            "InternLM2ForCausalLM",
            "InternVLChatModel",
            "LlamaForCausalLM",
            "LlavaForConditionalGeneration",
            "MiniCPMForCausalLM",
            "MistralForCausalLM",
            "MixtralForCausalLM",
            "MllamaForConditionalGeneration",
            "Phi3ForCausalLM",
            "Qwen2ForCausalLM",
            "Qwen2MoeForCausalLM",
            "Qwen2VLForConditionalGeneration",
            "Qwen2_5_VLForConditionalGeneration",
            "Qwen3ForCausalLM",
            "Qwen3MoeForCausalLM",
            "StableLmForCausalLM",
        }
    ),
    gguf=False,
    notes=(
        "GGUF loading in vLLM is experimental and is not pinned as supported "
        "at this image digest; treat GGUF-on-vLLM as a discovery hint only.",
    ),
)

SGLANG_REGISTRY = RuntimeSupportRegistry(
    runtime="sglang",
    version="sglang-0.4 docs snapshot",
    source="https://github.com/sgl-project/sglang/tree/main/docs/supported_models",
    captured_at="2026-07-22",
    architectures=frozenset(
        {
            "ChatGLMModel",
            "DeepseekV2ForCausalLM",
            "DeepseekV3ForCausalLM",
            "Gemma2ForCausalLM",
            "GlmForCausalLM",
            "InternLM2ForCausalLM",
            "LlamaForCausalLM",
            "LlavaForConditionalGeneration",
            "MistralForCausalLM",
            "MixtralForCausalLM",
            "Qwen2ForCausalLM",
            "Qwen2MoeForCausalLM",
            "Qwen2VLForConditionalGeneration",
            "Qwen2_5_VLForConditionalGeneration",
            "Qwen3ForCausalLM",
            "Qwen3MoeForCausalLM",
        }
    ),
    gguf=False,
    notes=(
        "dstack has no pinned SGLang image digest yet; a matched architecture "
        "still requires the agent to pin an image before submission.",
    ),
)

LLAMACPP_REGISTRY = RuntimeSupportRegistry(
    runtime="llama.cpp",
    version="llama-server OpenAI-compatible API",
    source="https://github.com/ggml-org/llama.cpp/tree/master/tools/server",
    captured_at="2026-07-22",
    gguf=True,
    notes=(
        "llama.cpp serves GGUF checkpoints natively through llama-server.",
        "dstack has no pinned llama.cpp image digest yet; the agent must "
        "select and digest-pin a server image.",
    ),
)

VLLM_OMNI_REGISTRY = RuntimeSupportRegistry(
    runtime="vllm-omni",
    version="vllm-omni native pipeline snapshot",
    source="https://docs.vllm.ai/projects/vllm-omni/en/latest/models/supported_models/",
    captured_at="2026-07-22",
    pipelines=frozenset(NATIVE_OMNI_PIPELINES),
    notes=(
        "Pipelines outside the native set may still run through the documented "
        "generic Diffusers adapter (`--omni --diffusion-load-format diffusers`).",
    ),
)

DIFFUSERS_ADAPTER_REGISTRY = RuntimeSupportRegistry(
    runtime="diffusers",
    version="DiffusionPipeline.from_pretrained generic contract",
    source="https://huggingface.co/docs/diffusers/api/pipelines/overview",
    captured_at="2026-07-22",
    notes=(
        "Applies only to repositories with a standard model_index.json; the "
        "contract is a documented adapter, so a real smoke test is mandatory.",
    ),
)

# Recipes validated end to end on this platform. Keyed by lowercase repo id.
EXACT_RECIPES: dict[str, ExactRecipe] = {
    "qwen/qwen2.5-7b-instruct": ExactRecipe(
        model="Qwen/Qwen2.5-7B-Instruct",
        runtime="vllm",
        evidence=(
            "Validated endpoint preset: chat completions and tool_choice:auto "
            "verified through the dstack proxy (service qwen25-7b, 2026-07)."
        ),
        verified_at="2026-07",
        launch_hints=("--enable-auto-tool-choice", "--tool-call-parser hermes"),
    ),
    "eniora/juggernaut_xl_ragnarok": ExactRecipe(
        model="eniora/Juggernaut_XL_Ragnarok",
        runtime="diffusers",
        evidence=(
            "Validated endpoint preset: 1024x1024 PNG rendered through the "
            "dstack proxy (service juggernaut-xl-ragnarok, 2026-07)."
        ),
        verified_at="2026-07",
        notes=("StableDiffusionXLPipeline via the generic Diffusers adapter.",),
    ),
    "wan-ai/wan2.2-ti2v-5b": ExactRecipe(
        model="Wan-AI/Wan2.2-TI2V-5B",
        runtime="vllm-omni",
        evidence=(
            "Validated endpoint preset: MP4 returned through the dstack proxy "
            "(service wan22-ti2v-5b, 2026-07)."
        ),
        verified_at="2026-07",
        notes=("Wan runtime recipe; the raw repo has no standard model_index.json.",),
    ),
}


def find_exact_recipe(model: str) -> Optional[ExactRecipe]:
    return EXACT_RECIPES.get(model.strip().lower())


__all__ = [
    "DIFFUSERS_ADAPTER_REGISTRY",
    "EXACT_RECIPES",
    "ExactRecipe",
    "LLAMACPP_REGISTRY",
    "RuntimeSupportRegistry",
    "SGLANG_REGISTRY",
    "VLLM_OMNI_REGISTRY",
    "VLLM_REGISTRY",
    "find_exact_recipe",
]
