"""
Shared constants and settings for proxy components (gateway + in-server proxy).
"""

import os
from pathlib import Path

# Inference endpoints exposed by the in-replica HTTP router. Applies to both
# SGLang's router and Dynamo's `dynamo.frontend` — they share the
# OpenAI-compatible endpoint surface.
ROUTER_WHITELISTED_PATHS: tuple[str, ...] = (
    "/generate",
    "/v1/",
    "/chat/completions",
)

# Generated-asset retention policy. See
# `dstack._internal.proxy.lib.services.generated_assets` for semantics.
PROXY_ASSETS_DEFAULT_RETENTION_DAYS = 30
PROXY_ASSETS_DEFAULT_MAX_TOTAL_BYTES = 10 * 1024**3  # 10 GiB


def get_proxy_assets_dir() -> Path:
    configured = os.getenv("DSTACK_PROXY_ASSETS_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    server_dir = Path(os.getenv("DSTACK_SERVER_DIR", "~/.dstack/server")).expanduser()
    return (server_dir / "data" / "generated-assets").resolve()


def get_proxy_assets_retention_days() -> int:
    """Days after which a generated asset expires. `0` disables age expiry."""
    return _get_non_negative_int_env(
        "DSTACK_PROXY_ASSETS_RETENTION_DAYS", PROXY_ASSETS_DEFAULT_RETENTION_DAYS
    )


def get_proxy_assets_max_total_bytes() -> int:
    """Total blob size budget for generated assets. `0` disables the budget."""
    return _get_non_negative_int_env(
        "DSTACK_PROXY_ASSETS_MAX_TOTAL_BYTES", PROXY_ASSETS_DEFAULT_MAX_TOTAL_BYTES
    )


def _get_non_negative_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative, got {parsed}")
    return parsed
