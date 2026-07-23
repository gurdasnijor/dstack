"""Pinned gguf-parser adapter and workload-dependent estimate cache.

Ported from gurdasnijor/dh @ e9ce1b951c9bf08adf57d6576cef5a4897ada3ac
(``dh/gguf.py``). The pinned parser version, per-platform checksums, cache-key
identity, invocation, and output normalization are unchanged; only the storage
locations and environment variable names are dstack-owned.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

from dstack._internal.utils.common import get_dstack_dir

GGUF_ESTIMATE_CACHE_SCHEMA = 1
GGUF_PARSER_VERSION = "v0.25.0"
GGUF_PARSER_RELEASE = (
    f"https://github.com/gpustack/gguf-parser-go/releases/download/{GGUF_PARSER_VERSION}"
)
GGUF_PARSER_ASSETS = {
    ("darwin", "x86_64"): (
        "gguf-parser-darwin-amd64",
        "db8c40bb12189485dd8f9631a45dc84bfda0ef400273aa97e0bce467b1b8cc30",
    ),
    ("darwin", "arm64"): (
        "gguf-parser-darwin-arm64",
        "904454340d8a917aaa9ba3c81bfdaa7f484f1c5296252669e8dff6ee88cd162b",
    ),
    ("linux", "x86_64"): (
        "gguf-parser-linux-amd64",
        "18b9bebfef7c40b661ef265e755a5096bf86150261c3321896ff09bc5d663b32",
    ),
    ("linux", "aarch64"): (
        "gguf-parser-linux-arm64",
        "62e749c68fb9bcab710460352d27b4e9a8862aa0ecb3e93c88eca3d6f71ab064",
    ),
    ("windows", "x86_64"): (
        "gguf-parser-windows-amd64.exe",
        "34ac741259c1b0349482124da5a944cc03463730dfb6cf28334e91dc9b1d3182",
    ),
    ("windows", "arm64"): (
        "gguf-parser-windows-arm64.exe",
        "b7aaa5f732b239e6df1eba46486834c7c8bb5e793def1008033f09ed670dafd8",
    ),
}
_PARSER_ENV = "DSTACK_GGUF_PARSER"
_AUTO_INSTALL_ENV = "DSTACK_GGUF_PARSER_AUTO_INSTALL"


class GGUFParserError(RuntimeError):
    pass


def _platform_key(system: str | None = None, machine: str | None = None) -> tuple[str, str]:
    normalized_system = (system or platform.system()).lower()
    normalized_machine = (machine or platform.machine()).lower()
    normalized_machine = {
        "amd64": "x86_64",
        "arm64": "arm64" if normalized_system != "linux" else "aarch64",
    }.get(normalized_machine, normalized_machine)
    return normalized_system, normalized_machine


def managed_parser_asset(
    system: str | None = None, machine: str | None = None
) -> tuple[str, str] | None:
    return GGUF_PARSER_ASSETS.get(_platform_key(system, machine))


def managed_parser_path() -> Path | None:
    asset = managed_parser_asset()
    if asset is None:
        return None
    suffix = ".exe" if asset[0].endswith(".exe") else ""
    return (
        get_dstack_dir() / "tools" / "gguf-parser" / GGUF_PARSER_VERSION / f"gguf-parser{suffix}"
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _auto_install_enabled() -> bool:
    value = os.environ.get(_AUTO_INSTALL_ENV, "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def install_managed_parser() -> str | None:
    asset = managed_parser_asset()
    target = managed_parser_path()
    if asset is None or target is None:
        return None
    name, expected_digest = asset
    if target.is_file() and _file_sha256(target) == expected_digest:
        return str(target)

    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        f"{GGUF_PARSER_RELEASE}/{name}", headers={"User-Agent": "dstack-endpoint-inspect"}
    )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=".gguf-parser-", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            digest = hashlib.sha256()
            with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
                while chunk := response.read(1024 * 1024):
                    temporary.write(chunk)
                    digest.update(chunk)
        if digest.hexdigest() != expected_digest:
            raise GGUFParserError(
                "downloaded gguf-parser checksum does not match the pinned release"
            )
        temporary_path.chmod(0o755)
        temporary_path.replace(target)
    except GGUFParserError:
        raise
    except OSError as error:
        raise GGUFParserError(
            f"cannot download pinned gguf-parser {GGUF_PARSER_VERSION}: {error}"
        ) from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return str(target)


def parser_executable(*, install: bool = True) -> str | None:
    configured = os.environ.get(_PARSER_ENV)
    if configured:
        return configured
    if executable := shutil.which("gguf-parser"):
        return executable
    asset = managed_parser_asset()
    target = managed_parser_path()
    if asset is not None and target is not None and target.is_file():
        if _file_sha256(target) == asset[1]:
            return str(target)
    if install and _auto_install_enabled():
        return install_managed_parser()
    return None


def parser_version(executable: str) -> str:
    result = subprocess.run([executable, "--version"], capture_output=True, text=True)
    if result.returncode:
        raise GGUFParserError(
            f"gguf-parser --version failed: {(result.stderr or result.stdout).strip()}"
        )
    version = result.stdout.strip() or result.stderr.strip()
    if not version:
        raise GGUFParserError("gguf-parser returned an empty version")
    return version


def estimate_key_payload(
    *,
    repo_sha: str,
    files: Sequence[str],
    parser_version_value: str,
    context: int,
    parallel_size: int,
    gpu_layers: int,
    gpu_layers_step: int | None = None,
    runtime_arguments: Mapping[str, Any] | None = None,
    cache_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema": GGUF_ESTIMATE_CACHE_SCHEMA,
        "repo_sha": repo_sha,
        "files": sorted(files),
        "parser_version": parser_version_value,
        "context": int(context),
        "parallel_size": int(parallel_size),
        "gpu_layers": int(gpu_layers),
        "gpu_layers_step": (int(gpu_layers_step) if gpu_layers_step is not None else None),
        "runtime_arguments": dict(sorted((runtime_arguments or {}).items())),
        "cache_context": dict(sorted((cache_context or {}).items())),
    }


def estimate_cache_path(payload: Mapping[str, Any]) -> Path:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    key = hashlib.sha256(encoded).hexdigest()[:24]
    return get_dstack_dir() / "cache" / "gguf-est" / "v1" / f"{key}.json"


def _read_cache(path: Path, payload: Mapping[str, Any]) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or value.get("key") != payload:
        return None
    result = value.get("result")
    return result if isinstance(result, dict) else None


def _write_cache(path: Path, payload: Mapping[str, Any], result: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    value = {"key": payload, "result": result}
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def estimate(
    *,
    repo: str,
    repo_sha: str,
    selected_file: str,
    shard_set: Sequence[str],
    context: int,
    parallel_size: int,
    gpu_layers: int,
    gpu_layers_step: int | None = None,
    runtime_arguments: Mapping[str, Any] | None = None,
    cache_context: Mapping[str, Any] | None = None,
    refresh: bool = False,
    executable: str | None = None,
) -> dict[str, Any] | None:
    executable = executable or parser_executable()
    if executable is None:
        return None
    version = parser_version(executable)
    payload = estimate_key_payload(
        repo_sha=repo_sha,
        files=shard_set,
        parser_version_value=version,
        context=context,
        parallel_size=parallel_size,
        gpu_layers=gpu_layers,
        gpu_layers_step=gpu_layers_step,
        runtime_arguments=runtime_arguments,
        cache_context=cache_context,
    )
    cache = estimate_cache_path(payload)
    if not refresh:
        cached = _read_cache(cache, payload)
        if cached is not None:
            return cached
    command = [
        executable,
        "--hf-repo",
        repo,
        "--hf-file",
        selected_file,
        "--ctx-size",
        str(context),
        "--parallel-size",
        str(parallel_size),
        "--gpu-layers",
        str(gpu_layers),
        "--estimate",
        "--json",
    ]
    if gpu_layers_step is not None:
        command.extend(["--gpu-layers-step", str(gpu_layers_step)])
    for name, value in sorted((runtime_arguments or {}).items()):
        command.append(str(name))
        if value is not True and value is not None:
            command.append(str(value))
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise GGUFParserError(
            f"gguf-parser estimate failed: {(result.stderr or result.stdout).strip()}"
        )
    try:
        parsed = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise GGUFParserError("gguf-parser returned invalid JSON") from error
    if not isinstance(parsed, dict):
        raise GGUFParserError("gguf-parser JSON result must be an object")
    _write_cache(cache, payload, parsed)
    return parsed


def configurations(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    estimate_result = result.get("estimate")
    items = estimate_result.get("items") if isinstance(estimate_result, Mapping) else None
    if isinstance(items, list):
        parsed_items: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, Mapping):
                continue
            ram = item.get("ram")
            vrams = item.get("vrams")
            if not isinstance(ram, Mapping) or not isinstance(vrams, list):
                continue
            try:
                host = int(ram["nonuma"])
                device_values = [
                    int(device["nonuma"])
                    for device in vrams
                    if isinstance(device, Mapping) and device.get("nonuma") is not None
                ]
                layers = int(item.get("offloadLayers") or 0)
            except (KeyError, TypeError, ValueError):
                continue
            full = item.get("fullOffloaded") is True
            strategy = (
                "full-offload" if full else "cpu-only" if layers == 0 else f"partial:{layers}"
            )
            parsed_items.append(
                {
                    "strategy": strategy,
                    "gpu_layers": layers,
                    "vram_bytes": max(device_values, default=0),
                    "host_memory_bytes": host,
                    "raw": dict(item),
                }
            )
        return parsed_items

    raw = result.get("estimates") or result.get("configurations")
    values = raw if isinstance(raw, list) else [result]
    normalized: list[dict[str, Any]] = []
    for value in values:
        if not isinstance(value, Mapping):
            continue
        vram = value.get("vram_bytes") or value.get("gpu_memory_bytes")
        host = value.get("ram_bytes") or value.get("host_memory_bytes")
        if vram is None or host is None:
            continue
        layers = int(value.get("gpu_layers") or 0)
        strategy = value.get("strategy")
        if not strategy:
            strategy = (
                "cpu-only"
                if layers == 0
                else "full-offload"
                if layers >= 999
                else f"partial:{layers}"
            )
        normalized.append(
            {
                "strategy": str(strategy),
                "gpu_layers": layers,
                "vram_bytes": int(vram),
                "host_memory_bytes": int(host),
                "raw": dict(value),
            }
        )
    return normalized


__all__ = [
    "GGUF_PARSER_VERSION",
    "GGUFParserError",
    "configurations",
    "estimate",
    "parser_executable",
]
