#!/usr/bin/env python3
"""Benchmark and validate an OpenAI-compatible image generation endpoint."""

import argparse
import base64
import json
import math
import os
import shlex
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib import request
from urllib.parse import urlparse

VERSION = "1.0.0"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--body", required=True, type=Path)
    parser.add_argument("--bearer-env", default="DSTACK_ENDPOINT_BEARER_TOKEN")
    parser.add_argument("--requests", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.requests < 1 or args.concurrency < 1 or args.warmups < 0:
        parser.error("requests/concurrency must be positive and warmups must be non-negative")

    body = json.loads(args.body.read_text(encoding="utf-8"))
    if not isinstance(body, dict):
        parser.error("body must contain a JSON object")
    if body.get("response_format") != "b64_json":
        parser.error("body must set response_format to b64_json")

    token = os.getenv(args.bearer_env)
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    payload = json.dumps(body).encode()
    expected_width, expected_height = _requested_dimensions(body)
    expected_outputs = int(body.get("n", 1))

    for _ in range(args.warmups):
        _send_and_validate(
            args.url,
            payload,
            headers,
            args.timeout,
            expected_width,
            expected_height,
            expected_outputs,
        )

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        results = list(
            executor.map(
                lambda _: _send_and_validate(
                    args.url,
                    payload,
                    headers,
                    args.timeout,
                    expected_width,
                    expected_height,
                    expected_outputs,
                ),
                range(args.requests),
            )
        )
    duration = time.perf_counter() - started
    latencies = [result[0] for result in results]
    total_outputs = sum(result[1] for result in results)
    total_bytes = sum(result[2] for result in results)
    request_path = urlparse(args.url).path
    if request_path.endswith("/v1/images/generations"):
        request_path = "/v1/images/generations"

    workload: dict[str, Any] = {
        "api": "images_generations",
        "request_path": request_path,
        "num_requests": args.requests,
        "concurrency": args.concurrency,
        "outputs_per_request": expected_outputs,
        "output_unit": "image",
        "parameters": {
            key: body[key] for key in ("response_format", "seed", "guidance_scale") if key in body
        },
    }
    if expected_width is not None and expected_height is not None:
        workload.update(width=expected_width, height=expected_height)
    steps = body.get("num_inference_steps", body.get("steps"))
    if isinstance(steps, int) and steps > 0:
        workload["num_inference_steps"] = steps

    report = {
        "tool": "dstack benchmark images",
        "tool_version": VERSION,
        "command": "python benchmark_images.py " + shlex.join(sys.argv[1:]),
        "workload": workload,
        "metrics": {
            "successful_requests": args.requests,
            "failed_requests": 0,
            "duration_seconds": duration,
            "total_outputs": total_outputs,
            "total_output_bytes": total_bytes,
            "latency_ms": {
                "mean": sum(latencies) / len(latencies),
                "p50": _percentile(latencies, 0.50),
                "p99": _percentile(latencies, 0.99),
            },
        },
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output is None:
        sys.stdout.write(rendered)
    else:
        args.output.write_text(rendered, encoding="utf-8")
    return 0


def _send_and_validate(
    url: str,
    payload: bytes,
    headers: dict[str, str],
    timeout: float,
    expected_width: int | None,
    expected_height: int | None,
    expected_outputs: int,
) -> tuple[float, int, int]:
    started = time.perf_counter()
    with request.urlopen(
        request.Request(url, data=payload, headers=headers, method="POST"),
        timeout=timeout,
    ) as response:
        response_body = response.read()
    latency_ms = (time.perf_counter() - started) * 1000
    data = json.loads(response_body)
    outputs = data.get("data") if isinstance(data, dict) else None
    if not isinstance(outputs, list) or len(outputs) != expected_outputs:
        raise ValueError(f"expected {expected_outputs} image outputs")
    total_bytes = 0
    for output in outputs:
        encoded = output.get("b64_json") if isinstance(output, dict) else None
        if not isinstance(encoded, str) or not encoded:
            raise ValueError("image output does not contain b64_json")
        image = base64.b64decode(encoded, validate=True)
        width, height = _image_dimensions(image)
        if expected_width is not None and (width, height) != (expected_width, expected_height):
            raise ValueError(
                f"expected {expected_width}x{expected_height} image, got {width}x{height}"
            )
        total_bytes += len(image)
    return latency_ms, len(outputs), total_bytes


def _requested_dimensions(body: dict[str, Any]) -> tuple[int | None, int | None]:
    if isinstance(body.get("width"), int) and isinstance(body.get("height"), int):
        return body["width"], body["height"]
    size = body.get("size")
    if isinstance(size, str) and "x" in size:
        width, height = size.lower().split("x", 1)
        if width.isdigit() and height.isdigit():
            return int(width), int(height)
    return None, None


def _image_dimensions(data: bytes) -> tuple[int, int]:
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
        return struct.unpack(">II", data[16:24])
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP" and len(data) >= 30:
        if data[12:16] == b"VP8X":
            width = 1 + int.from_bytes(data[24:27], "little")
            height = 1 + int.from_bytes(data[27:30], "little")
            return width, height
    if data.startswith(b"\xff\xd8"):
        index = 2
        while index + 9 < len(data):
            if data[index] != 0xFF:
                index += 1
                continue
            marker = data[index + 1]
            if marker in {
                0xC0,
                0xC1,
                0xC2,
                0xC3,
                0xC5,
                0xC6,
                0xC7,
                0xC9,
                0xCA,
                0xCB,
                0xCD,
                0xCE,
                0xCF,
            }:
                return struct.unpack(">HH", data[index + 5 : index + 9])[::-1]
            if index + 4 > len(data):
                break
            index += 2 + int.from_bytes(data[index + 2 : index + 4], "big")
    raise ValueError("image output is not a supported PNG, JPEG, or WebP image")


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


if __name__ == "__main__":
    raise SystemExit(main())
