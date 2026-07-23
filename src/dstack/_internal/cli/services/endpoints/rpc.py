"""Unix-socket JSON-RPC transport between the agent shell and the controller.

The parent CLI process serves the controller's typed operations on a Unix
domain socket inside the workspace root. The agent shell talks to it through
the installed ``endpoint`` command using a random session token — never a
dstack or endpoint credential. One JSON object per line in each direction:

    -> {"token": ..., "op": ..., "params": {...}}
    <- {"ok": true, "result": {...}}
    <- {"ok": false, "error": ..., "failure_class": ...}
"""

import asyncio
import hmac
import json
import secrets
import sys
from pathlib import Path
from typing import Any, Optional

from dstack._internal.cli.services.endpoints.agent import redact
from dstack._internal.cli.services.endpoints.controller import (
    ControllerError,
    EndpointController,
)

CONTROLLER_SOCKET_ENV = "DSTACK_ENDPOINT_CONTROLLER_SOCKET"
CONTROLLER_TOKEN_ENV = "DSTACK_ENDPOINT_CONTROLLER_TOKEN"
_MAX_REQUEST_BYTES = 8 * 1024 * 1024

_OPERATIONS = (
    "submit_candidate",
    "stop_and_handoff",
    "verify_final",
    "finalize_preset",
    "get_endpoint_context",
    "get_run_status",
    "get_run_logs",
    "list_offers",
    "request_service_http",
)


class EndpointControllerServer:
    """Serves one controller on a workspace-local Unix socket."""

    def __init__(
        self,
        controller: EndpointController,
        *,
        socket_path: Path,
        redacted_values: tuple[str, ...] = (),
        token: Optional[str] = None,
    ) -> None:
        self.controller = controller
        self.socket_path = socket_path
        self.token = token or secrets.token_hex(16)
        self._redacted_values = redacted_values
        self._server: Optional[asyncio.AbstractServer] = None

    async def __aenter__(self) -> "EndpointControllerServer":
        self._server = await asyncio.start_unix_server(
            self._handle, path=str(self.socket_path), limit=_MAX_REQUEST_BYTES
        )
        return self

    async def __aexit__(self, *exc_info) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        Path(self.socket_path).unlink(missing_ok=True)

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            line = await reader.readline()
            response = await self._respond(line)
            writer.write((json.dumps(response, ensure_ascii=False, default=str) + "\n").encode())
            await writer.drain()
        except (ConnectionError, asyncio.LimitOverrunError, ValueError):
            pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, OSError):
                pass

    async def _respond(self, line: bytes) -> dict[str, Any]:
        try:
            request = json.loads(line.decode(errors="replace"))
        except json.JSONDecodeError:
            return {"ok": False, "error": "invalid request", "failure_class": "controller"}
        if not isinstance(request, dict):
            return {"ok": False, "error": "invalid request", "failure_class": "controller"}
        token = str(request.get("token") or "")
        if not hmac.compare_digest(token, self.token):
            return {"ok": False, "error": "invalid token", "failure_class": "controller"}
        op = request.get("op")
        params = request.get("params") or {}
        if op not in _OPERATIONS or not isinstance(params, dict):
            return {
                "ok": False,
                "error": f"unknown operation: {op}",
                "failure_class": "controller",
            }
        method = getattr(self.controller, op)
        try:
            result = await asyncio.to_thread(lambda: method(**params))
        except ControllerError as e:
            return {
                "ok": False,
                "error": redact(str(e), self._redacted_values),
                "failure_class": e.failure_class,
            }
        except TypeError as e:
            return {
                "ok": False,
                "error": f"invalid parameters for {op}: {e}",
                "failure_class": "controller",
            }
        except Exception as e:
            return {
                "ok": False,
                "error": redact(f"{op} failed: {e}", self._redacted_values),
                "failure_class": "controller",
            }
        encoded = json.loads(json.dumps(result, default=str))
        return {"ok": True, "result": self._redact_value(encoded)}

    def _redact_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return redact(value, self._redacted_values)
        if isinstance(value, list):
            return [self._redact_value(item) for item in value]
        if isinstance(value, dict):
            return {key: self._redact_value(item) for key, item in value.items()}
        return value


def get_controller_client_script() -> str:
    """The `endpoint` command installed into the agent workspace bin."""
    return f'''#!{sys.executable}
"""Typed run-lifecycle operations for endpoint preset creation.

Usage:
  endpoint submit --file CONFIG.yml --purpose TEXT --workload WORKLOAD.json
  endpoint stop-handoff RUN_NAME [--requirements REQS.json]
  endpoint verify RUN_NAME --workload WORKLOAD.json
  endpoint finalize RUN_NAME --report REPORT.json
  endpoint context
  endpoint status RUN_NAME
  endpoint logs RUN_NAME
  endpoint offers [--filters FILTERS.json]
  endpoint http RUN_NAME METHOD PATH [--body BODY_FILE] [--header K:V ...]

Every command prints one JSON object. `ok` is false when the controller
refused the operation; the error explains why and never contains secrets.
"""
import argparse
import base64
import json
import os
import socket
import sys


def _call(op, params):
    path = os.environ.get({CONTROLLER_SOCKET_ENV!r})
    token = os.environ.get({CONTROLLER_TOKEN_ENV!r})
    if not path or not token:
        print(json.dumps({{"ok": False, "error": "controller socket is not configured"}}))
        raise SystemExit(2)
    request = json.dumps({{"token": token, "op": op, "params": params}}) + "\\n"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.connect(path)
        sock.sendall(request.encode())
        chunks = []
        while True:
            chunk = sock.recv(1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
    raw = b"".join(chunks).decode(errors="replace").strip()
    print(raw)
    try:
        response = json.loads(raw)
    except json.JSONDecodeError:
        raise SystemExit(3)
    raise SystemExit(0 if response.get("ok") else 1)


def _load_json(path, default=None):
    if path is None:
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(prog="endpoint", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    submit = commands.add_parser("submit")
    submit.add_argument("--file", required=True)
    submit.add_argument("--purpose", required=True)
    submit.add_argument("--workload", required=True)

    stop = commands.add_parser("stop-handoff")
    stop.add_argument("run_name")
    stop.add_argument("--requirements")

    verify = commands.add_parser("verify")
    verify.add_argument("run_name")
    verify.add_argument("--workload", required=True)

    finalize = commands.add_parser("finalize")
    finalize.add_argument("run_name")
    finalize.add_argument("--report", required=True)

    commands.add_parser("context")

    status = commands.add_parser("status")
    status.add_argument("run_name")

    logs = commands.add_parser("logs")
    logs.add_argument("run_name")

    offers = commands.add_parser("offers")
    offers.add_argument("--filters")

    http = commands.add_parser("http")
    http.add_argument("run_name")
    http.add_argument("method")
    http.add_argument("path")
    http.add_argument("--body")
    http.add_argument("--header", action="append", default=[])

    args = parser.parse_args()
    if args.command == "submit":
        with open(args.file, encoding="utf-8") as f:
            configuration_yaml = f.read()
        _call(
            "submit_candidate",
            {{
                "configuration_yaml": configuration_yaml,
                "purpose": args.purpose,
                "expected_workload": _load_json(args.workload, {{}}),
            }},
        )
    elif args.command == "stop-handoff":
        _call(
            "stop_and_handoff",
            {{
                "run_name": args.run_name,
                "handoff_requirements": _load_json(args.requirements),
            }},
        )
    elif args.command == "verify":
        _call(
            "verify_final",
            {{"run_name": args.run_name, "declared_workload": _load_json(args.workload, {{}})}},
        )
    elif args.command == "finalize":
        _call(
            "finalize_preset",
            {{"run_name": args.run_name, "report_metadata": _load_json(args.report, {{}})}},
        )
    elif args.command == "context":
        _call("get_endpoint_context", {{}})
    elif args.command == "status":
        _call("get_run_status", {{"run_name": args.run_name}})
    elif args.command == "logs":
        _call("get_run_logs", {{"run_name": args.run_name}})
    elif args.command == "offers":
        _call("list_offers", {{"filters": _load_json(args.filters, {{}})}})
    elif args.command == "http":
        body_base64 = None
        if args.body:
            with open(args.body, "rb") as f:
                body_base64 = base64.b64encode(f.read()).decode()
        headers = {{}}
        for item in args.header:
            key, _, value = item.partition(":")
            headers[key.strip()] = value.strip()
        _call(
            "request_service_http",
            {{
                "run_name": args.run_name,
                "method": args.method,
                "path": args.path,
                "body_base64": body_base64,
                "headers": headers,
            }},
        )


if __name__ == "__main__":
    main()
'''


__all__ = [
    "CONTROLLER_SOCKET_ENV",
    "CONTROLLER_TOKEN_ENV",
    "EndpointControllerServer",
    "get_controller_client_script",
]
