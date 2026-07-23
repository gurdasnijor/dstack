"""Transport tests for the controller's Unix-socket JSON-RPC."""

import asyncio
import json
import shutil
import tempfile
from pathlib import Path

import pytest

from dstack._internal.cli.services.endpoints.controller import ControllerError
from dstack._internal.cli.services.endpoints.rpc import EndpointControllerServer

# Unix sockets: POSIX only (unmarked tests are skipped on Windows).
pytestmark = pytest.mark.asyncio


@pytest.fixture
def socket_dir():
    # Keep the socket path under the macOS AF_UNIX length limit.
    path = Path(tempfile.mkdtemp(prefix="dpe-rpc-", dir="/tmp"))
    yield path
    shutil.rmtree(path, ignore_errors=True)


class ScriptedController:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def get_endpoint_context(self):
        self.calls.append(("get_endpoint_context", {}))
        return {"policy": {"build_name": "qwen-build"}, "note": "token dstack-secret inside"}

    def submit_candidate(self, **params):
        self.calls.append(("submit_candidate", params))
        if params.get("purpose") == "refused":
            raise ControllerError("run budget exhausted", failure_class="budget")
        return {"run_name": "qwen-build-1"}


async def _call(socket_path, payload) -> dict:
    reader, writer = await asyncio.open_unix_connection(str(socket_path))
    writer.write((json.dumps(payload) + "\n").encode())
    await writer.drain()
    writer.write_eof()
    line = await reader.readline()
    writer.close()
    return json.loads(line.decode())


async def test_round_trip_refusals_and_redaction(socket_dir):
    controller = ScriptedController()
    server = EndpointControllerServer(
        controller,  # type: ignore[arg-type]
        socket_path=socket_dir / "ctl.sock",
        redacted_values=("dstack-secret",),
    )
    async with server:
        ok = await _call(
            server.socket_path,
            {
                "token": server.token,
                "op": "submit_candidate",
                "params": {"purpose": "candidate"},
            },
        )
        refused = await _call(
            server.socket_path,
            {
                "token": server.token,
                "op": "submit_candidate",
                "params": {"purpose": "refused"},
            },
        )
        context = await _call(
            server.socket_path, {"token": server.token, "op": "get_endpoint_context", "params": {}}
        )
        bad_token = await _call(
            server.socket_path,
            {"token": "wrong", "op": "get_endpoint_context", "params": {}},
        )
        unknown = await _call(
            server.socket_path, {"token": server.token, "op": "drop_tables", "params": {}}
        )

    assert ok == {"ok": True, "result": {"run_name": "qwen-build-1"}}
    assert refused["ok"] is False
    assert refused["failure_class"] == "budget"
    assert "run budget exhausted" in refused["error"]
    # Secrets never cross the socket, even when embedded in results.
    assert "dstack-secret" not in json.dumps(context)
    assert "[redacted]" in context["result"]["note"]
    assert bad_token == {"ok": False, "error": "invalid token", "failure_class": "controller"}
    assert unknown["ok"] is False and "unknown operation" in unknown["error"]
    assert [name for name, _ in controller.calls] == [
        "submit_candidate",
        "submit_candidate",
        "get_endpoint_context",
    ]


async def test_socket_is_removed_on_exit(socket_dir):
    server = EndpointControllerServer(
        ScriptedController(),  # type: ignore[arg-type]
        socket_path=socket_dir / "ctl.sock",
    )
    async with server:
        assert server.socket_path.exists()
    assert not server.socket_path.exists()
