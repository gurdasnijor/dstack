"""Controller-owned `inspect_model` stage for endpoint preset creation.

Runs before the agent is launched: resolves the requested revision to an
immutable SHA, captures the raw normalized Hub metadata snapshot, and produces
ranked runtime candidates with evidence. The stage is advisory — when it cannot
run (non-Hub source, network failure), preset creation continues on the agent
research path with an explicit note.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from dstack._internal.cli.models.endpoints import EndpointConfiguration
from dstack._internal.cli.services.endpoints.inspect.classify import (
    ModelInspection,
    classify_model,
)
from dstack._internal.cli.services.endpoints.inspect.hub import (
    HubSnapshot,
    build_model_shape,
    fetch_hub_snapshot,
    split_model_selector,
)

_HF_REPO_RE = re.compile(r"^[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*$")


@dataclass(frozen=True)
class InspectionResult:
    inspection: Optional[ModelInspection]
    snapshot: Optional[HubSnapshot]
    skipped_reason: Optional[str] = None
    error: Optional[str] = None


def _hub_repo_for_configuration(configuration: EndpointConfiguration) -> Optional[str]:
    model = configuration.model
    if model.source_type not in ("auto", "huggingface"):
        return None
    locator = model.exact_repo or (model.api_model_name if model.allows_variant_selection else "")
    if not locator:
        return None
    repo, _selector = split_model_selector(locator)
    if "://" in repo or repo.startswith((".", "/", "~")):
        return None
    if not _HF_REPO_RE.fullmatch(repo):
        return None
    return locator


def inspect_endpoint_model(
    configuration: EndpointConfiguration,
    *,
    token: Optional[str] = None,
) -> InspectionResult:
    """Run the deterministic inspection stage for a Hugging Face model source."""
    locator = _hub_repo_for_configuration(configuration)
    if locator is None:
        return InspectionResult(
            inspection=None,
            snapshot=None,
            skipped_reason=(
                "model source is not a Hugging Face repository id; "
                "deterministic Hub inspection does not apply"
            ),
        )
    requested_revision = configuration.model.requested_revision
    try:
        snapshot = fetch_hub_snapshot(locator, revision=requested_revision, token=token)
        _repo, selector = split_model_selector(locator)
        ir = build_model_shape(
            snapshot.repo,
            snapshot.revision,
            snapshot.model_info,
            snapshot.documents,
            gguf_selector=selector,
        )
        inspection = classify_model(ir, requested_revision=requested_revision)
    except Exception as error:
        return InspectionResult(
            inspection=None,
            snapshot=None,
            error=f"deterministic model inspection failed: {error}",
        )
    return InspectionResult(inspection=inspection, snapshot=snapshot)


def inspection_from_snapshot(
    snapshot: Mapping[str, Any],
    *,
    requested_revision: Optional[str] = None,
    gguf_selector: Optional[str] = None,
    gguf_file: Optional[str] = None,
) -> ModelInspection:
    """Classify from a stored snapshot or captured fixture, fully offline.

    Accepts either a :class:`HubSnapshot` ``to_data()`` payload or a dh-style
    model fixture (``model_info`` + ``files``).
    """
    documents = snapshot.get("documents")
    if documents is None:
        documents = snapshot.get("files") or {}
    ir = build_model_shape(
        str(snapshot["repo"]),
        str(snapshot["revision"]),
        snapshot["model_info"],
        documents,
        gguf_selector=gguf_selector,
        gguf_file=gguf_file,
    )
    return classify_model(ir, requested_revision=requested_revision)


__all__ = [
    "InspectionResult",
    "inspect_endpoint_model",
    "inspection_from_snapshot",
]
