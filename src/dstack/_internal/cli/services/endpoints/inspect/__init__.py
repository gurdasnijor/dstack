"""Deterministic, metadata-only model inspection for endpoint preset creation.

The algorithms in this package are ported from the reference implementation in
`gurdasnijor/dh` at commit e9ce1b951c9bf08adf57d6576cef5a4897ada3ac. Parity is
enforced by golden fixtures generated from that commit; see
`src/tests/_internal/cli/services/endpoints/inspect/` for the fixture corpus and
regeneration instructions.
"""

from dstack._internal.cli.services.endpoints.inspect.classify import (
    ModelInspection,
    RuntimeCandidate,
    classify_model,
)
from dstack._internal.cli.services.endpoints.inspect.hub import (
    HubSnapshot,
    ModelMetadataError,
    build_model_shape,
    build_model_shape_from_fixture,
    fetch_hub_snapshot,
)
from dstack._internal.cli.services.endpoints.inspect.service import (
    inspect_endpoint_model,
    inspection_from_snapshot,
)

DH_SOURCE_COMMIT = "e9ce1b951c9bf08adf57d6576cef5a4897ada3ac"

__all__ = [
    "DH_SOURCE_COMMIT",
    "HubSnapshot",
    "ModelInspection",
    "ModelMetadataError",
    "RuntimeCandidate",
    "build_model_shape",
    "build_model_shape_from_fixture",
    "classify_model",
    "fetch_hub_snapshot",
    "inspect_endpoint_model",
    "inspection_from_snapshot",
]
