"""
Durable store for generated media assets (images, videos) produced by model
endpoints and served through the proxy.

Layout (under the assets root, `DSTACK_PROXY_ASSETS_DIR` or
`$DSTACK_SERVER_DIR/data/generated-assets`):

```
blobs/<asset_id>        raw bytes
meta/<asset_id>.json    metadata (see AssetMetadata)
```

The legacy flat layout (`<asset_id>.bin` + `<asset_id>.json` side by side,
retained newest-32) is migrated in place on first use.

Retention policy
----------------
Assets are retained by age and by total size, not by a fixed count:

- `DSTACK_PROXY_ASSETS_RETENTION_DAYS` (default 30): an asset expires this
  many days after creation. `0` disables age-based expiry. The expiry computed
  at write time is stored with the asset; assets written before this policy
  existed expire relative to their creation time using the current setting.
- `DSTACK_PROXY_ASSETS_MAX_TOTAL_BYTES` (default 10 GiB): when the total blob
  size exceeds this budget, the oldest assets are collected first until the
  store fits. `0` disables the size budget. The newest asset is never
  collected for budget reasons.

Garbage collection runs after every write and can be invoked manually,
including as a dry run, via:

```
python -m dstack._internal.proxy.lib.services.generated_assets gc [--apply]
python -m dstack._internal.proxy.lib.services.generated_assets backup <archive.tar.gz>
python -m dstack._internal.proxy.lib.services.generated_assets restore <archive.tar.gz> [--overwrite]
```

Backup / restore
----------------
`export_backup` produces a self-describing tar.gz (manifest with per-blob
SHA-256, metadata, and blobs) and `restore_backup` verifies and restores it,
so user-visible assets survive a server replacement, not just a container
restart, when backups are taken from the durable volume.
"""

import argparse
import hashlib
import json
import secrets
import sys
import tarfile
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from dstack._internal.proxy.lib.const import (
    PROXY_ASSETS_DEFAULT_MAX_TOTAL_BYTES,
    PROXY_ASSETS_DEFAULT_RETENTION_DAYS,
    get_proxy_assets_dir,
    get_proxy_assets_max_total_bytes,
    get_proxy_assets_retention_days,
)

ASSET_ID_PREFIX = "dstack_asset_"
CACHED_VIDEO_ID_PREFIX = "dstack_cached_"

_ASSET_ID_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
_BACKUP_MANIFEST_VERSION = 1


@dataclass(frozen=True)
class AssetMetadata:
    project_name: str
    model_name: str
    media_type: str
    created_at: int
    size_bytes: int = 0
    model_revision: Optional[str] = None
    created_by: Optional[str] = None
    expires_at: Optional[int] = None
    """Unix timestamp after which the asset may be garbage-collected.
    `None` means no age-based expiry was recorded at write time."""


@dataclass(frozen=True)
class StoredAsset:
    metadata: AssetMetadata
    content: bytes

    @property
    def project_name(self) -> str:
        return self.metadata.project_name

    @property
    def model_name(self) -> str:
        return self.metadata.model_name

    @property
    def media_type(self) -> str:
        return self.metadata.media_type

    @property
    def created_at(self) -> int:
        return self.metadata.created_at


@dataclass(frozen=True)
class GCEntry:
    asset_id: str
    reason: str  # "expired", "size-budget", or "orphaned"
    size_bytes: int
    created_at: Optional[int]


@dataclass
class GCPlan:
    dry_run: bool
    collected: list[GCEntry] = field(default_factory=list)
    retained_count: int = 0
    retained_bytes: int = 0

    @property
    def collected_bytes(self) -> int:
        return sum(e.size_bytes for e in self.collected)


class GeneratedAssetStore:
    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = (root or get_proxy_assets_dir()).resolve()
        self.blobs_dir = self.root / "blobs"
        self.meta_dir = self.root / "meta"

    def store(
        self,
        project_name: str,
        model_name: str,
        content: bytes,
        media_type: str,
        *,
        prefix: str = ASSET_ID_PREFIX,
        model_revision: Optional[str] = None,
        created_by: Optional[str] = None,
    ) -> str:
        self._ensure_layout()
        asset_id = f"{prefix}{secrets.token_urlsafe(18)}"
        now = int(time.time())
        retention_days = get_proxy_assets_retention_days()
        metadata = AssetMetadata(
            project_name=project_name,
            model_name=model_name,
            media_type=media_type,
            created_at=now,
            size_bytes=len(content),
            model_revision=model_revision,
            created_by=created_by,
            expires_at=now + retention_days * 86400 if retention_days else None,
        )
        # The blob is written first and the metadata write is the commit point:
        # `get` requires both files.
        self._blob_path(asset_id).write_bytes(content)
        self._write_metadata(asset_id, metadata)
        self.collect_garbage()
        return asset_id

    def get(self, project_name: str, asset_id: str) -> Optional[StoredAsset]:
        if not _is_valid_asset_id(asset_id):
            return None
        self._ensure_layout()
        metadata = self._read_metadata(asset_id)
        if metadata is None or metadata.project_name != project_name:
            return None
        try:
            content = self._blob_path(asset_id).read_bytes()
        except FileNotFoundError:
            return None
        return StoredAsset(metadata=metadata, content=content)

    def list_metadata(self) -> dict[str, AssetMetadata]:
        self._ensure_layout()
        assets = {}
        for meta_path in self.meta_dir.glob("*.json"):
            metadata = self._read_metadata(meta_path.stem)
            if metadata is not None:
                assets[meta_path.stem] = metadata
        return assets

    def plan_gc(self, now: Optional[int] = None) -> GCPlan:
        return self._plan_gc(now if now is not None else int(time.time()))

    def collect_garbage(self, dry_run: bool = False, now: Optional[int] = None) -> GCPlan:
        plan = self.plan_gc(now)
        plan.dry_run = dry_run
        if dry_run:
            return plan
        for entry in plan.collected:
            # Metadata first: an interrupted collection leaves an orphaned
            # blob, which a later run removes as "orphaned".
            self._meta_path(entry.asset_id).unlink(missing_ok=True)
            self._blob_path(entry.asset_id).unlink(missing_ok=True)
        return plan

    def export_backup(self, archive_path: Path) -> int:
        """Writes a verified backup archive. Returns the number of assets."""
        self._ensure_layout()
        assets = self.list_metadata()
        manifest = {
            "version": _BACKUP_MANIFEST_VERSION,
            "created_at": int(time.time()),
            "assets": [],
        }
        with tarfile.open(archive_path, "w:gz") as tar:
            for asset_id, metadata in sorted(assets.items()):
                blob_path = self._blob_path(asset_id)
                try:
                    content = blob_path.read_bytes()
                except FileNotFoundError:
                    continue
                manifest["assets"].append(
                    {
                        "id": asset_id,
                        "sha256": hashlib.sha256(content).hexdigest(),
                        "size_bytes": len(content),
                        "metadata": asdict(metadata),
                    }
                )
                tar.add(blob_path, arcname=f"blobs/{asset_id}")
                tar.add(self._meta_path(asset_id), arcname=f"meta/{asset_id}.json")
            manifest_bytes = json.dumps(manifest, indent=2).encode()
            with tempfile.NamedTemporaryFile() as f:
                f.write(manifest_bytes)
                f.flush()
                tar.add(f.name, arcname="manifest.json")
        return len(manifest["assets"])

    def restore_backup(self, archive_path: Path, overwrite: bool = False) -> int:
        """Restores a backup archive, verifying blob integrity.

        Returns the number of restored assets. Existing assets are kept
        unless `overwrite` is set.
        """
        self._ensure_layout()
        restored = 0
        with tarfile.open(archive_path, "r:gz") as tar:
            manifest_member = tar.extractfile("manifest.json")
            if manifest_member is None:
                raise ValueError("Backup archive has no manifest.json")
            manifest = json.loads(manifest_member.read())
            if manifest.get("version") != _BACKUP_MANIFEST_VERSION:
                raise ValueError(f"Unsupported backup manifest version: {manifest.get('version')}")
            for entry in manifest["assets"]:
                asset_id = entry["id"]
                if not _is_valid_asset_id(asset_id):
                    raise ValueError(f"Backup contains invalid asset ID: {asset_id!r}")
                if self._meta_path(asset_id).exists() and not overwrite:
                    continue
                blob_member = tar.extractfile(f"blobs/{asset_id}")
                meta_member = tar.extractfile(f"meta/{asset_id}.json")
                if blob_member is None or meta_member is None:
                    raise ValueError(f"Backup is missing files for asset {asset_id}")
                content = blob_member.read()
                if hashlib.sha256(content).hexdigest() != entry["sha256"]:
                    raise ValueError(f"Backup blob checksum mismatch for asset {asset_id}")
                metadata = _parse_metadata(json.loads(meta_member.read()))
                if metadata is None:
                    raise ValueError(f"Backup has invalid metadata for asset {asset_id}")
                self._blob_path(asset_id).write_bytes(content)
                self._write_metadata(asset_id, metadata)
                restored += 1
        return restored

    def _plan_gc(self, now: int) -> GCPlan:
        self._ensure_layout()
        plan = GCPlan(dry_run=True)
        retention_days = get_proxy_assets_retention_days()
        max_total_bytes = get_proxy_assets_max_total_bytes()

        live: list[tuple[str, AssetMetadata]] = []
        for meta_path in self.meta_dir.glob("*.json"):
            asset_id = meta_path.stem
            metadata = self._read_metadata(asset_id)
            if metadata is None or not self._blob_path(asset_id).exists():
                plan.collected.append(
                    GCEntry(asset_id=asset_id, reason="orphaned", size_bytes=0, created_at=None)
                )
                continue
            expires_at = metadata.expires_at
            if expires_at is None and retention_days:
                # Asset written before expiry stamping: apply the current policy.
                expires_at = metadata.created_at + retention_days * 86400
            if expires_at is not None and expires_at <= now:
                plan.collected.append(
                    GCEntry(
                        asset_id=asset_id,
                        reason="expired",
                        size_bytes=metadata.size_bytes,
                        created_at=metadata.created_at,
                    )
                )
                continue
            live.append((asset_id, metadata))

        # Blobs without metadata are unreachable; collect them.
        meta_ids = {p.stem for p in self.meta_dir.glob("*.json")}
        for blob_path in self.blobs_dir.iterdir():
            if blob_path.name not in meta_ids:
                plan.collected.append(
                    GCEntry(
                        asset_id=blob_path.name,
                        reason="orphaned",
                        size_bytes=blob_path.stat().st_size,
                        created_at=None,
                    )
                )

        live.sort(key=lambda item: item[1].created_at)
        total_bytes = sum(m.size_bytes for _, m in live)
        if max_total_bytes:
            for asset_id, metadata in live[:-1] if live else []:
                if total_bytes <= max_total_bytes:
                    break
                plan.collected.append(
                    GCEntry(
                        asset_id=asset_id,
                        reason="size-budget",
                        size_bytes=metadata.size_bytes,
                        created_at=metadata.created_at,
                    )
                )
                total_bytes -= metadata.size_bytes
        collected_ids = {e.asset_id for e in plan.collected}
        plan.retained_count = sum(1 for asset_id, _ in live if asset_id not in collected_ids)
        plan.retained_bytes = sum(
            m.size_bytes for asset_id, m in live if asset_id not in collected_ids
        )
        return plan

    def _ensure_layout(self) -> None:
        self.blobs_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self._migrate_legacy_layout()

    def _migrate_legacy_layout(self) -> None:
        for legacy_blob in self.root.glob("*.bin"):
            asset_id = legacy_blob.stem
            if not _is_valid_asset_id(asset_id):
                continue
            legacy_meta = self.root / f"{asset_id}.json"
            if not legacy_meta.exists():
                legacy_blob.unlink(missing_ok=True)
                continue
            try:
                legacy_data = json.loads(legacy_meta.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            metadata = _parse_metadata(legacy_data)
            if metadata is None:
                continue
            if metadata.size_bytes == 0:
                metadata = AssetMetadata(
                    **{**asdict(metadata), "size_bytes": legacy_blob.stat().st_size}
                )
            legacy_blob.rename(self._blob_path(asset_id))
            self._write_metadata(asset_id, metadata)
            legacy_meta.unlink(missing_ok=True)

    def _write_metadata(self, asset_id: str, metadata: AssetMetadata) -> None:
        path = self._meta_path(asset_id)
        tmp_path = path.with_name(f".{path.name}.tmp")
        tmp_path.write_text(json.dumps(asdict(metadata)))
        tmp_path.replace(path)

    def _read_metadata(self, asset_id: str) -> Optional[AssetMetadata]:
        try:
            data = json.loads(self._meta_path(asset_id).read_text())
        except (OSError, json.JSONDecodeError):
            return None
        return _parse_metadata(data)

    def _blob_path(self, asset_id: str) -> Path:
        return self.blobs_dir / asset_id

    def _meta_path(self, asset_id: str) -> Path:
        return self.meta_dir / f"{asset_id}.json"


def _is_valid_asset_id(asset_id: str) -> bool:
    return (
        asset_id.startswith((ASSET_ID_PREFIX, CACHED_VIDEO_ID_PREFIX))
        and len(asset_id) > 0
        and all(character in _ASSET_ID_CHARS for character in asset_id)
    )


def _parse_metadata(data: object) -> Optional[AssetMetadata]:
    if not isinstance(data, dict):
        return None
    try:
        return AssetMetadata(
            project_name=str(data["project_name"]),
            model_name=str(data["model_name"]),
            media_type=str(data["media_type"]),
            created_at=int(data["created_at"]),
            size_bytes=int(data.get("size_bytes", 0)),
            model_revision=data.get("model_revision"),
            created_by=data.get("created_by"),
            expires_at=int(data["expires_at"]) if data.get("expires_at") is not None else None,
        )
    except (KeyError, TypeError, ValueError):
        return None


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m dstack._internal.proxy.lib.services.generated_assets",
        description=(
            "Manage the generated-asset store. Retention defaults:"
            f" {PROXY_ASSETS_DEFAULT_RETENTION_DAYS} days,"
            f" {PROXY_ASSETS_DEFAULT_MAX_TOTAL_BYTES} total bytes."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    gc_parser = subparsers.add_parser("gc", help="Collect expired and over-budget assets")
    gc_parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete assets. Without this flag the collection is a dry run.",
    )
    backup_parser = subparsers.add_parser("backup", help="Write a backup archive")
    backup_parser.add_argument("archive", type=Path)
    restore_parser = subparsers.add_parser("restore", help="Restore from a backup archive")
    restore_parser.add_argument("archive", type=Path)
    restore_parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    store = GeneratedAssetStore()
    if args.command == "gc":
        plan = store.collect_garbage(dry_run=not args.apply)
        mode = "collected" if args.apply else "would collect (dry run)"
        print(f"{mode}: {len(plan.collected)} assets, {plan.collected_bytes} bytes")
        for entry in plan.collected:
            print(f"  {entry.asset_id}  {entry.reason}  {entry.size_bytes} bytes")
        print(f"retained: {plan.retained_count} assets, {plan.retained_bytes} bytes")
    elif args.command == "backup":
        count = store.export_backup(args.archive)
        print(f"backed up {count} assets to {args.archive}")
    elif args.command == "restore":
        count = store.restore_backup(args.archive, overwrite=args.overwrite)
        print(f"restored {count} assets from {args.archive}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
