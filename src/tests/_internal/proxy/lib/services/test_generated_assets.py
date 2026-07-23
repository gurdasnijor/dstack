import json
import time
from pathlib import Path

import pytest

from dstack._internal.proxy.lib.services.generated_assets import (
    ASSET_ID_PREFIX,
    GeneratedAssetStore,
    main,
)

PROJECT = "test-proj"
DAY = 86400


@pytest.fixture
def store(tmp_path: Path) -> GeneratedAssetStore:
    return GeneratedAssetStore(root=tmp_path)


def _store_asset(store: GeneratedAssetStore, content: bytes = b"png-bytes", **kwargs) -> str:
    defaults = dict(
        project_name=PROJECT,
        model_name="model/one",
        media_type="image/png",
        model_revision="abc123",
    )
    defaults.update(kwargs)
    return store.store(content=content, **defaults)


class TestStoreAndGet:
    def test_round_trip_preserves_metadata(self, store: GeneratedAssetStore):
        asset_id = _store_asset(store)
        asset = store.get(PROJECT, asset_id)
        assert asset is not None
        assert asset.content == b"png-bytes"
        assert asset.media_type == "image/png"
        assert asset.model_name == "model/one"
        assert asset.metadata.model_revision == "abc123"
        assert asset.metadata.size_bytes == len(b"png-bytes")
        assert asset.metadata.expires_at is not None
        assert asset.metadata.expires_at > asset.metadata.created_at

    def test_metadata_stored_separately_from_blob(
        self, store: GeneratedAssetStore, tmp_path: Path
    ):
        asset_id = _store_asset(store)
        assert (tmp_path / "blobs" / asset_id).is_file()
        assert (tmp_path / "meta" / f"{asset_id}.json").is_file()

    def test_get_enforces_project_ownership(self, store: GeneratedAssetStore):
        asset_id = _store_asset(store)
        assert store.get("other-project", asset_id) is None

    def test_get_rejects_invalid_ids(self, store: GeneratedAssetStore):
        assert store.get(PROJECT, "no-such-prefix") is None
        assert store.get(PROJECT, f"{ASSET_ID_PREFIX}../../etc/passwd") is None
        assert store.get(PROJECT, f"{ASSET_ID_PREFIX}missing") is None


class TestRetention:
    def test_expired_assets_are_collected_by_age(
        self, store: GeneratedAssetStore, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("DSTACK_PROXY_ASSETS_RETENTION_DAYS", "7")
        asset_id = _store_asset(store)
        now = int(time.time())

        plan = store.collect_garbage(dry_run=True, now=now + 6 * DAY)
        assert plan.collected == []

        plan = store.collect_garbage(now=now + 8 * DAY)
        assert [e.asset_id for e in plan.collected] == [asset_id]
        assert plan.collected[0].reason == "expired"
        assert store.get(PROJECT, asset_id) is None

    def test_zero_retention_days_disables_age_expiry(
        self, store: GeneratedAssetStore, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("DSTACK_PROXY_ASSETS_RETENTION_DAYS", "0")
        asset_id = _store_asset(store)
        plan = store.collect_garbage(now=int(time.time()) + 365 * DAY)
        assert plan.collected == []
        assert store.get(PROJECT, asset_id) is not None

    def test_size_budget_collects_oldest_first(
        self, store: GeneratedAssetStore, monkeypatch: pytest.MonkeyPatch
    ):
        oldest = _store_asset(store, content=b"x" * 400)
        self._age_asset(store, oldest, by_seconds=200)
        middle = _store_asset(store, content=b"y" * 400)
        self._age_asset(store, middle, by_seconds=100)
        newest = _store_asset(store, content=b"z" * 400)

        monkeypatch.setenv("DSTACK_PROXY_ASSETS_MAX_TOTAL_BYTES", "1000")
        plan = store.collect_garbage()
        assert [e.asset_id for e in plan.collected] == [oldest]
        assert plan.collected[0].reason == "size-budget"
        assert store.get(PROJECT, oldest) is None
        assert store.get(PROJECT, middle) is not None
        assert store.get(PROJECT, newest) is not None

    def test_newest_asset_survives_even_over_budget(
        self, store: GeneratedAssetStore, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("DSTACK_PROXY_ASSETS_MAX_TOTAL_BYTES", "10")
        asset_id = _store_asset(store, content=b"x" * 100)
        plan = store.collect_garbage()
        assert plan.collected == []
        assert store.get(PROJECT, asset_id) is not None

    def test_dry_run_reports_without_deleting(
        self, store: GeneratedAssetStore, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("DSTACK_PROXY_ASSETS_RETENTION_DAYS", "1")
        asset_id = _store_asset(store)
        plan = store.collect_garbage(dry_run=True, now=int(time.time()) + 2 * DAY)
        assert [e.asset_id for e in plan.collected] == [asset_id]
        assert plan.dry_run
        assert store.get(PROJECT, asset_id) is not None

    def test_write_triggers_collection(
        self, store: GeneratedAssetStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("DSTACK_PROXY_ASSETS_MAX_TOTAL_BYTES", "150")
        first = _store_asset(store, content=b"x" * 100)
        self._age_asset(store, first, by_seconds=100)
        _store_asset(store, content=b"y" * 100)
        assert store.get(PROJECT, first) is None

    @staticmethod
    def _age_asset(store: GeneratedAssetStore, asset_id: str, by_seconds: int) -> None:
        meta_path = store.meta_dir / f"{asset_id}.json"
        data = json.loads(meta_path.read_text())
        data["created_at"] -= by_seconds
        if data.get("expires_at") is not None:
            data["expires_at"] -= by_seconds
        meta_path.write_text(json.dumps(data))


class TestLegacyMigration:
    def test_legacy_flat_layout_is_migrated(self, tmp_path: Path):
        asset_id = f"{ASSET_ID_PREFIX}legacy00001"
        (tmp_path / f"{asset_id}.bin").write_bytes(b"legacy-bytes")
        (tmp_path / f"{asset_id}.json").write_text(
            json.dumps(
                {
                    "project_name": PROJECT,
                    "model_name": "model/legacy",
                    "media_type": "video/mp4",
                    "created_at": 1700000000,
                }
            )
        )
        store = GeneratedAssetStore(root=tmp_path)
        asset = store.get(PROJECT, asset_id)
        assert asset is not None
        assert asset.content == b"legacy-bytes"
        assert asset.media_type == "video/mp4"
        assert asset.metadata.size_bytes == len(b"legacy-bytes")
        assert not (tmp_path / f"{asset_id}.bin").exists()
        assert not (tmp_path / f"{asset_id}.json").exists()


class TestBackupRestore:
    def test_server_replacement_round_trip(self, tmp_path: Path):
        old_store = GeneratedAssetStore(root=tmp_path / "old-server")
        ids = [
            _store_asset(old_store, content=b"image-bytes"),
            _store_asset(
                old_store,
                content=b"video-bytes",
                media_type="video/mp4",
                model_name="model/video",
            ),
        ]
        archive = tmp_path / "assets-backup.tar.gz"
        assert old_store.export_backup(archive) == 2

        # A replacement server starts from an empty data directory.
        new_store = GeneratedAssetStore(root=tmp_path / "new-server")
        assert new_store.restore_backup(archive) == 2
        for asset_id in ids:
            original = old_store.get(PROJECT, asset_id)
            restored = new_store.get(PROJECT, asset_id)
            assert restored is not None and original is not None
            assert restored.content == original.content
            assert restored.metadata == original.metadata

    def test_restore_verifies_blob_integrity(self, tmp_path: Path):
        store = GeneratedAssetStore(root=tmp_path / "server")
        _store_asset(store)
        archive = tmp_path / "backup.tar.gz"
        store.export_backup(archive)

        # Corrupt the archived blob by rewriting the archive with bad bytes.
        import tarfile

        corrupted = tmp_path / "corrupted.tar.gz"
        with tarfile.open(archive, "r:gz") as src, tarfile.open(corrupted, "w:gz") as dst:
            for member in src.getmembers():
                extracted = src.extractfile(member)
                assert extracted is not None
                data = extracted.read()
                if member.name.startswith("blobs/"):
                    data = b"tampered!" + data
                    member.size = len(data)
                import io

                dst.addfile(member, io.BytesIO(data))

        target = GeneratedAssetStore(root=tmp_path / "target")
        with pytest.raises(ValueError, match="checksum mismatch"):
            target.restore_backup(corrupted)

    def test_restore_skips_existing_without_overwrite(self, tmp_path: Path):
        store = GeneratedAssetStore(root=tmp_path / "server")
        _store_asset(store)
        archive = tmp_path / "backup.tar.gz"
        store.export_backup(archive)
        assert store.restore_backup(archive) == 0
        assert store.restore_backup(archive, overwrite=True) == 1


class TestCli:
    def test_gc_dry_run_by_default(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ):
        monkeypatch.setenv("DSTACK_PROXY_ASSETS_DIR", str(tmp_path))
        monkeypatch.setenv("DSTACK_PROXY_ASSETS_RETENTION_DAYS", "0")
        store = GeneratedAssetStore()
        asset_id = _store_asset(store)

        assert main(["gc"]) == 0
        out = capsys.readouterr().out
        assert "dry run" in out
        assert store.get(PROJECT, asset_id) is not None

    def test_backup_and_restore_commands(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ):
        assets_dir = tmp_path / "assets"
        monkeypatch.setenv("DSTACK_PROXY_ASSETS_DIR", str(assets_dir))
        monkeypatch.setenv("DSTACK_PROXY_ASSETS_RETENTION_DAYS", "0")
        asset_id = _store_asset(GeneratedAssetStore())
        archive = tmp_path / "backup.tar.gz"

        assert main(["backup", str(archive)]) == 0
        assert "backed up 1 assets" in capsys.readouterr().out

        replacement_dir = tmp_path / "replacement"
        monkeypatch.setenv("DSTACK_PROXY_ASSETS_DIR", str(replacement_dir))
        assert main(["restore", str(archive)]) == 0
        assert "restored 1 assets" in capsys.readouterr().out
        assert GeneratedAssetStore().get(PROJECT, asset_id) is not None
