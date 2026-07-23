import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List, TextIO

import yaml
from pydantic import ValidationError

from dstack._internal.cli.models.endpoint_presets import EndpointPreset
from dstack._internal.cli.models.endpoints import EndpointConfiguration
from dstack._internal.cli.services.endpoints.presets import endpoint_preset_to_data
from dstack._internal.core.errors import CLIError, ConfigurationError
from dstack._internal.utils.common import get_dstack_dir


class EndpointPresetStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or get_dstack_dir() / "presets"

    def list(self) -> list[EndpointPreset]:
        if not self.root.exists():
            return []
        presets = [self._load(path) for path in self.root.glob("models--*/*.yaml")]
        return sorted(presets, key=lambda preset: (preset.base.lower(), preset.id))

    def get(self, preset_id: str) -> EndpointPreset | None:
        paths = self._find_preset_paths(preset_id)
        if not paths:
            return None
        if len(paths) > 1:
            raise CLIError(f"Endpoint preset ID {preset_id!r} is not unique")
        path = paths[0]
        preset = self._load(path)
        if preset.id != preset_id:
            raise CLIError(f"Endpoint preset file {path} does not match its path")
        return preset

    def save(self, preset: EndpointPreset) -> Path:
        path = self._path(preset.base, preset.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        stored_preset, staged_assets = self._stage_assets(preset, path)
        content = yaml.safe_dump(endpoint_preset_to_data(stored_preset), sort_keys=False)
        fd, temporary_path = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{preset.id}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
            self._replace_assets(path, staged_assets)
            os.replace(temporary_path, path)
        finally:
            try:
                Path(temporary_path).unlink()
            except FileNotFoundError:
                pass
            if staged_assets is not None:
                shutil.rmtree(staged_assets, ignore_errors=True)
        return path

    def delete(self, preset_id: str) -> bool:
        preset = self.get(preset_id)
        if preset is None:
            return False
        path = self._path(preset.base, preset.id)
        path.unlink()
        shutil.rmtree(self._assets_path(path), ignore_errors=True)
        self._remove_empty_assets_directory(path)
        try:
            path.parent.rmdir()
        except OSError:
            pass
        return True

    def delete_for_base(self, base: str) -> int:
        directory = self._directory(base)
        paths = list(directory.glob("*.yaml"))
        presets = [self._load(path) for path in paths]
        if any(preset.base != base for preset in presets):
            raise CLIError(f"Endpoint preset directory {directory} contains another base model")
        for path in paths:
            path.unlink()
            shutil.rmtree(self._assets_path(path), ignore_errors=True)
        if paths:
            self._remove_empty_assets_directory(paths[0])
        try:
            directory.rmdir()
        except OSError:
            pass
        return len(presets)

    def _load(self, path: Path) -> EndpointPreset:
        try:
            with path.open(encoding="utf-8") as f:
                preset = EndpointPreset.parse_obj(yaml.safe_load(f))
            for mapping in preset.service.files:
                local_path = Path(mapping.local_path).expanduser()
                if not local_path.is_absolute():
                    mapping.local_path = str((path.parent / local_path).resolve())
            return preset
        except (OSError, ValidationError, yaml.YAMLError) as e:
            raise CLIError(f"Invalid endpoint preset file {path}: {e}") from e

    def _stage_assets(
        self,
        preset: EndpointPreset,
        path: Path,
    ) -> tuple[EndpointPreset, Path | None]:
        stored_preset = preset.copy(deep=True)
        if not stored_preset.service.files:
            return stored_preset, None
        staged_assets = Path(
            tempfile.mkdtemp(
                dir=path.parent,
                prefix=f".{preset.id}.assets.",
            )
        )
        try:
            for index, mapping in enumerate(stored_preset.service.files):
                source = Path(mapping.local_path).expanduser().resolve()
                if not source.exists():
                    raise CLIError(f"Endpoint preset file {mapping.local_path} does not exist")
                destination = staged_assets / f"{index}-{source.name}"
                if source.is_dir():
                    shutil.copytree(source, destination)
                else:
                    shutil.copy2(source, destination)
                mapping.local_path = str(Path("assets") / preset.id / destination.name)
        except Exception:
            shutil.rmtree(staged_assets, ignore_errors=True)
            raise
        return stored_preset, staged_assets

    def _replace_assets(self, path: Path, staged_assets: Path | None) -> None:
        assets_path = self._assets_path(path)
        if staged_assets is None:
            shutil.rmtree(assets_path, ignore_errors=True)
            self._remove_empty_assets_directory(path)
            return
        assets_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(assets_path, ignore_errors=True)
        os.replace(staged_assets, assets_path)

    @staticmethod
    def _assets_path(path: Path) -> Path:
        return path.parent / "assets" / path.stem

    @staticmethod
    def _remove_empty_assets_directory(path: Path) -> None:
        try:
            (path.parent / "assets").rmdir()
        except OSError:
            pass

    def _path(self, base: str, preset_id: str) -> Path:
        if not preset_id or any(char in preset_id for char in "/\\"):
            raise CLIError("Endpoint preset ID must not contain path separators")
        return self._directory(base) / f"{preset_id}.yaml"

    def _find_preset_paths(self, preset_id: str) -> List[Path]:
        if not preset_id or any(char in preset_id for char in "/\\"):
            raise CLIError("Endpoint preset ID must not contain path separators")
        return [
            path
            for directory in self.root.glob("models--*")
            if (path := directory / f"{preset_id}.yaml").is_file()
        ]

    def _directory(self, base: str) -> Path:
        directory = "models--" + base.replace("/", "--").replace("\\", "--")
        return self.root / directory


def load_endpoint_configuration(path: str) -> tuple[str, EndpointConfiguration]:
    if path == "-":
        return "-", _parse_endpoint_configuration(sys.stdin)
    configuration_path = Path(path)
    if not configuration_path.is_file():
        raise ConfigurationError(f"Configuration file {path} does not exist")
    try:
        with configuration_path.open(encoding="utf-8") as f:
            configuration = _parse_endpoint_configuration(f)
    except OSError as e:
        raise ConfigurationError(f"Failed to load configuration from {path}") from e
    return str(configuration_path.resolve()), configuration


def _parse_endpoint_configuration(stream: TextIO) -> EndpointConfiguration:
    try:
        data = yaml.safe_load(stream)
        if not isinstance(data, dict):
            raise ConfigurationError("Endpoint configuration must be a YAML object")
        return EndpointConfiguration.parse_obj(data)
    except ValidationError as e:
        raise ConfigurationError(e) from e
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Invalid endpoint configuration: {e}") from e
