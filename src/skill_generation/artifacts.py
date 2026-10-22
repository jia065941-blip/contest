"""Static bundle inspection: JSON, local regular files, and SHA-256 only."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Mapping

from .contracts import ModelManifest
from .errors import ArtifactError, ManifestError


@dataclass(frozen=True)
class CheckedBundle:
    root: Path
    manifest: ModelManifest
    artifacts: Mapping[str, Path]
    fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        object.__setattr__(self, "artifacts", MappingProxyType(dict(self.artifacts)))


def _no_links(path: Path) -> None:
    for item in (*reversed(path.parents), path):
        if item.is_symlink():
            raise ArtifactError("symlink_forbidden", "path", "no symbolic links", "symlink")


def _artifact_path(root: Path, relative: str) -> Path:
    parts = relative.split("/")
    if "\\" in relative or ":" in relative or PurePosixPath(relative).is_absolute() or any(p in ("", ".", "..") for p in parts):
        raise ArtifactError("unsafe_path", "artifacts.path", "canonical relative POSIX path", "unsafe path")
    path = root.joinpath(*parts)
    _no_links(path)
    if not path.resolve().is_relative_to(root):
        raise ArtifactError("path_escape", "artifacts.path", "path inside bundle", "outside bundle")
    return path


def _read_regular(path: Path, *, collect: bool, limit: int | None = None) -> tuple[str, bytes]:
    _no_links(path)
    try:
        # Resolve every directory with no-follow descriptors, not a check/open race.
        absolute = path.absolute()
        directory = os.open(absolute.anchor, os.O_RDONLY | os.O_DIRECTORY)
        try:
            for part in absolute.parts[1:-1]:
                next_directory = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
                os.close(directory)
                directory = next_directory
            fd = os.open(absolute.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        finally:
            os.close(directory)
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise ArtifactError("invalid_file", "path", "regular file without hard links", "invalid file")
            digest = hashlib.sha256()
            chunks = []
            total = 0
            while chunk := stream.read(1024 * 1024):
                total += len(chunk)
                if limit is not None and total > limit:
                    raise ArtifactError("file_too_large", "path", f"<= {limit} bytes", "size limit exceeded")
                digest.update(chunk)
                if collect:
                    chunks.append(chunk)
            after = os.fstat(stream.fileno())
            if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise ArtifactError("file_changed", "path", "stable file", "changed during read")
            return digest.hexdigest(), b"".join(chunks)
    except OSError as exc:
        raise ArtifactError("file_read_failed", "path", "readable regular file", type(exc).__name__) from exc


def read_artifact(bundle: CheckedBundle, name: str) -> bytes:
    """Recheck paths and digest over the exact bytes a backend will consume."""
    spec = next((a for a in bundle.manifest.artifacts if a.name == name), None)
    if spec is None or name not in bundle.artifacts:
        raise ArtifactError("missing_artifact", "artifacts", name, "missing")
    path = _artifact_path(bundle.root, spec.path)
    if path != bundle.artifacts[name]:
        raise ArtifactError("artifact_path_mismatch", "artifacts", "manifest path", "mismatch")
    digest, data = _read_regular(path, collect=True)
    if digest != spec.sha256:
        raise ArtifactError("hash_mismatch", f"artifacts.{name}.sha256", spec.sha256, digest)
    return data


def _object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ManifestError("duplicate_key", "manifest", "unique JSON keys", "duplicate")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise ManifestError("invalid_json_number", "manifest", "finite JSON number", "nonstandard constant")


def load_bundle(path: str | os.PathLike[str]) -> CheckedBundle:
    try:
        supplied = Path(os.path.abspath(os.fspath(path)))
        _no_links(supplied)
        manifest_path = supplied / "manifest.json" if supplied.is_dir() else supplied
        root = manifest_path.parent.resolve(strict=True)
        manifest_digest, raw = _read_regular(manifest_path, collect=True, limit=1024 * 1024)
        data = json.loads(raw.decode("utf-8"), object_pairs_hook=_object, parse_constant=_constant)
        manifest = ModelManifest.from_dict(data)
        artifacts = {}
        artifact_digests = []
        for spec in manifest.artifacts:
            target = _artifact_path(root, spec.path)
            if target == manifest_path:
                raise ArtifactError("manifest_as_artifact", "artifacts.path", "separate artifact", "manifest")
            digest, _ = _read_regular(target, collect=False)
            if digest != spec.sha256:
                raise ArtifactError("hash_mismatch", f"artifacts.{spec.name}.sha256", spec.sha256, digest)
            artifacts[spec.name] = target
            artifact_digests.append((spec.name, spec.path, digest))
        fingerprint = hashlib.sha256()
        fingerprint.update(b"skill-generation-bundle-v1\0")
        fingerprint.update(bytes.fromhex(manifest_digest))
        for name, relative, digest in artifact_digests:
            fingerprint.update(b"\0")
            fingerprint.update(name.encode("utf-8"))
            fingerprint.update(b"\0")
            fingerprint.update(relative.encode("utf-8"))
            fingerprint.update(b"\0")
            fingerprint.update(bytes.fromhex(digest))
        return CheckedBundle(root, manifest, artifacts, fingerprint.hexdigest())
    except (UnicodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise ManifestError("invalid_json", "manifest", "UTF-8 JSON object", type(exc).__name__) from exc
    except OSError as exc:
        raise ArtifactError("bundle_read_failed", "path", "readable bundle", type(exc).__name__) from exc
