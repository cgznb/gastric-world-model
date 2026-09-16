"""Bounded, count-only release scanning for credentials and raw clinical assets."""

from __future__ import annotations

import math
import os
import re
import stat
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .errors import ConfigurationError

RELEASE_SCAN_SCHEMA_VERSION = "stageworld-release-scan-v1"
DEFAULT_CONTENT_LIMIT_BYTES = 8 * 1024 * 1024

_SCOPED_DIRECTORIES = {
    "artifacts": "artifact",
    "configs": "config",
    "constraints": "source",
    "docs": "documentation",
    "literature": "documentation",
    "prompts": "documentation",
    "scripts": "source",
    "specs": "documentation",
    "src": "source",
    "tests": "test_fixture",
    "third_party": "source",
}
_EXCLUDED_DIRECTORIES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
}
_PRIVATE_RUNTIME_ARTIFACT_PREFIX = "real"
_ROOT_DOCUMENT_SUFFIXES = {".md", ".rst"}
_ROOT_CONFIG_SUFFIXES = {".cfg", ".env", ".ini", ".toml", ".yaml", ".yml"}

_RAW_ASSET_SUFFIXES = (
    (".nii.gz", "raw_asset.nifti"),
    (".dicom", "raw_asset.dicom"),
    (".dcm", "raw_asset.dicom"),
    (".svs", "raw_asset.wsi"),
    (".ndpi", "raw_asset.wsi"),
    (".mrxs", "raw_asset.wsi"),
    (".scn", "raw_asset.wsi"),
    (".bif", "raw_asset.wsi"),
    (".vms", "raw_asset.wsi"),
    (".vmu", "raw_asset.wsi"),
    (".czi", "raw_asset.wsi"),
    (".nii", "raw_asset.nifti"),
    (".xlsx", "raw_asset.excel"),
    (".xlsm", "raw_asset.excel"),
    (".xlsb", "raw_asset.excel"),
    (".xls", "raw_asset.excel"),
)

_PROVIDER_CREDENTIAL_PATTERNS = (
    (
        "credential.private_key",
        re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----"),
    ),
    ("credential.aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    (
        "credential.github_token",
        re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b"),
    ),
    ("credential.huggingface_token", re.compile(r"\bhf_[A-Za-z0-9]{30,}\b")),
    ("credential.openai_key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b")),
    (
        "credential.slack_token",
        re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
    ),
    ("credential.google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    (
        "credential.jwt",
        re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
    ),
    (
        "credential.authenticated_url",
        re.compile(r"\b(?:https?|postgres(?:ql)?|mysql|mongodb(?:\+srv)?)://[^\s/:]+:[^\s/@]+@"),
    ),
)
_SECRET_KEY = (
    r"(?:[a-z0-9]+[_-])?api[_-]?key|aws[_-]?secret[_-]?access[_-]?key|"
    r"access[_-]?token|auth[_-]?token|bearer[_-]?token|refresh[_-]?token|"
    r"(?:hf|huggingface|github|gitlab|openai|anthropic|slack|wandb)[_-]?token|"
    r"client[_-]?secret|private[_-]?key|secret|token|password|passwd|credentials?"
)
_QUOTED_SECRET_ASSIGNMENT = re.compile(
    rf"(?i)[\"']?\b(?P<key>{_SECRET_KEY})\b[\"']?\s*[:=]\s*"
    rf"[\"'](?P<value>[^\"'\r\n]{{8,}})[\"']"
)
_UNQUOTED_SECRET_ASSIGNMENT = re.compile(
    rf"(?i)[\"']?\b(?P<key>{_SECRET_KEY})\b[\"']?\s*[:=]\s*"
    rf"(?P<value>[^\s,;#}}\]]{{12,}})"
)
_PLACEHOLDER_MARKERS = (
    "changeme",
    "dummy",
    "example",
    "fake",
    "not-a-real",
    "not_real",
    "not-used",
    "not_used",
    "placeholder",
    "redacted",
    "replace-me",
    "replace_me",
    "synthetic",
    "test-only",
    "test_only",
    "your-",
    "your_",
)

_POSIX_ABSOLUTE_PATH = re.compile(
    r"(?<![/:>}A-Za-z0-9_.-])/(?:[^/\s`'\"<>|]+/)*[^/\s`'\"<>|]+"
)
_WINDOWS_ABSOLUTE_PATH = re.compile(
    r"\b[A-Za-z]:\\(?:[^\\\s`'\"<>|]+\\)*[^\\\s`'\"<>|]+"
)
_SENSITIVE_PATH_COMPONENTS = {
    "clinical",
    "dicom",
    "imaging",
    "nifti",
    "pathology",
    "patient",
    "patients",
    "slides",
    "wsi",
    "dataset",
    "datasets",
    "临床",
    "患者",
    "影像",
    "数据",
    "病理",
}


@dataclass(frozen=True, slots=True)
class ReleaseScanReport:
    scanned_files: int
    scanned_bytes: int
    scope_file_counts: tuple[tuple[str, int], ...]
    finding_counts: tuple[tuple[str, int], ...]
    finding_scope_counts: tuple[tuple[str, int], ...]
    files_with_findings: int
    content_limit_bytes: int
    schema_version: str = RELEASE_SCAN_SCHEMA_VERSION

    @property
    def passed(self) -> bool:
        return not self.finding_counts

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": "ok" if self.passed else "blocked",
            "scanned_files": self.scanned_files,
            "scanned_bytes": self.scanned_bytes,
            "scope_file_counts": dict(self.scope_file_counts),
            "finding_count": sum(count for _, count in self.finding_counts),
            "finding_counts": dict(self.finding_counts),
            "finding_scope_counts": dict(self.finding_scope_counts),
            "files_with_findings": self.files_with_findings,
            "content_limit_bytes": self.content_limit_bytes,
            "scope_complete": not any(
                rule.startswith("scan.") for rule, _ in self.finding_counts
            ),
            "count_only": True,
            "matched_values_emitted": False,
            "file_paths_emitted": False,
            "external_data_roots_scanned": False,
        }


def _scope_for(relative_path: Path) -> str | None:
    parts = relative_path.parts
    if not parts:
        return None
    if len(parts) == 1:
        if relative_path.suffix.lower() in _ROOT_DOCUMENT_SUFFIXES:
            return "documentation"
        if relative_path.name.startswith(".env") or relative_path.suffix.lower() in (
            _ROOT_CONFIG_SUFFIXES
        ):
            return "config"
        return "source"
    return _SCOPED_DIRECTORIES.get(parts[0])


def _is_private_runtime_artifact_directory(relative_path: Path) -> bool:
    """Return whether a directory is a local real-data artifact namespace."""

    return (
        len(relative_path.parts) == 2
        and relative_path.parts[0] == "artifacts"
        and relative_path.parts[1].startswith(_PRIVATE_RUNTIME_ARTIFACT_PREFIX)
    )


def _raw_asset_rule(path: Path) -> str | None:
    lowered = path.name.lower()
    explicit = next(
        (rule for suffix, rule in _RAW_ASSET_SUFFIXES if lowered.endswith(suffix)), None
    )
    if explicit is not None:
        return explicit
    if lowered.endswith((".tif", ".tiff")):
        components = {
            re.sub(r"[^0-9a-z_-]", "", component.lower()) for component in path.parts[:-1]
        }
        if components & {"pathology", "slides", "wsi"}:
            return "raw_asset.wsi"
    return None


def _magic_asset_rule(data: bytes) -> str | None:
    if len(data) >= 132 and data[128:132] == b"DICM":
        return "raw_asset.dicom"
    if len(data) >= 348 and data[344:348] in {b"n+1\x00", b"ni1\x00"}:
        return "raw_asset.nifti"
    if len(data) >= 12 and data[4:12] in {b"n+2\x00\r\n\x1a\n", b"ni2\x00\r\n\x1a\n"}:
        return "raw_asset.nifti"
    return None


def _character_entropy(value: str) -> float:
    frequencies = Counter(value)
    length = len(value)
    return -sum((count / length) * math.log2(count / length) for count in frequencies.values())


def _looks_like_placeholder(value: str) -> bool:
    normalized = value.strip().strip("\"'").lower()
    if normalized in {"", "false", "none", "null", "true"}:
        return True
    if normalized.startswith(("$", "<", "{", "env(", "getenv(", "os.environ")):
        return True
    return any(marker in normalized for marker in _PLACEHOLDER_MARKERS)


def _has_high_confidence_assigned_secret(line: str) -> bool:
    matches = [
        *_QUOTED_SECRET_ASSIGNMENT.finditer(line),
        *_UNQUOTED_SECRET_ASSIGNMENT.finditer(line),
    ]
    for match in matches:
        value = match.group("value").strip().strip("\"'")
        if _looks_like_placeholder(value) or len(value) < 16:
            continue
        has_letter = any(character.isalpha() for character in value)
        if has_letter and _character_entropy(value) >= 3.0:
            return True
    return False


def _is_sensitive_absolute_path(candidate: str) -> bool:
    lowered = candidate.rstrip(".,;:)]}").lower()
    if lowered.startswith(("/path/to/", "/example/", "/your/")):
        return False
    components = [
        re.sub(r"[^0-9a-z_-]", "", component)
        for component in re.split(r"[/\\]+", lowered)
        if component
    ]
    if not components:
        return False
    if re.fullmatch(r"data\d*", components[0]):
        return True
    return any(component in _SENSITIVE_PATH_COMPONENTS for component in components)


def _content_findings(data: bytes) -> Counter[str]:
    findings: Counter[str] = Counter()
    text = data.decode("utf-8", errors="ignore")
    for line in text.splitlines():
        for rule, pattern in _PROVIDER_CREDENTIAL_PATTERNS:
            if pattern.search(line):
                findings[rule] += 1
        if _has_high_confidence_assigned_secret(line):
            findings["credential.assigned_secret"] += 1
        path_candidates = (
            *_POSIX_ABSOLUTE_PATH.findall(line),
            *_WINDOWS_ABSOLUTE_PATH.findall(line),
        )
        if any(_is_sensitive_absolute_path(candidate) for candidate in path_candidates):
            findings["sensitive_path.absolute_data"] += 1
    return findings


def scan_release_tree(
    root: str | Path,
    *,
    content_limit_bytes: int = DEFAULT_CONTENT_LIMIT_BYTES,
) -> ReleaseScanReport:
    """Scan fixed release scopes without following links or emitting matched data."""

    root_path = Path(root)
    if not root_path.is_dir() or root_path.is_symlink():
        raise ConfigurationError(
            code="INVALID_RELEASE_SCAN_ROOT",
            message="Release scan root must be a real local directory, not a symlink.",
        )
    if content_limit_bytes < 512:
        raise ConfigurationError(
            code="INVALID_RELEASE_SCAN_LIMIT",
            message="Release scan content limit must be at least 512 bytes.",
        )

    scope_counts: Counter[str] = Counter()
    finding_counts: Counter[str] = Counter()
    finding_scope_counts: Counter[str] = Counter()
    finding_files: set[Path] = set()
    scanned_files = 0
    scanned_bytes = 0
    def record(relative_path: Path, scope: str, rule: str, count: int = 1) -> None:
        finding_counts[rule] += count
        finding_scope_counts[scope] += count
        finding_files.add(relative_path)

    def record_walk_error(error: OSError) -> None:
        reported_path = root_path
        if error.filename is not None:
            reported_path = Path(os.fsdecode(error.filename))
        try:
            relative = reported_path.relative_to(root_path)
        except ValueError:
            relative = Path(".")
        scope = _scope_for(relative) or "unscoped"
        record(relative, scope, "scan.unreadable_directory")

    for current_root, directory_names, file_names in os.walk(
        root_path,
        followlinks=False,
        onerror=record_walk_error,
    ):
        current = Path(current_root)
        relative_directory = current.relative_to(root_path)
        retained_directories: list[str] = []
        for name in sorted(directory_names):
            candidate = current / name
            relative = candidate.relative_to(root_path)
            if name in _EXCLUDED_DIRECTORIES or name.endswith(".egg-info"):
                continue
            if _is_private_runtime_artifact_directory(relative):
                continue
            if relative_directory == Path(".") and name not in _SCOPED_DIRECTORIES:
                record(relative, "unscoped", "scan.unscoped_directory")
                continue
            if candidate.is_symlink():
                scope = _scope_for(relative)
                if scope is not None:
                    record(relative, scope, "scan.symlink_not_followed")
                continue
            retained_directories.append(name)
        directory_names[:] = retained_directories

        for name in sorted(file_names):
            path = current / name
            relative = path.relative_to(root_path)
            scope = _scope_for(relative)
            if scope is None:
                continue
            scanned_files += 1
            scope_counts[scope] += 1
            try:
                metadata = path.lstat()
            except OSError:
                record(relative, scope, "scan.unreadable_file")
                continue
            scanned_bytes += int(metadata.st_size)
            if stat.S_ISLNK(metadata.st_mode):
                record(relative, scope, "scan.symlink_not_followed")
                continue
            if not stat.S_ISREG(metadata.st_mode):
                record(relative, scope, "scan.nonregular_file")
                continue

            suffix_rule = _raw_asset_rule(path)
            if suffix_rule is not None:
                record(relative, scope, suffix_rule)
                continue
            try:
                if metadata.st_size > content_limit_bytes:
                    with path.open("rb") as handle:
                        header = handle.read(560)
                    magic_rule = _magic_asset_rule(header)
                    record(
                        relative,
                        scope,
                        "scan.content_limit_exceeded" if magic_rule is None else magic_rule,
                    )
                    continue
                data = path.read_bytes()
            except OSError:
                record(relative, scope, "scan.unreadable_file")
                continue
            magic_rule = _magic_asset_rule(data[:560])
            if magic_rule is not None:
                record(relative, scope, magic_rule)
                continue
            for rule, count in _content_findings(data).items():
                record(relative, scope, rule, count)

    return ReleaseScanReport(
        scanned_files=scanned_files,
        scanned_bytes=scanned_bytes,
        scope_file_counts=tuple(sorted(scope_counts.items())),
        finding_counts=tuple(sorted(finding_counts.items())),
        finding_scope_counts=tuple(sorted(finding_scope_counts.items())),
        files_with_findings=len(finding_files),
        content_limit_bytes=content_limit_bytes,
    )
