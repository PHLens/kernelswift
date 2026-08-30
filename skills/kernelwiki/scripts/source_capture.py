from __future__ import annotations

import argparse
import base64
import binascii
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime
from enum import Enum
import fnmatch
import hashlib
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
from typing import Any, Protocol
import urllib.error
import urllib.parse
import urllib.request

import yaml

from provenance import FILE_MODES, FILE_ROLES
from kernelwiki_common import (
    KernelWikiError,
    canonical_json_bytes,
    load_yaml_document,
    parse_huawei_cann_document_revision,
    require_within,
    run_cli,
    sha256_bytes,
    sha256_file,
)


REPOSITORY_FIELDS = frozenset(
    {
        "id",
        "host",
        "repo",
        "lane",
        "languages",
        "target_families",
        "kernel_path_globs",
        "skip_path_globs",
        "search_terms",
    }
)
LEDGER_FIELDS = frozenset(
    {
        "schema_version",
        "repository_id",
        "repo",
        "searched_at",
        "keywords_used",
        "total_candidates",
        "included",
        "deferred",
        "excluded",
        "prs",
    }
)
CANDIDATE_FIELDS = frozenset(
    {
        "number",
        "title",
        "date",
        "url",
        "decision",
        "reason",
        "kernel_types",
        "techniques",
        "languages",
        "changed_paths",
        "files_reviewed_count",
        "target_evidence",
        "discovery_state",
    }
)
CAPTURE_METADATA_FIELDS = frozenset(
    {
        "source_id",
        "title",
        "repository_id",
        "captured_at",
        "target_disposition",
        "languages",
        "kernel_types",
        "techniques",
        "hardware_features",
        "tags",
        "license_state",
        "audiences",
    }
)
CAPTURE_SELECTION_FIELDS = frozenset(
    {"upstream_path", "heading_path", "role", "mode"}
)
GITHUB_CAPTURE_MANIFEST_FIELDS = frozenset(
    {"schema_version", "metadata", "selections"}
)
MANUAL_CAPTURE_MANIFEST_FIELDS = frozenset(
    {"schema_version", "metadata", "source_kind", "url", "document_revision", "files"}
)
MANUAL_FILE_FIELDS = frozenset(
    {"input_path", "upstream_path", "heading_path", "role", "mode"}
)
DECISIONS = frozenset({"include", "defer", "exclude"})
DISCOVERY_STATES = frozenset({"reviewed", "returned", "not-returned"})
APPROVED_LICENSE = "approved"
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SOURCE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
REPOSITORY_ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
UTC_ISO8601_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$"
)
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
NEXT_LINK_RE = re.compile(r'<([^>]+)>;\s*rel="next"')
TAXONOMY_PATH = Path(__file__).resolve().parents[1] / "data" / "taxonomy.yaml"

APPROVED_REPOSITORY_CONFIGS: Mapping[str, Mapping[str, Any]] = {
    "triton-ascend": {
        "host": "github",
        "repo": "Ascend/triton-ascend",
        "lane": "ascend-native",
        "languages": ("triton", "python", "cpp"),
        "target_families": ("ascend",),
    },
    "vllm-ascend": {
        "host": "github",
        "repo": "vllm-project/vllm-ascend",
        "lane": "ascend-native",
        "languages": ("ascendc", "cpp", "python"),
        "target_families": ("ascend",),
    },
    "cann-samples": {
        "host": "manual",
        "repo": "Ascend/cann-samples",
        "lane": "ascend-native-manual",
        "languages": ("ascendc", "cpp"),
        "target_families": ("ascend",),
    },
    "huawei-ascend-docs": {
        "host": "manual",
        "repo": "hiascend.com/CANNCommunityEdition",
        "lane": "ascend-native-manual",
        "languages": ("ascendc", "cpp"),
        "target_families": ("ascend",),
    },
    "triton-ascend-kernels": {
        "host": "manual",
        "repo": "Ascend/triton-ascend-kernels",
        "lane": "reviewed-holdout",
        "languages": ("triton", "python"),
        "target_families": ("ascend",),
    },
    "mskl": {
        "host": "manual",
        "repo": "Ascend/mskl",
        "lane": "ascend-native-manual",
        "languages": ("python", "cpp"),
        "target_families": ("ascend",),
    },
}
APPROVED_KERNEL_GLOBS = ("**/ops/**", "**/kernels/**", "**/*ascendc*")
APPROVED_SKIP_GLOBS = ("**/docs/**", "**/tests/**", "**/benchmarks/**")
APPROVED_SEARCH_TERMS = (
    "AscendC",
    "kernel",
    "performance",
    "fused",
    "topk",
    "attention",
    "reduction",
    "tiling",
    "double buffer",
)


@dataclass(frozen=True)
class RepositorySpec:
    repository_id: str
    host: str
    repo: str
    lane: str
    languages: tuple[str, ...]
    target_families: tuple[str, ...]
    kernel_path_globs: tuple[str, ...]
    skip_path_globs: tuple[str, ...]
    search_terms: tuple[str, ...]


@dataclass(frozen=True)
class Candidate:
    number: int
    title: str = ""
    date: str | None = None
    url: str = ""
    decision: str = "defer"
    reason: str = "unreviewed-discovery"
    kernel_types: tuple[str, ...] = ()
    techniques: tuple[str, ...] = ()
    languages: tuple[str, ...] = ()
    changed_paths: tuple[str, ...] = ()
    files_reviewed_count: int = 0
    target_evidence: tuple[str, ...] = ()
    discovery_state: str = "returned"


@dataclass(frozen=True)
class CandidateLedger:
    schema_version: int
    repository_id: str
    repo: str
    searched_at: str
    keywords_used: tuple[str, ...]
    total_candidates: int
    included: int
    deferred: int
    excluded: int
    prs: tuple[Candidate, ...]

    @property
    def by_number(self) -> Mapping[int, Candidate]:
        return {candidate.number: candidate for candidate in self.prs}


@dataclass(frozen=True)
class SourceCaptureMetadata:
    source_id: str
    title: str
    repository_id: str
    captured_at: str
    target_disposition: str
    languages: tuple[str, ...]
    kernel_types: tuple[str, ...]
    techniques: tuple[str, ...]
    hardware_features: tuple[str, ...]
    tags: tuple[str, ...]
    license_state: str
    audiences: tuple[str, ...]


@dataclass(frozen=True)
class CaptureSelection:
    upstream_path: str
    heading_path: str | None
    role: str
    mode: str


@dataclass(frozen=True)
class GitHubPRCaptureRequest:
    skill_root: Path
    metadata: SourceCaptureMetadata
    repo: str
    number: int
    selections: tuple[CaptureSelection, ...]


@dataclass(frozen=True)
class GitHubCommitCaptureRequest:
    skill_root: Path
    metadata: SourceCaptureMetadata
    repo: str
    sha: str
    selections: tuple[CaptureSelection, ...]


@dataclass(frozen=True)
class ManualFileSelection:
    input_path: Path
    upstream_path: str | None
    heading_path: str | None
    role: str
    mode: str


@dataclass(frozen=True)
class ManualCaptureManifest:
    schema_version: int
    metadata: SourceCaptureMetadata
    source_kind: str
    url: str
    document_revision: str
    files: tuple[ManualFileSelection, ...]


@dataclass(frozen=True)
class ManualCaptureRequest:
    skill_root: Path
    manifest_path: Path


class CaptureState(str, Enum):
    PREPARED = "prepared"
    ARTIFACT_PUBLISHED = "artifact-published"
    SOURCE_PUBLISHED = "source-published"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled-back"


@dataclass(frozen=True)
class CaptureTransaction:
    schema_version: int
    transaction_id: str
    source_id: str
    state: CaptureState
    staged_source: str
    staged_artifact: str | None
    final_source: str
    final_artifact: str | None
    owns_final_source: bool
    owns_final_artifact: bool
    source_sha256: str
    provenance_sha256: str | None


@dataclass(frozen=True)
class CaptureRecovery:
    transaction_id: str
    previous_state: CaptureState
    final_state: CaptureState
    actions: tuple[str, ...]


@dataclass(frozen=True)
class CaptureResult:
    source_id: str
    source_path: Path
    artifact_dir: Path | None
    captured_files: tuple[Path, ...]


class CaptureGitHubClient(Protocol):
    def get_paginated(self, url: str) -> list[Any]: ...

    def get_json(self, url: str) -> Any: ...

    def get_file_bytes(self, repo: str, path: str, ref: str) -> bytes: ...

    def get_patch_bytes(self, url: str) -> bytes: ...


class PaginatedJsonClient(Protocol):
    def get_paginated(self, url: str) -> list[Any]: ...


class StableArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise KernelWikiError("cli-input-invalid", message)


def validate_capture_skill_root(value: Path) -> Path:
    if not isinstance(value, Path):
        raise KernelWikiError("capture-path-invalid", "skill_root must be a pathlib.Path")
    try:
        metadata = os.lstat(value)
        resolved = value.resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise KernelWikiError(
            "capture-path-invalid", "skill_root must be an existing real directory", value
        ) from error
    if stat.S_ISLNK(metadata.st_mode) or not resolved.is_dir():
        raise KernelWikiError(
            "capture-path-invalid", "skill_root must be a real non-symlink directory", value
        )
    return resolved


def _git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def _require_mapping(value: Any, code: str, path: Path | None = None) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise KernelWikiError(code, "expected a mapping", path)
    return value


def _require_exact_fields(
    value: Mapping[str, Any], fields: frozenset[str], code: str, path: Path | None
) -> None:
    if set(value) != set(fields):
        missing = sorted(fields - set(value))
        unknown = sorted(set(value) - fields)
        raise KernelWikiError(code, f"missing={missing}, unknown={unknown}", path)


def _require_string(value: Any, code: str, field: str, path: Path | None) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise KernelWikiError(code, f"{field} must be a nonblank string", path)
    return value


def _string_tuple(
    value: Any,
    code: str,
    path: Path | None,
    *,
    field: str,
    sorted_unique: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(item, str) and item and item.strip() == item for item in value
    ):
        raise KernelWikiError(code, f"{field} must be a string list", path)
    result = tuple(value)
    if sorted_unique and result != tuple(sorted(set(result))):
        raise KernelWikiError(code, f"{field} must be sorted and unique", path)
    return result


def _require_utc_iso8601(
    value: Any, code: str, field: str, path: Path | None
) -> str:
    text = _require_string(value, code, field, path)
    if UTC_ISO8601_RE.fullmatch(text) is None:
        raise KernelWikiError(code, f"{field} must be UTC ISO-8601", path)
    try:
        datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as error:
        raise KernelWikiError(code, f"{field} must be UTC ISO-8601", path) from error
    return text


def _require_date(value: Any, code: str, field: str, path: Path | None) -> str:
    text = _require_string(value, code, field, path)
    if DATE_RE.fullmatch(text) is None:
        raise KernelWikiError(code, f"{field} must be YYYY-MM-DD", path)
    try:
        date.fromisoformat(text)
    except ValueError as error:
        raise KernelWikiError(code, f"{field} must be YYYY-MM-DD", path) from error
    return text


def _require_capture_not_before(
    captured_at: str, events: Sequence[str], *, authority: str
) -> None:
    captured = datetime.fromisoformat(captured_at[:-1] + "+00:00")
    latest = max(datetime.fromisoformat(event[:-1] + "+00:00") for event in events)
    if captured < latest:
        raise KernelWikiError(
            "capture-timestamp-invalid", f"captured_at predates {authority} evidence"
        )


def validate_repository_relative_posix_path(
    value: Any, *, code: str, field: str, path: Path | None = None
) -> str:
    text = _require_string(value, code, field, path)
    pure = PurePosixPath(text)
    if (
        pure.is_absolute()
        or "\\" in text
        or "\0" in text
        or text.endswith("/")
        or any(part in {"", ".", ".."} for part in text.split("/"))
        or pure.as_posix() != text
    ):
        raise KernelWikiError(code, f"{field} must be a normalized relative POSIX path", path)
    return text


def _taxonomy_values() -> Mapping[str, frozenset[str]]:
    document = _require_mapping(load_yaml_document(TAXONOMY_PATH), "taxonomy-invalid", TAXONOMY_PATH)
    return {
        "languages": frozenset(document["languages"]),
        "kernel_types": frozenset(document["kernel_types"]),
        "techniques": frozenset(document["techniques"]),
        "hardware_features": frozenset(document["hardware_features"]),
        "tags": frozenset(document["tags"]),
        "audiences": frozenset(document["audiences"]),
        "target_matches": frozenset(document["target_matches"]),
        "license_states": frozenset(document["license_states"]),
    }


def _require_taxonomy_subset(
    values: Sequence[str], *, allowed: frozenset[str], code: str, field: str, path: Path
) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise KernelWikiError(code, f"{field} contains unknown values {unknown}", path)


def _require_github_api_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.netloc != "api.github.com":
        raise KernelWikiError("github-url-invalid", f"refusing non-GitHub API URL {url}")
    return url


class GitHubApiRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        resolved = urllib.parse.urljoin(request.full_url, newurl)
        _require_github_api_url(resolved)
        return super().redirect_request(request, fp, code, message, headers, resolved)


class GitHubPatchRedirectHandler(urllib.request.HTTPRedirectHandler):
    ALLOWED_HOSTS = frozenset({"github.com", "patch-diff.githubusercontent.com"})

    def redirect_request(self, request, fp, code, message, headers, newurl):
        resolved = urllib.parse.urljoin(request.full_url, newurl)
        parsed = urllib.parse.urlparse(resolved)
        if parsed.scheme != "https" or parsed.netloc not in self.ALLOWED_HOSTS:
            raise KernelWikiError("github-url-invalid", f"refusing patch redirect {resolved}")
        return super().redirect_request(request, fp, code, message, headers, resolved)


class GitHubClient:
    def __init__(
        self,
        *,
        token: str | None = None,
        opener: Any | None = None,
        patch_opener: Any | None = None,
        timeout: int = 30,
    ):
        self.token = os.environ.get("GITHUB_TOKEN") if token is None else token
        self.opener = opener or urllib.request.build_opener(GitHubApiRedirectHandler())
        self.patch_opener = patch_opener or urllib.request.build_opener(
            GitHubPatchRedirectHandler()
        )
        self.timeout = timeout

    def _request_json(self, url: str) -> tuple[Any, str | None]:
        _require_github_api_url(url)
        headers = {"Accept": "application/vnd.github+json", "User-Agent": "KernelWiki/1"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        try:
            with self.opener.open(
                urllib.request.Request(url, headers=headers), timeout=self.timeout
            ) as response:
                raw = response.read()
                link = response.headers.get("Link")
        except urllib.error.HTTPError as error:
            remaining = error.headers.get("X-RateLimit-Remaining") if error.headers else None
            reset = error.headers.get("X-RateLimit-Reset") if error.headers else None
            if error.code == 429 or (error.code == 403 and remaining == "0"):
                suffix = f"; reset={reset}" if reset else ""
                raise KernelWikiError(
                    "github-rate-limit", f"GitHub API rate limit reached{suffix}"
                ) from error
            raise KernelWikiError(
                "github-http-error", f"GitHub API returned HTTP {error.code}"
            ) from error
        except urllib.error.URLError as error:
            raise KernelWikiError("github-network", str(error.reason)) from error
        except (http.client.HTTPException, TimeoutError, OSError, ValueError) as error:
            raise KernelWikiError("github-network", f"GitHub transport failed: {error}") from error
        try:
            return json.loads(raw.decode("utf-8")), link
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise KernelWikiError("github-json-invalid", f"invalid GitHub JSON from {url}") from error

    def get_json(self, url: str) -> Any:
        return self._request_json(url)[0]

    def get_paginated(self, url: str) -> list[Any]:
        items: list[Any] = []
        next_url: str | None = url
        seen: set[str] = set()
        while next_url is not None:
            if next_url in seen:
                raise KernelWikiError("github-pagination-cycle", f"pagination repeated {next_url}")
            seen.add(next_url)
            payload, link = self._request_json(next_url)
            page = payload.get("items") if isinstance(payload, Mapping) else payload
            if not isinstance(page, list):
                raise KernelWikiError("github-json-invalid", "GitHub page must contain a list")
            items.extend(page)
            match = NEXT_LINK_RE.search(link or "")
            next_url = match.group(1) if match else None
            if next_url is not None:
                _require_github_api_url(next_url)
        return items

    def get_file_bytes(self, repo: str, path: str, ref: str) -> bytes:
        if REPO_RE.fullmatch(repo) is None or GIT_SHA_RE.fullmatch(ref) is None:
            raise KernelWikiError("github-content-invalid", "invalid repository or Git ref")
        path = validate_repository_relative_posix_path(
            path, code="github-content-invalid", field="content path"
        )
        encoded = "/".join(urllib.parse.quote(part, safe="") for part in path.split("/"))
        payload = self.get_json(
            f"https://api.github.com/repos/{urllib.parse.quote(repo, safe='/')}/contents/{encoded}?ref={ref}"
        )
        if not isinstance(payload, Mapping) or payload.get("encoding") != "base64":
            raise KernelWikiError("github-json-invalid", "content response requires base64 content")
        encoded_content = payload.get("content")
        if not isinstance(encoded_content, str):
            raise KernelWikiError("github-json-invalid", "content response requires base64 content")
        compact = encoded_content.replace("\n", "")
        try:
            data = base64.b64decode(compact, validate=True)
        except (binascii.Error, ValueError, TypeError) as error:
            raise KernelWikiError("github-json-invalid", "content response has invalid base64") from error
        if _git_blob_sha(data) != payload.get("sha"):
            raise KernelWikiError(
                "github-content-hash-mismatch", "content bytes do not match Git blob SHA"
            )
        return data

    def get_patch_bytes(self, url: str) -> bytes:
        parsed = urllib.parse.urlparse(url)
        if (
            parsed.scheme != "https"
            or parsed.netloc != "github.com"
            or not parsed.path.endswith(".patch")
            or parsed.query
            or parsed.fragment
        ):
            raise KernelWikiError("github-url-invalid", f"invalid GitHub patch URL {url}")
        try:
            with self.patch_opener.open(
                urllib.request.Request(
                    url, headers={"Accept": "text/x-patch", "User-Agent": "KernelWiki/1"}
                ),
                timeout=self.timeout,
            ) as response:
                data = response.read()
        except urllib.error.HTTPError as error:
            raise KernelWikiError(
                "github-http-error", f"GitHub patch returned HTTP {error.code}"
            ) from error
        except urllib.error.URLError as error:
            raise KernelWikiError("github-network", str(error.reason)) from error
        except (http.client.HTTPException, TimeoutError, OSError, ValueError) as error:
            raise KernelWikiError(
                "github-network", f"GitHub patch transport failed: {error}"
            ) from error
        if not isinstance(data, bytes) or not data:
            raise KernelWikiError(
                "github-response-invalid", "GitHub patch response must be nonempty bytes"
            )
        return data


def load_source_registry(path: Path) -> Mapping[str, RepositorySpec]:
    document = _require_mapping(load_yaml_document(path), "repository-registry-invalid", path)
    if set(document) != {"schema_version", "repositories"} or document.get("schema_version") != 1:
        raise KernelWikiError("repository-registry-invalid", "registry schema is invalid", path)
    raw_repositories = document.get("repositories")
    if not isinstance(raw_repositories, list):
        raise KernelWikiError("repository-registry-invalid", "repositories must be a list", path)
    result: dict[str, RepositorySpec] = {}
    for raw in raw_repositories:
        row = _require_mapping(raw, "repository-spec-invalid", path)
        _require_exact_fields(row, REPOSITORY_FIELDS, "repository-spec-invalid", path)
        repository_id = _require_string(row["id"], "repository-spec-invalid", "id", path)
        expected = APPROVED_REPOSITORY_CONFIGS.get(repository_id)
        if expected is None or repository_id in result:
            raise KernelWikiError("repository-spec-invalid", "repository is not approved or is duplicate", path)
        spec = RepositorySpec(
            repository_id,
            _require_string(row["host"], "repository-spec-invalid", "host", path),
            _require_string(row["repo"], "repository-spec-invalid", "repo", path),
            _require_string(row["lane"], "repository-spec-invalid", "lane", path),
            _string_tuple(row["languages"], "repository-spec-invalid", path, field="languages"),
            _string_tuple(
                row["target_families"], "repository-spec-invalid", path, field="target_families"
            ),
            _string_tuple(
                row["kernel_path_globs"], "repository-spec-invalid", path, field="kernel_path_globs"
            ),
            _string_tuple(
                row["skip_path_globs"], "repository-spec-invalid", path, field="skip_path_globs"
            ),
            _string_tuple(
                row["search_terms"], "repository-spec-invalid", path, field="search_terms"
            ),
        )
        actual = {
            "host": spec.host,
            "repo": spec.repo,
            "lane": spec.lane,
            "languages": spec.languages,
            "target_families": spec.target_families,
        }
        if actual != expected or spec.kernel_path_globs != APPROVED_KERNEL_GLOBS or spec.skip_path_globs != APPROVED_SKIP_GLOBS or spec.search_terms != APPROVED_SEARCH_TERMS:
            raise KernelWikiError("repository-spec-invalid", "repository differs from reviewed configuration", path)
        result[repository_id] = spec
    if set(result) != set(APPROVED_REPOSITORY_CONFIGS):
        raise KernelWikiError("repository-registry-invalid", "registry does not contain the approved set", path)
    return result


def _candidate_from_mapping(raw: Any, path: Path) -> Candidate:
    row = _require_mapping(raw, "candidate-invalid", path)
    _require_exact_fields(row, CANDIDATE_FIELDS, "candidate-invalid", path)
    number = row["number"]
    if type(number) is not int or number <= 0:
        raise KernelWikiError("candidate-invalid", "number must be positive", path)
    changed_paths = _string_tuple(
        row["changed_paths"], "candidate-invalid", path, field="changed_paths", sorted_unique=True
    )
    for changed_path in changed_paths:
        validate_repository_relative_posix_path(
            changed_path, code="candidate-invalid", field="changed_path", path=path
        )
    decision = _require_string(row["decision"], "candidate-invalid", "decision", path)
    state = _require_string(row["discovery_state"], "candidate-invalid", "discovery_state", path)
    if decision not in DECISIONS or state not in DISCOVERY_STATES:
        raise KernelWikiError("candidate-invalid", "candidate enum value is invalid", path)
    candidate_date = row["date"]
    if candidate_date is not None:
        candidate_date = _require_date(candidate_date, "candidate-invalid", "date", path)
    files_reviewed = row["files_reviewed_count"]
    if type(files_reviewed) is not int or files_reviewed < 0 or files_reviewed != len(changed_paths):
        raise KernelWikiError("candidate-invalid", "files_reviewed_count is invalid", path)
    return Candidate(
        number=number,
        title=_require_string(row["title"], "candidate-invalid", "title", path),
        date=candidate_date,
        url=_require_string(row["url"], "candidate-invalid", "url", path),
        decision=decision,
        reason=_require_string(row["reason"], "candidate-invalid", "reason", path),
        kernel_types=_string_tuple(row["kernel_types"], "candidate-invalid", path, field="kernel_types", sorted_unique=True),
        techniques=_string_tuple(row["techniques"], "candidate-invalid", path, field="techniques", sorted_unique=True),
        languages=_string_tuple(row["languages"], "candidate-invalid", path, field="languages", sorted_unique=True),
        changed_paths=changed_paths,
        files_reviewed_count=files_reviewed,
        target_evidence=_string_tuple(row["target_evidence"], "candidate-invalid", path, field="target_evidence", sorted_unique=True),
        discovery_state=state,
    )


def load_candidate_ledger(
    path: Path, *, repository: RepositorySpec | None = None
) -> CandidateLedger:
    document = _require_mapping(load_yaml_document(path), "candidate-ledger-invalid", path)
    _require_exact_fields(document, LEDGER_FIELDS, "candidate-ledger-invalid", path)
    if document.get("schema_version") != 1 or type(document.get("schema_version")) is not int:
        raise KernelWikiError("candidate-ledger-invalid", "schema_version must be integer 1", path)
    prs_raw = document["prs"]
    if not isinstance(prs_raw, list):
        raise KernelWikiError("candidate-ledger-invalid", "prs must be a list", path)
    prs = tuple(_candidate_from_mapping(item, path) for item in prs_raw)
    numbers = [item.number for item in prs]
    if numbers != sorted(set(numbers)):
        raise KernelWikiError("candidate-ledger-invalid", "PR numbers must be sorted and unique", path)
    counts = {}
    for field in ("total_candidates", "included", "deferred", "excluded"):
        value = document[field]
        if type(value) is not int or value < 0:
            raise KernelWikiError("candidate-ledger-invalid", f"{field} must be nonnegative", path)
        counts[field] = value
    expected_counts = {
        "total_candidates": len(prs),
        "included": sum(item.decision == "include" for item in prs),
        "deferred": sum(item.decision == "defer" for item in prs),
        "excluded": sum(item.decision == "exclude" for item in prs),
    }
    if counts != expected_counts:
        raise KernelWikiError("candidate-ledger-invalid", "ledger counts do not match candidates", path)
    ledger = CandidateLedger(
        1,
        _require_string(document["repository_id"], "candidate-ledger-invalid", "repository_id", path),
        _require_string(document["repo"], "candidate-ledger-invalid", "repo", path),
        _require_utc_iso8601(document["searched_at"], "candidate-ledger-invalid", "searched_at", path),
        _string_tuple(document["keywords_used"], "candidate-ledger-invalid", path, field="keywords_used"),
        counts["total_candidates"],
        counts["included"],
        counts["deferred"],
        counts["excluded"],
        prs,
    )
    if repository is not None and (
        ledger.repository_id != repository.repository_id
        or ledger.repo != repository.repo
        or ledger.keywords_used != repository.search_terms
    ):
        raise KernelWikiError("candidate-ledger-invalid", "ledger repository identity differs", path)
    return ledger


def _candidate_mapping(candidate: Candidate) -> dict[str, Any]:
    return {
        "number": candidate.number,
        "title": candidate.title,
        "date": candidate.date,
        "url": candidate.url,
        "decision": candidate.decision,
        "reason": candidate.reason,
        "kernel_types": list(candidate.kernel_types),
        "techniques": list(candidate.techniques),
        "languages": list(candidate.languages),
        "changed_paths": list(candidate.changed_paths),
        "files_reviewed_count": candidate.files_reviewed_count,
        "target_evidence": list(candidate.target_evidence),
        "discovery_state": candidate.discovery_state,
    }


def render_candidate_ledger(ledger: CandidateLedger) -> str:
    document = {
        "schema_version": 1,
        "repository_id": ledger.repository_id,
        "repo": ledger.repo,
        "searched_at": ledger.searched_at,
        "keywords_used": list(ledger.keywords_used),
        "total_candidates": ledger.total_candidates,
        "included": ledger.included,
        "deferred": ledger.deferred,
        "excluded": ledger.excluded,
        "prs": [_candidate_mapping(item) for item in sorted(ledger.prs, key=lambda item: item.number)],
    }
    return yaml.safe_dump(document, sort_keys=False, allow_unicode=True)


def merge_discovery(
    existing: CandidateLedger, discovered: Sequence[Candidate], *, searched_at: str
) -> CandidateLedger:
    _require_utc_iso8601(searched_at, "candidate-ledger-invalid", "searched_at", None)
    incoming = {item.number: item for item in discovered}
    if len(incoming) != len(tuple(discovered)):
        raise KernelWikiError("candidate-duplicate", "candidate numbers must be unique")
    merged: list[Candidate] = []
    for current in existing.prs:
        found = incoming.pop(current.number, None)
        if found is None:
            merged.append(replace(current, discovery_state="not-returned"))
        else:
            merged.append(
                replace(
                    found,
                    decision=current.decision,
                    reason=current.reason,
                    kernel_types=current.kernel_types or found.kernel_types,
                    techniques=current.techniques or found.techniques,
                    languages=current.languages or found.languages,
                    target_evidence=current.target_evidence or found.target_evidence,
                    discovery_state="returned",
                )
            )
    merged.extend(
        replace(item, decision="defer", reason="unreviewed-discovery", discovery_state="returned")
        for item in incoming.values()
    )
    ordered = tuple(sorted(merged, key=lambda item: item.number))
    return CandidateLedger(
        1,
        existing.repository_id,
        existing.repo,
        searched_at,
        existing.keywords_used,
        len(ordered),
        sum(item.decision == "include" for item in ordered),
        sum(item.decision == "defer" for item in ordered),
        sum(item.decision == "exclude" for item in ordered),
        ordered,
    )


def _matches_glob(path: str, pattern: str) -> bool:
    return fnmatch.fnmatchcase(path, pattern) or (
        pattern.startswith("**/") and fnmatch.fnmatchcase(path, pattern[3:])
    )


def _is_kernel_path(repository: RepositorySpec, path: str) -> bool:
    return not any(
        _matches_glob(path, pattern) for pattern in repository.skip_path_globs
    ) and any(_matches_glob(path, pattern) for pattern in repository.kernel_path_globs)


def _search_url(repository: RepositorySpec, term: str) -> str:
    query = urllib.parse.urlencode(
        {"q": f"repo:{repository.repo} is:pr {term}", "per_page": "100"}
    )
    return f"https://api.github.com/search/issues?{query}"


def _files_url(repository: RepositorySpec, number: int) -> str:
    return f"https://api.github.com/repos/{urllib.parse.quote(repository.repo, safe='/')}/pulls/{number}/files?per_page=100"


def _changed_file_paths(rows: Sequence[Any]) -> tuple[str, ...]:
    paths = [
        validate_repository_relative_posix_path(
            _require_mapping(row, "github-json-invalid").get("filename"),
            code="github-json-invalid",
            field="changed-file filename",
        )
        for row in rows
    ]
    if len(paths) != len(set(paths)):
        raise KernelWikiError("github-json-invalid", "changed-file names must be unique")
    return tuple(sorted(paths))


def discover_candidates(
    repository: RepositorySpec,
    *,
    terms: Sequence[str] | None = None,
    client: PaginatedJsonClient,
    limit: int | None = None,
) -> tuple[Candidate, ...]:
    if repository.host == "manual":
        raise KernelWikiError("adapter-manual", "repository requires reviewed manual candidates")
    if repository.host != "github":
        raise KernelWikiError("adapter-unknown", f"unsupported repository host {repository.host}")
    selected_terms = tuple(repository.search_terms if terms is None else terms)
    if not selected_terms or not all(isinstance(term, str) and term.strip() for term in selected_terms):
        raise KernelWikiError("discovery-term-invalid", "search terms must be nonblank")
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise KernelWikiError("discovery-limit-invalid", "limit must be positive")
    issues: dict[int, tuple[str, str, str]] = {}
    for term in selected_terms:
        for raw in client.get_paginated(_search_url(repository, term)):
            row = _require_mapping(raw, "github-json-invalid")
            number = row.get("number")
            if type(number) is not int or number <= 0:
                raise KernelWikiError("github-json-invalid", "PR number must be positive")
            expected_url = f"https://github.com/{repository.repo}/pull/{number}"
            title = _require_string(row.get("title"), "github-json-invalid", "title", None)
            url = _require_string(row.get("html_url"), "github-json-invalid", "html_url", None)
            created = _require_utc_iso8601(
                row.get("created_at"), "github-json-invalid", "created_at", None
            )
            if url != expected_url:
                raise KernelWikiError("github-json-invalid", "PR URL does not match repository")
            value = (title, url, created)
            if number in issues and issues[number] != value:
                raise KernelWikiError("github-duplicate-conflict", f"conflicting PR {number}")
            issues[number] = value
    candidates = []
    for number in sorted(issues):
        changed_paths = _changed_file_paths(client.get_paginated(_files_url(repository, number)))
        if not any(_is_kernel_path(repository, path) for path in changed_paths):
            continue
        title, url, created = issues[number]
        candidates.append(
            Candidate(
                number=number,
                title=title,
                date=created[:10],
                url=url,
                languages=tuple(sorted(repository.languages)),
                changed_paths=changed_paths,
                files_reviewed_count=len(changed_paths),
            )
        )
    ordered = tuple(candidates)
    return ordered if limit is None else ordered[:limit]


def render_discovery(candidates: Sequence[Candidate]) -> str:
    items = tuple(candidates)
    if len({item.number for item in items}) != len(items):
        raise KernelWikiError("candidate-duplicate", "candidate numbers must be unique")
    return canonical_json_bytes(
        {
            "schema_version": 1,
            "candidates": [
                _candidate_mapping(item) for item in sorted(items, key=lambda item: item.number)
            ],
        }
    ).decode("utf-8")


def _capture_metadata_from_mapping(raw: Any, path: Path) -> SourceCaptureMetadata:
    row = _require_mapping(raw, "capture-manifest-field", path)
    _require_exact_fields(row, CAPTURE_METADATA_FIELDS, "capture-manifest-field", path)
    taxonomy = _taxonomy_values()
    values = {
        field: _string_tuple(
            row[field], "capture-metadata-invalid", path, field=field, sorted_unique=True
        )
        for field in (
            "languages",
            "kernel_types",
            "techniques",
            "hardware_features",
            "tags",
            "audiences",
        )
    }
    for field, items in values.items():
        _require_taxonomy_subset(
            items,
            allowed=taxonomy[field],
            code="capture-metadata-invalid",
            field=field,
            path=path,
        )
    source_id = _require_string(row["source_id"], "capture-metadata-invalid", "source_id", path)
    repository_id = _require_string(
        row["repository_id"], "capture-metadata-invalid", "repository_id", path
    )
    target = _require_string(
        row["target_disposition"], "capture-metadata-invalid", "target_disposition", path
    )
    license_state = _require_string(
        row["license_state"], "capture-metadata-invalid", "license_state", path
    )
    if SOURCE_ID_RE.fullmatch(source_id) is None or REPOSITORY_ID_RE.fullmatch(repository_id) is None:
        raise KernelWikiError("capture-metadata-invalid", "source or repository id is invalid", path)
    if target not in taxonomy["target_matches"] or license_state not in taxonomy["license_states"]:
        raise KernelWikiError("capture-metadata-invalid", "closed taxonomy value is invalid", path)
    return SourceCaptureMetadata(
        source_id,
        _require_string(row["title"], "capture-metadata-invalid", "title", path),
        repository_id,
        _require_utc_iso8601(row["captured_at"], "capture-metadata-invalid", "captured_at", path),
        target,
        values["languages"],
        values["kernel_types"],
        values["techniques"],
        values["hardware_features"],
        values["tags"],
        license_state,
        values["audiences"],
    )


def _validate_capture_metadata(metadata: SourceCaptureMetadata) -> SourceCaptureMetadata:
    row = {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in metadata.__dict__.items()
    }
    return _capture_metadata_from_mapping(row, Path("<capture-request>"))


def _selection_from_mapping(raw: Any, path: Path) -> CaptureSelection:
    row = _require_mapping(raw, "capture-manifest-field", path)
    _require_exact_fields(row, CAPTURE_SELECTION_FIELDS, "capture-manifest-field", path)
    heading = row["heading_path"]
    if heading is not None:
        heading = _require_string(heading, "capture-selection-invalid", "heading_path", path)
    return CaptureSelection(
        validate_repository_relative_posix_path(
            row["upstream_path"], code="capture-selection-invalid", field="upstream_path", path=path
        ),
        heading,
        _require_string(row["role"], "capture-selection-invalid", "role", path),
        _require_string(row["mode"], "capture-selection-invalid", "mode", path),
    )


def _validate_capture_selections(
    selections: Sequence[CaptureSelection], path: Path | None = None
) -> tuple[CaptureSelection, ...]:
    items = tuple(selections)
    keys = []
    for item in items:
        key = validate_repository_relative_posix_path(
            item.upstream_path, code="capture-selection-invalid", field="upstream_path", path=path
        )
        if item.role not in FILE_ROLES or item.mode not in FILE_MODES or item.mode == "derived":
            raise KernelWikiError("capture-selection-invalid", "selection role or mode is invalid", path)
        if item.mode == "extracted" and item.heading_path is None:
            raise KernelWikiError("capture-selection-invalid", "extracted selection needs heading", path)
        keys.append(key)
    if keys != sorted(set(keys)):
        raise KernelWikiError("capture-selection-invalid", "selections must be sorted and unique", path)
    return items


def _validate_manual_selections(
    selections: Sequence[ManualFileSelection], path: Path
) -> tuple[ManualFileSelection, ...]:
    items = tuple(selections)
    names = []
    for item in items:
        upstream = item.upstream_path
        if upstream is not None:
            upstream = validate_repository_relative_posix_path(
                upstream, code="capture-selection-invalid", field="upstream_path", path=path
            )
        local_name = upstream or item.input_path.name
        validate_repository_relative_posix_path(
            local_name, code="capture-selection-invalid", field="local path", path=path
        )
        if item.role not in FILE_ROLES or item.mode not in FILE_MODES:
            raise KernelWikiError("capture-selection-invalid", "manual role or mode is invalid", path)
        if item.mode in {"verbatim", "upstream-patch", "extracted"} and upstream is None:
            raise KernelWikiError("capture-selection-invalid", "upstream-backed file needs upstream_path", path)
        if item.mode == "extracted" and item.heading_path is None:
            raise KernelWikiError("capture-selection-invalid", "extracted file needs heading", path)
        if item.mode == "derived" and upstream is not None:
            raise KernelWikiError("capture-selection-invalid", "derived file cannot claim upstream_path", path)
        names.append(local_name)
    if names != sorted(set(names)):
        raise KernelWikiError("capture-selection-invalid", "manual files must be sorted and unique", path)
    return items


def load_github_capture_manifest(
    path: Path,
) -> tuple[SourceCaptureMetadata, tuple[CaptureSelection, ...]]:
    document = _require_mapping(load_yaml_document(path), "capture-manifest-invalid", path)
    _require_exact_fields(
        document, GITHUB_CAPTURE_MANIFEST_FIELDS, "capture-manifest-field", path
    )
    if document["schema_version"] != 1 or type(document["schema_version"]) is not int:
        raise KernelWikiError("capture-manifest-invalid", "schema_version must be integer 1", path)
    raw = document["selections"]
    if not isinstance(raw, list):
        raise KernelWikiError("capture-manifest-invalid", "selections must be a list", path)
    metadata = _capture_metadata_from_mapping(document["metadata"], path)
    selections = tuple(_selection_from_mapping(item, path) for item in raw)
    return metadata, _validate_capture_selections(selections, path)


def load_manual_capture_manifest(path: Path) -> ManualCaptureManifest:
    document = _require_mapping(load_yaml_document(path), "capture-manifest-invalid", path)
    _require_exact_fields(
        document, MANUAL_CAPTURE_MANIFEST_FIELDS, "capture-manifest-field", path
    )
    if document["schema_version"] != 1 or type(document["schema_version"]) is not int:
        raise KernelWikiError("capture-manifest-invalid", "schema_version must be integer 1", path)
    raw_files = document["files"]
    if not isinstance(raw_files, list):
        raise KernelWikiError("capture-manifest-invalid", "files must be a list", path)
    files = []
    for raw in raw_files:
        row = _require_mapping(raw, "capture-manifest-field", path)
        _require_exact_fields(row, MANUAL_FILE_FIELDS, "capture-manifest-field", path)
        input_text = validate_repository_relative_posix_path(
            row["input_path"], code="capture-manifest-invalid", field="input_path", path=path
        )
        upstream = row["upstream_path"]
        if upstream is not None:
            upstream = validate_repository_relative_posix_path(
                upstream, code="capture-selection-invalid", field="upstream_path", path=path
            )
        heading = row["heading_path"]
        if heading is not None:
            heading = _require_string(heading, "capture-selection-invalid", "heading_path", path)
        files.append(
            ManualFileSelection(
                path.parent / Path(*PurePosixPath(input_text).parts),
                upstream,
                heading,
                _require_string(row["role"], "capture-selection-invalid", "role", path),
                _require_string(row["mode"], "capture-selection-invalid", "mode", path),
            )
        )
    source_kind = _require_string(
        document["source_kind"], "capture-manifest-invalid", "source_kind", path
    )
    url = _require_string(document["url"], "capture-manifest-invalid", "url", path)
    if source_kind not in {"manual-doc", "official-doc"} or urllib.parse.urlparse(url).scheme != "https":
        raise KernelWikiError("capture-manifest-invalid", "manual source kind or URL is invalid", path)
    result = ManualCaptureManifest(
        1,
        _capture_metadata_from_mapping(document["metadata"], path),
        source_kind,
        url,
        _require_string(
            document["document_revision"], "capture-manifest-invalid", "document_revision", path
        ),
        tuple(files),
    )
    _validate_manual_selections(result.files, path)
    return result


def _capture_metadata_mapping(
    metadata: SourceCaptureMetadata, *, source_kind: str, url: str, artifact_dir: str | None
) -> dict[str, Any]:
    document = {
        "schema_version": 1,
        "id": metadata.source_id,
        "source_kind": source_kind,
        "title": metadata.title,
        "url": url,
        "repository_id": metadata.repository_id,
        "captured_at": metadata.captured_at,
        "target_disposition": metadata.target_disposition,
        "languages": list(metadata.languages),
        "kernel_types": list(metadata.kernel_types),
        "techniques": list(metadata.techniques),
        "hardware_features": list(metadata.hardware_features),
        "tags": list(metadata.tags),
        "license_state": metadata.license_state,
        "audiences": list(metadata.audiences),
    }
    if artifact_dir is not None:
        document["artifact_dir"] = artifact_dir
    return document


def _render_source_markdown(metadata: Mapping[str, Any], body: str) -> str:
    frontmatter = yaml.safe_dump(dict(metadata), sort_keys=False, allow_unicode=True)
    return f"---\n{frontmatter}---\n{body.rstrip()}\n"


@dataclass(frozen=True)
class _CaptureAsset:
    local_path: str
    upstream_path: str | None
    heading_path: str | None
    role: str
    mode: str
    data: bytes


def _asset_mode(assets: Sequence[_CaptureAsset]) -> str:
    modes = {item.mode for item in assets}
    if modes <= {"verbatim", "upstream-patch"}:
        return "verbatim"
    if modes == {"extracted"}:
        return "extracted"
    if modes == {"derived"}:
        return "derived"
    raise KernelWikiError("capture-mode-mismatch", f"incompatible asset modes {sorted(modes)}")


def _provenance_document(
    *,
    metadata: SourceCaptureMetadata,
    origin_url: str,
    upstream_repo: str | None,
    upstream_sha: str | None,
    assets: Sequence[_CaptureAsset],
) -> dict[str, Any]:
    if metadata.license_state != APPROVED_LICENSE or metadata.audiences != ("designer",):
        raise KernelWikiError("license-code-exposure", "retained assets require approved Designer-only capture")
    return {
        "schema_version": 1,
        "origin_url": origin_url,
        "upstream_repo": upstream_repo,
        "upstream_sha": upstream_sha,
        "license_state": metadata.license_state,
        "retrieved_at": metadata.captured_at,
        "asset_mode": _asset_mode(assets),
        "allowed_audiences": ["designer"],
        "coder_access": "denied",
        "source_ids": [metadata.source_id],
        "files": [
            {
                "local_path": item.local_path,
                "upstream_path": item.upstream_path,
                "heading_path": item.heading_path,
                "role": item.role,
                "mode": item.mode,
                "sha256": sha256_bytes(item.data),
            }
            for item in sorted(assets, key=lambda item: item.local_path)
        ],
    }


def _relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _entry_exists(path: Path) -> bool:
    return os.path.lexists(path)


def _remove_empty_parents(path: Path, stop: Path) -> None:
    parent = path.parent
    while parent != stop:
        try:
            parent.rmdir()
        except OSError:
            return
        parent = parent.parent


def recover_capture_transactions(skill_root: Path) -> tuple[CaptureRecovery, ...]:
    root = validate_capture_skill_root(skill_root)
    staging_root = root / ".capture-staging"
    if not staging_root.exists():
        return ()
    recoveries = []
    for entry in sorted(staging_root.iterdir(), key=lambda path: path.name):
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        else:
            entry.unlink()
        recoveries.append(
            CaptureRecovery(
                entry.name,
                CaptureState.PREPARED,
                CaptureState.ROLLED_BACK,
                ("remove-stale-staging",),
            )
        )
    try:
        staging_root.rmdir()
    except OSError:
        pass
    return tuple(recoveries)


def _validate_staged_capture(
    skill_root: Path,
    source_relative: Path,
    staged_source: Path,
    artifact_relative: Path | None,
    staged_artifact: Path | None,
) -> None:
    from validate import validate_skill_root

    with tempfile.TemporaryDirectory() as directory:
        isolated = Path(directory) / "kernelwiki"
        isolated.mkdir()
        shutil.copytree(skill_root / "data", isolated / "data")
        for name in ("sources", "wiki", "artifacts"):
            current = skill_root / name
            if current.exists():
                shutil.copytree(current, isolated / name)
        target_source = isolated / source_relative
        target_source.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(staged_source, target_source)
        if artifact_relative is not None and staged_artifact is not None:
            target_artifact = isolated / artifact_relative
            target_artifact.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(staged_artifact, target_artifact)
        validate_skill_root(isolated, check_generated=False)


def _publish_capture(
    *,
    skill_root: Path,
    metadata: SourceCaptureMetadata,
    source_kind: str,
    source_url: str,
    source_relative: Path,
    body: str,
    assets: Sequence[_CaptureAsset],
    origin_url: str,
    upstream_repo: str | None,
    upstream_sha: str | None,
    failure_hook: Callable[[str], None] | None = None,
) -> CaptureResult:
    from catalog import build_generated_outputs, write_generated_outputs
    from validate import validate_skill_root

    root = validate_capture_skill_root(skill_root)
    recover_capture_transactions(root)
    metadata = _validate_capture_metadata(metadata)
    retained = tuple(assets) if metadata.license_state == APPROVED_LICENSE else ()
    artifact_relative = Path("artifacts") / metadata.source_id if retained else None
    source_metadata = _capture_metadata_mapping(
        metadata,
        source_kind=source_kind,
        url=source_url,
        artifact_dir=None if artifact_relative is None else artifact_relative.as_posix(),
    )
    final_source = require_within(root, root / source_relative)
    final_artifact = None if artifact_relative is None else require_within(root, root / artifact_relative)
    if _entry_exists(final_source) or (
        final_artifact is not None and _entry_exists(final_artifact)
    ):
        raise KernelWikiError("capture-exists", "immutable capture destination already exists")
    baseline = build_generated_outputs(validate_skill_root(root, check_generated=False))
    staging_root = root / ".capture-staging"
    staging_root.mkdir(exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f"{metadata.source_id}-", dir=staging_root))
    staged_source = stage / "source.md"
    staged_artifact = stage / "artifact" if retained else None
    created_source = False
    created_artifact = False
    try:
        staged_source.write_text(
            _render_source_markdown(source_metadata, body), encoding="utf-8"
        )
        if staged_artifact is not None:
            staged_artifact.mkdir()
            for asset in retained:
                local_path = validate_repository_relative_posix_path(
                    asset.local_path,
                    code="capture-selection-invalid",
                    field="local asset path",
                )
                output = require_within(staged_artifact, staged_artifact / local_path)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(asset.data)
            provenance = _provenance_document(
                metadata=metadata,
                origin_url=origin_url,
                upstream_repo=upstream_repo,
                upstream_sha=upstream_sha,
                assets=retained,
            )
            (staged_artifact / "PROVENANCE.yaml").write_text(
                yaml.safe_dump(provenance, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
        _validate_staged_capture(
            root, source_relative, staged_source, artifact_relative, staged_artifact
        )
        if failure_hook is not None:
            failure_hook("after-staging")
        if final_source.exists() or (final_artifact is not None and final_artifact.exists()):
            raise KernelWikiError("capture-exists", "immutable capture destination already exists")
        if staged_artifact is not None and final_artifact is not None:
            final_artifact.parent.mkdir(parents=True, exist_ok=True)
            os.rename(staged_artifact, final_artifact)
            created_artifact = True
            if failure_hook is not None:
                failure_hook("after-artifact-publish")
        final_source.parent.mkdir(parents=True, exist_ok=True)
        os.link(staged_source, final_source)
        created_source = True
        staged_source.unlink()
        if failure_hook is not None:
            failure_hook("after-source-publish")
        write_generated_outputs(root)
        validate_skill_root(root)
        captured_files = tuple(
            sorted(
                (
                    path
                    for path in (final_artifact.rglob("*") if final_artifact is not None else ())
                    if path.is_file()
                ),
                key=lambda path: path.as_posix(),
            )
        )
        return CaptureResult(metadata.source_id, final_source, final_artifact, captured_files)
    except Exception as error:
        if created_source and final_source.exists():
            final_source.unlink()
        if created_artifact and final_artifact is not None and final_artifact.exists():
            shutil.rmtree(final_artifact)
        _remove_empty_parents(final_source, root)
        if final_artifact is not None:
            _remove_empty_parents(final_artifact, root)
        try:
            write_generated_outputs(root, baseline)
        except Exception:
            pass
        if isinstance(error, KernelWikiError) and error.code == "capture-exists":
            raise
        message = error.message if isinstance(error, KernelWikiError) else str(error)
        raise KernelWikiError("capture-publish-failed", message, final_source) from error
    finally:
        shutil.rmtree(stage, ignore_errors=True)
        try:
            staging_root.rmdir()
        except OSError:
            pass


def _repository_for_capture(
    skill_root: Path, repository_id: str, repo: str
) -> RepositorySpec:
    repository = load_source_registry(skill_root / "data" / "source-repositories.yaml").get(
        repository_id
    )
    if repository is None or repository.repo != repo:
        raise KernelWikiError(
            "capture-repository-mismatch", "capture repository identity is not registered"
        )
    return repository


def _pr_source_relative(metadata: SourceCaptureMetadata, number: int) -> Path:
    prefix = f"source-{metadata.repository_id}-pr-{number}"
    if metadata.source_id == prefix:
        suffix = ""
    elif re.fullmatch(re.escape(prefix) + r"-r[1-9][0-9]*", metadata.source_id):
        suffix = metadata.source_id[len(prefix) :]
    else:
        raise KernelWikiError(
            "capture-source-id-mismatch",
            f"PR source_id must equal {prefix} or use an -rN revision suffix",
        )
    return Path("sources") / "prs" / metadata.repository_id / f"PR-{number}{suffix}.md"


def _require_pr_document(
    repository: RepositorySpec, number: int, raw: Any
) -> Mapping[str, Any]:
    row = _require_mapping(raw, "github-json-invalid")
    html = f"https://github.com/{repository.repo}/pull/{number}"
    expected = {
        "url": f"https://api.github.com/repos/{repository.repo}/pulls/{number}",
        "html_url": html,
        "diff_url": html + ".diff",
        "patch_url": html + ".patch",
        "number": number,
    }
    if any(row.get(key) != value for key, value in expected.items()):
        raise KernelWikiError("github-json-invalid", "PR identity or URL differs")
    for field in ("title", "state", "created_at", "updated_at"):
        _require_string(row.get(field), "github-json-invalid", field, None)
    for field in ("created_at", "updated_at"):
        _require_utc_iso8601(row[field], "github-json-invalid", field, None)
    for field in ("closed_at", "merged_at"):
        if row.get(field) is not None:
            _require_utc_iso8601(row[field], "github-json-invalid", field, None)
    user = row.get("user")
    head = row.get("head")
    if not isinstance(user, Mapping) or not isinstance(head, Mapping):
        raise KernelWikiError("github-json-invalid", "PR user and head are required")
    _require_string(user.get("login"), "github-json-invalid", "author", None)
    if GIT_SHA_RE.fullmatch(str(head.get("sha", ""))) is None:
        raise KernelWikiError("github-json-invalid", "PR head SHA is invalid")
    if type(row.get("changed_files")) is not int or row["changed_files"] < 0:
        raise KernelWikiError("github-json-invalid", "changed_files is invalid")
    return row


def _pr_file_facts(
    repository: RepositorySpec, head_sha: str, rows: Sequence[Any]
) -> tuple[tuple[str, ...], list[dict[str, Any]], bytes]:
    changed_paths = _changed_file_paths(rows)
    rows_by_name = {_require_mapping(row, "github-json-invalid")["filename"]: row for row in rows}
    facts = []
    fragments = []
    for filename in changed_paths:
        row = rows_by_name[filename]
        status = _require_string(row.get("status"), "github-json-invalid", "status", None)
        blob_sha = _require_string(row.get("sha"), "github-json-invalid", "sha", None)
        if GIT_SHA_RE.fullmatch(blob_sha) is None:
            raise KernelWikiError("github-json-invalid", "changed-file SHA is invalid")
        numeric = {}
        for field in ("additions", "deletions", "changes"):
            value = row.get(field)
            if type(value) is not int or value < 0:
                raise KernelWikiError("github-json-invalid", f"{field} is invalid")
            numeric[field] = value
        previous = row.get("previous_filename")
        if previous is not None:
            previous = validate_repository_relative_posix_path(
                previous, code="github-json-invalid", field="previous_filename"
            )
        patch = row.get("patch")
        if patch is not None and not isinstance(patch, str):
            raise KernelWikiError("github-json-invalid", "patch fragment is invalid")
        facts.append(
            {
                "filename": filename,
                "previous_filename": previous,
                "status": status,
                "sha": blob_sha,
                **numeric,
                "blob_url": row.get("blob_url"),
                "raw_url": row.get("raw_url"),
                "contents_url": row.get("contents_url"),
                "patch_available": patch is not None,
                "patch_fragment_sha256": None
                if patch is None
                else sha256_bytes(patch.encode("utf-8")),
            }
        )
        fragments.append(
            {"filename": filename, "previous_filename": previous, "patch": patch}
        )
    return changed_paths, facts, canonical_json_bytes(fragments)


def capture_github_pr(
    request: GitHubPRCaptureRequest,
    client: CaptureGitHubClient,
    *,
    failure_hook: Callable[[str], None] | None = None,
) -> CaptureResult:
    metadata = _validate_capture_metadata(request.metadata)
    selections = _validate_capture_selections(request.selections)
    root = validate_capture_skill_root(request.skill_root)
    repository = _repository_for_capture(root, metadata.repository_id, request.repo)
    if repository.host != "github" or type(request.number) is not int or request.number <= 0:
        raise KernelWikiError("capture-request-invalid", "PR capture request is invalid")
    pr_url = f"https://api.github.com/repos/{repository.repo}/pulls/{request.number}"
    pr = _require_pr_document(repository, request.number, client.get_json(pr_url))
    _require_capture_not_before(
        metadata.captured_at,
        tuple(
            pr[field]
            for field in ("created_at", "updated_at", "closed_at", "merged_at")
            if pr.get(field) is not None
        ),
        authority="pull-request",
    )
    head_sha = pr["head"]["sha"]
    rows = client.get_paginated(pr_url + "/files?per_page=100")
    changed_paths, file_facts, patch_fragments = _pr_file_facts(repository, head_sha, rows)
    ledger = load_candidate_ledger(
        root / "candidates" / "repos" / f"{repository.repository_id}.yaml",
        repository=repository,
    )
    candidate = ledger.by_number.get(request.number)
    if (
        pr["changed_files"] != len(changed_paths)
        or candidate is None
        or candidate.decision != "include"
        or candidate.changed_paths != changed_paths
        or candidate.files_reviewed_count != len(changed_paths)
    ):
        raise KernelWikiError(
            "capture-file-accounting", "reviewed ledger does not match complete PR file list"
        )
    if not {item.upstream_path for item in selections} <= set(changed_paths):
        raise KernelWikiError("capture-selection-missing", "selected path is absent from PR")
    assets = []
    selected_hashes = {}
    exact_patch = None
    if metadata.license_state == APPROVED_LICENSE:
        exact_patch = client.get_patch_bytes(pr["patch_url"])
        assets.append(
            _CaptureAsset(
                "pr.patch",
                f"pull/{request.number}.patch",
                None,
                "pr-diff",
                "upstream-patch",
                exact_patch,
            )
        )
        facts = {item["filename"]: item for item in file_facts}
        for selection in selections:
            data = client.get_file_bytes(repository.repo, selection.upstream_path, head_sha)
            if _git_blob_sha(data) != facts[selection.upstream_path]["sha"]:
                raise KernelWikiError(
                    "github-content-hash-mismatch", "selected bytes do not match reviewed blob"
                )
            selected_hashes[selection.upstream_path] = sha256_bytes(data)
            assets.append(
                _CaptureAsset(
                    f"files/{selection.upstream_path}",
                    selection.upstream_path,
                    selection.heading_path,
                    selection.role,
                    selection.mode,
                    data,
                )
            )
    confirmed = _require_pr_document(repository, request.number, client.get_json(pr_url))
    if confirmed["head"]["sha"] != head_sha or confirmed["updated_at"] != pr["updated_at"]:
        raise KernelWikiError("capture-upstream-changed", "PR identity changed during capture")
    body_facts = {
        "repository": repository.repo,
        "pr_number": request.number,
        "api_url": pr["url"],
        "html_url": pr["html_url"],
        "diff_url": pr["diff_url"],
        "patch_url": pr["patch_url"],
        "upstream_title": pr["title"],
        "author": pr["user"]["login"],
        "state": pr["state"],
        "created_at": pr["created_at"],
        "updated_at": pr["updated_at"],
        "closed_at": pr.get("closed_at"),
        "merged_at": pr.get("merged_at"),
        "head_sha": head_sha,
        "merge_sha": pr.get("merge_commit_sha"),
        "changed_files": len(changed_paths),
        "description_sha256": sha256_bytes((pr.get("body") or "").encode("utf-8")),
        "api_patch_fragments_sha256": sha256_bytes(patch_fragments),
        "patch_sha256": None if exact_patch is None else sha256_bytes(exact_patch),
        "patch_capture_status": "not-retained-license"
        if exact_patch is None
        else "exact-upstream-patch",
        "selected_file_sha256": dict(sorted(selected_hashes.items())),
        "files": file_facts,
    }
    body = (
        f"# {metadata.title}\n\nImmutable reviewed GitHub pull-request capture.\n\n"
        "## Capture facts\n\n```json\n"
        + canonical_json_bytes(body_facts).decode("utf-8")
        + "```\n"
    )
    return _publish_capture(
        skill_root=root,
        metadata=metadata,
        source_kind="github-pr",
        source_url=pr["html_url"],
        source_relative=_pr_source_relative(metadata, request.number),
        body=body,
        assets=assets,
        origin_url=pr["html_url"],
        upstream_repo=repository.repo,
        upstream_sha=head_sha,
        failure_hook=failure_hook,
    )


def _require_commit_document(
    repository: RepositorySpec, sha: str, raw: Any
) -> Mapping[str, Any]:
    row = _require_mapping(raw, "github-json-invalid")
    if (
        row.get("sha") != sha
        or row.get("url") != f"https://api.github.com/repos/{repository.repo}/commits/{sha}"
        or row.get("html_url") != f"https://github.com/{repository.repo}/commit/{sha}"
    ):
        raise KernelWikiError("github-json-invalid", "commit identity or URL differs")
    commit = row.get("commit")
    if not isinstance(commit, Mapping) or not isinstance(commit.get("author"), Mapping) or not isinstance(commit.get("committer"), Mapping):
        raise KernelWikiError("github-json-invalid", "commit author and committer are required")
    for person in (commit["author"], commit["committer"]):
        _require_string(person.get("name"), "github-json-invalid", "commit name", None)
        _require_utc_iso8601(person.get("date"), "github-json-invalid", "commit date", None)
    return row


def capture_github_commit(
    request: GitHubCommitCaptureRequest,
    client: CaptureGitHubClient,
    *,
    failure_hook: Callable[[str], None] | None = None,
) -> CaptureResult:
    metadata = _validate_capture_metadata(request.metadata)
    selections = _validate_capture_selections(request.selections)
    root = validate_capture_skill_root(request.skill_root)
    if GIT_SHA_RE.fullmatch(request.sha) is None:
        raise KernelWikiError("capture-request-invalid", "commit SHA is invalid")
    repository = _repository_for_capture(root, metadata.repository_id, request.repo)
    if repository.host != "github":
        raise KernelWikiError("capture-request-invalid", "commit capture needs GitHub lane")
    commit = _require_commit_document(
        repository,
        request.sha,
        client.get_json(f"https://api.github.com/repos/{repository.repo}/commits/{request.sha}"),
    )
    _require_capture_not_before(
        metadata.captured_at,
        (commit["commit"]["author"]["date"], commit["commit"]["committer"]["date"]),
        authority="commit author/committer",
    )
    assets = []
    selected_hashes = {}
    if metadata.license_state == APPROVED_LICENSE:
        for selection in selections:
            data = client.get_file_bytes(repository.repo, selection.upstream_path, request.sha)
            selected_hashes[selection.upstream_path] = sha256_bytes(data)
            assets.append(
                _CaptureAsset(
                    f"files/{selection.upstream_path}",
                    selection.upstream_path,
                    selection.heading_path,
                    selection.role,
                    selection.mode,
                    data,
                )
            )
    api_author = commit.get("author") or {}
    body_facts = {
        "repository": repository.repo,
        "sha": request.sha,
        "api_url": commit["url"],
        "html_url": commit["html_url"],
        "author": commit["commit"]["author"]["name"],
        "author_login": api_author.get("login"),
        "authored_at": commit["commit"]["author"]["date"],
        "committed_at": commit["commit"]["committer"]["date"],
        "selected_file_sha256": dict(sorted(selected_hashes.items())),
    }
    body = (
        f"# {metadata.title}\n\nImmutable reviewed GitHub commit capture.\n\n"
        "## Capture facts\n\n```json\n"
        + canonical_json_bytes(body_facts).decode("utf-8")
        + "```\n"
    )
    return _publish_capture(
        skill_root=root,
        metadata=metadata,
        source_kind="github-commit",
        source_url=commit["html_url"],
        source_relative=Path("sources") / "commits" / repository.repository_id / f"{request.sha}.md",
        body=body,
        assets=assets,
        origin_url=commit["html_url"],
        upstream_repo=repository.repo,
        upstream_sha=request.sha,
        failure_hook=failure_hook,
    )


def _validate_manual_origin(
    repository: RepositorySpec, manifest: ManualCaptureManifest
) -> None:
    parsed = urllib.parse.urlparse(manifest.url)
    if repository.repository_id == "huawei-ascend-docs":
        revision = parse_huawei_cann_document_revision(manifest.url)
        if (
            revision is None
            or manifest.source_kind != "official-doc"
            or manifest.document_revision != revision
        ):
            raise KernelWikiError("capture-origin-mismatch", "Huawei document URL/revision differs")
        return
    parts = urllib.parse.unquote(parsed.path).strip("/").split("/")
    repo_parts = repository.repo.split("/")
    route = parts[2] if len(parts) > 2 else None
    valid_route = (route == "commit" and len(parts) == 4) or (
        route in {"blob", "tree"} and len(parts) >= 5
    )
    if (
        parsed.scheme != "https"
        or parsed.netloc != "github.com"
        or parts[:2] != repo_parts
        or not valid_route
        or len(parts) < 4
        or parts[3] != manifest.document_revision
        or GIT_SHA_RE.fullmatch(manifest.document_revision) is None
    ):
        raise KernelWikiError("capture-origin-mismatch", "manual GitHub origin is not revision-bound")


def _read_manual_input(manifest_path: Path, input_path: Path) -> bytes:
    root = manifest_path.parent.resolve(strict=True)
    try:
        resolved = input_path.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, RuntimeError, ValueError) as error:
        raise KernelWikiError("capture-input-invalid", "manual input escapes manifest directory", input_path) from error
    if not resolved.is_file():
        raise KernelWikiError("capture-input-invalid", "manual input must be a file", input_path)
    return resolved.read_bytes()


def capture_manual_source(
    request: ManualCaptureRequest,
    *,
    failure_hook: Callable[[str], None] | None = None,
) -> CaptureResult:
    root = validate_capture_skill_root(request.skill_root)
    manifest = load_manual_capture_manifest(request.manifest_path)
    metadata = _validate_capture_metadata(manifest.metadata)
    repository = load_source_registry(root / "data" / "source-repositories.yaml").get(
        metadata.repository_id
    )
    if repository is None or repository.host != "manual":
        raise KernelWikiError("capture-repository-mismatch", "manual capture needs manual lane")
    _validate_manual_origin(repository, manifest)
    assets = []
    selected_hashes = {}
    if metadata.license_state == APPROVED_LICENSE:
        for selection in manifest.files:
            data = _read_manual_input(request.manifest_path, selection.input_path)
            local_name = selection.upstream_path or selection.input_path.name
            selected_hashes[local_name] = sha256_bytes(data)
            assets.append(
                _CaptureAsset(
                    f"files/{local_name}",
                    selection.upstream_path,
                    selection.heading_path,
                    selection.role,
                    selection.mode,
                    data,
                )
            )
    body_facts = {
        "repository": repository.repo,
        "document_revision": manifest.document_revision,
        "origin_url": manifest.url,
        "selected_file_sha256": dict(sorted(selected_hashes.items())),
    }
    body = (
        f"# {metadata.title}\n\nImmutable reviewed manual source capture.\n\n"
        "## Capture facts\n\n```json\n"
        + canonical_json_bytes(body_facts).decode("utf-8")
        + "```\n"
    )
    return _publish_capture(
        skill_root=root,
        metadata=metadata,
        source_kind=manifest.source_kind,
        source_url=manifest.url,
        source_relative=Path("sources") / "docs" / f"{metadata.source_id}.md",
        body=body,
        assets=assets,
        origin_url=manifest.url,
        upstream_repo=repository.repo,
        upstream_sha=manifest.document_revision,
        failure_hook=failure_hook,
    )


def capture_result_document(result: CaptureResult, skill_root: Path) -> dict[str, Any]:
    root = validate_capture_skill_root(skill_root)
    return {
        "schema_version": 1,
        "source_id": result.source_id,
        "source_path": _relative(root, result.source_path),
        "source_sha256": sha256_file(result.source_path),
        "artifact_dir": None if result.artifact_dir is None else _relative(root, result.artifact_dir),
        "provenance_sha256": None
        if result.artifact_dir is None
        else sha256_file(result.artifact_dir / "PROVENANCE.yaml"),
        "captured_files": [
            {"path": _relative(root, path), "sha256": sha256_file(path)}
            for path in result.captured_files
        ],
    }


def main(argv: Sequence[str]) -> int:
    parser = StableArgumentParser(description="Read-only KernelWiki source candidate discovery")
    subparsers = parser.add_subparsers(
        dest="command", required=True, parser_class=StableArgumentParser
    )
    discover = subparsers.add_parser("discover")
    discover.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    discover.add_argument("--repository", required=True)
    discover.add_argument("--term", action="append", dest="terms")
    discover.add_argument("--limit", type=int, default=100)
    args = parser.parse_args(list(argv))
    root = validate_capture_skill_root(args.root)
    registry = load_source_registry(root / "data" / "source-repositories.yaml")
    repository = registry.get(args.repository)
    if repository is None:
        raise KernelWikiError("repository-unknown", f"unknown repository {args.repository}")
    if repository.host == "manual":
        raise KernelWikiError("adapter-manual", "repository requires reviewed manual candidates")
    load_candidate_ledger(
        root / "candidates" / "repos" / f"{repository.repository_id}.yaml",
        repository=repository,
    )
    candidates = discover_candidates(
        repository, terms=args.terms, client=GitHubClient(), limit=args.limit
    )
    print(render_discovery(candidates), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
