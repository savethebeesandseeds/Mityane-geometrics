#!/usr/bin/env python3
"""Read-only cleanup/evidence and maintained-document link checks.

Run in the managed container with ``python3 code/scripts/check-layout.py``.
Historical artifacts are verified from the recorded nonignored snapshot only;
the potentially large ignored output/build/dependency trees are never walked.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECEIPT = "doc/EMBEDDINGS_CLEANUP_2026-10-18.json"
HISTORICAL_ROOT = "archive/embeddings/pre-cleanup-2026-10-18"
IGNORED_ARTIFACTS = tuple(
    f"code/embeddings/{name}" for name in ("output", ".build", ".external")
)
REMOVED_COMPONENT_PATHS = (
    "code", "doc", "archive", "container.ps1", "setup.sh", "dependencies.lock",
    "LICENSE", "THIRD_PARTY_NOTICES.md", "Makefile",
)


class Checks:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.files = 0
        self.links = 0
        self.relocations = 0
        self.shell_roots = 0

    def require(self, condition: bool, message: str) -> bool:
        if not condition:
            self.errors.append(message)
        return condition


def repository_path(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty repository-relative path")
    relative = PurePosixPath(value.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts or ":" in value:
        raise ValueError(f"{label} is not a safe repository-relative path: {value}")
    path = ROOT.joinpath(*relative.parts)
    return path


def read_receipt(path: Path) -> dict:
    """Allow a compact public summary to redirect to its detailed manifest."""
    seen: set[Path] = set()
    while True:
        if path in seen:
            raise ValueError("cleanup detailed_manifest redirect contains a cycle")
        seen.add(path)
        document = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(document, dict):
            raise ValueError(f"cleanup receipt is not an object: {path}")
        if "preserved_files" in document:
            return document
        target = document.get("detailed_manifest")
        if target is None:
            raise ValueError("cleanup receipt has neither preserved_files nor detailed_manifest")
        path = repository_path(target, "detailed_manifest")


def hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def check_snapshot(receipt: dict, checks: Checks) -> None:
    historical = receipt.get("historical_root", HISTORICAL_ROOT)
    archive = repository_path(historical, "historical_root")
    if not checks.require(archive.is_dir(), f"historical snapshot missing: {historical}"):
        return
    entries = receipt.get("preserved_files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("preserved_files must be a nonempty list")
    expected: set[str] = set()
    archive_relative = archive.relative_to(ROOT).as_posix()
    for item in entries:
        if not isinstance(item, dict):
            raise ValueError("each preserved_files entry must be an object")
        recorded = item.get("path")
        # Detailed inventories may record paths within the snapshot or complete
        # repository-relative archive paths. Both describe the same exact tree.
        candidate = repository_path(recorded, "preserved file path")
        if candidate.is_relative_to(archive):
            relative = candidate.relative_to(archive).as_posix()
        else:
            relative = str(recorded).replace("\\", "/")
            candidate = repository_path(f"{archive_relative}/{relative}", "preserved file path")
        if not checks.require(relative not in expected, f"duplicate preserved file: {relative}"):
            continue
        expected.add(relative)
        size, digest = item.get("bytes"), item.get("sha256")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError(f"invalid preserved byte count: {relative}")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise ValueError(f"invalid preserved SHA256: {relative}")
        try:
            actual_digest, actual_size = hash_file(candidate)
        except OSError as error:
            checks.require(False, f"cannot read preserved file: {relative}: {error}")
            continue
        checks.require(actual_size == size, f"preserved size differs: {relative}")
        checks.require(actual_digest == digest.lower(), f"preserved SHA256 differs: {relative}")
        checks.files += 1
        if checks.files % 1000 == 0:
            print(f"Snapshot hashes: {checks.files}/{len(entries)} files", flush=True)
    print("Checking exact archived file inventory...", flush=True)
    # os.walk consumes directory listings without adding an is_file/stat call
    # for every leaf, which is expensive through a Windows Docker bind mount.
    actual = {
        (Path(parent) / name).relative_to(archive).as_posix()
        for parent, _, names in os.walk(archive)
        for name in names
    }
    for relative in sorted(expected - actual):
        checks.require(False, f"snapshot inventory member missing: {relative}")
    for relative in sorted(actual - expected):
        checks.require(False, f"snapshot has an unrecorded file: {relative}")
    # The exact member-set comparison establishes count without relying on a
    # potentially stale summary count or walking ignored research artifacts.
    checks.require(len(actual) == len(entries), "snapshot file count differs from preserved_files")


def check_relocations(receipt: dict, checks: Checks) -> None:
    entries = receipt.get("relocations")
    if not isinstance(entries, list) or not entries:
        raise ValueError("relocations must be a nonempty list")
    for item in entries:
        if not isinstance(item, dict):
            raise ValueError("each relocation must be an object")
        old = repository_path(item.get("old"), "old relocation path")
        new = repository_path(item.get("new"), "new relocation path")
        checks.require(new.exists(), f"relocated destination missing: {new.relative_to(ROOT)}")
        if old != new:
            checks.require(not old.exists(), f"retired source path remains: {old.relative_to(ROOT)}")
        checks.relocations += 1
    for relative in REMOVED_COMPONENT_PATHS:
        path = ROOT / "code/embeddings" / relative
        checks.require(not path.exists(), f"redundant component path remains: {path.relative_to(ROOT)}")


def check_ignore_protection(checks: Checks) -> None:
    # Ask Git about one hypothetical child per protected tree. Do not enumerate
    # datasets, checkpoints, SDK copies or build products in those trees.
    probes = [f"{relative}/__layout_ignore_probe__" for relative in IGNORED_ARTIFACTS]
    result = subprocess.run(
        ["git", "-c", f"safe.directory={ROOT}", "check-ignore", "--no-index", "--stdin"],
        cwd=ROOT, input="\n".join(probes) + "\n", capture_output=True, text=True,
        check=False,
    )
    if result.returncode not in (0, 1):
        checks.require(False, f"git check-ignore failed: {result.stderr.strip()}")
        return
    ignored = set(result.stdout.splitlines())
    for relative, probe in zip(IGNORED_ARTIFACTS, probes):
        checks.require(probe in ignored, f"historical artifact directory is not ignored: {relative}")
        checks.require((ROOT / relative).is_dir(), f"historical artifact directory is missing: {relative}")


def markdown_destinations(text: str):
    """Yield line/destination pairs, excluding fenced and inline code examples."""
    lines: list[str] = []
    fence: str | None = None
    for line in text.splitlines(keepends=True):
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if marker:
            run = marker.group(1)
            if fence is None:
                fence = run
            elif run[0] == fence[0] and len(run) >= len(fence):
                fence = None
            lines.append("\n" if line.endswith("\n") else "")
        elif fence is None:
            lines.append(re.sub(r"(`+).*?\1", "", line))
        else:
            lines.append("\n" if line.endswith("\n") else "")
    source = "".join(lines)
    for match in re.finditer(r"\]\(\s*", source):
        start = match.end()
        if start >= len(source):
            continue
        if source[start] == "<":
            end = source.find(">", start + 1)
            if end != -1:
                yield source.count("\n", 0, start) + 1, source[start + 1:end]
            continue
        depth, end = 0, start
        while end < len(source):
            char = source[end]
            if char == "\\" and end + 1 < len(source):
                end += 2
                continue
            if char.isspace() or (char == ")" and depth == 0):
                break
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            end += 1
        if end > start:
            yield source.count("\n", 0, start) + 1, source[start:end]
    for match in re.finditer(r"^ {0,3}\[[^\]\n]+\]:\s*(?:<([^>]+)>|(\S+))", source, re.MULTILINE):
        yield source.count("\n", 0, match.start()) + 1, match.group(1) or match.group(2)


def check_shell_roots(source: Path, checks: Checks) -> None:
    """Resolve literal dirname-based repository roots without running shell code."""
    pattern = re.compile(
        r'^\s*cd\s+--\s+"\$\(dirname\s+--\s+"\$\{BASH_SOURCE\[0\]\}"\)/'
        r'(?P<parents>(?:\.\./)*\.\.)"\s*$'
    )
    for line, text in enumerate(source.read_text(encoding="utf-8-sig").splitlines(), 1):
        match = pattern.fullmatch(text)
        if match is None:
            continue
        checks.shell_roots += 1
        target = (source.parent / match.group("parents")).resolve()
        checks.require(
            target == ROOT,
            f"shell repository root differs: {source.relative_to(ROOT)}:{line}: "
            f"{match.group('parents')} resolves to {target}",
        )


def check_markdown_links(checks: Checks) -> None:
    sources: list[Path] = []
    existence: dict[Path, bool] = {}
    # Prune ignored roots before descending; avoid traversing their many GB.
    for base, directories, files in os.walk(ROOT / "code"):
        parent = Path(base)
        directories[:] = [
            name for name in directories
            if name not in {".git", ".build", ".external", "output", "__pycache__", "archive"}
        ]
        sources.extend(parent / name for name in files if name.lower().endswith(".md"))
        for name in files:
            if name.lower().endswith(".sh"):
                check_shell_roots(parent / name, checks)
    sources.append(ROOT / "doc/EMBEDDINGS.md")
    for source in sorted(sources):
        if not checks.require(source.is_file(), f"maintained document missing: {source.relative_to(ROOT)}"):
            continue
        for line, destination in markdown_destinations(source.read_text(encoding="utf-8-sig")):
            destination = html.unescape(re.sub(r"\\([()<> ])", r"\1", destination))
            if not destination or destination.startswith("#"):
                continue
            local = unquote(destination.split("#", 1)[0].split("?", 1)[0])
            local = re.sub(r":\d+(?::\d+)?$", "", local)
            if not local or urlsplit(local).scheme:
                continue
            target = (ROOT / local.lstrip("/")) if local.startswith("/") else (source.parent / local)
            target = Path(os.path.abspath(target))
            # Frozen evidence keeps the original path identities; its internal
            # link graph is intentionally excluded from maintained link checks.
            if target.is_relative_to(ROOT / "archive"):
                continue
            checks.links += 1
            if target not in existence:
                existence[target] = target.exists()
            checks.require(
                existence[target],
                f"broken Markdown link: {source.relative_to(ROOT)}:{line}: {destination}",
            )


def main() -> int:
    if not Path("/.dockerenv").is_file() or ROOT != Path("/mityane"):
        print("Run through the managed mityane-geometrics container.", file=sys.stderr)
        return 2
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", default=DEFAULT_RECEIPT, help="repository-relative cleanup receipt")
    parser.add_argument(
        "--links-only", action="store_true",
        help="check ignored-artifact protection, maintained links and literal shell roots without hashing the snapshot or checking relocations",
    )
    args = parser.parse_args()
    checks = Checks()
    try:
        if not args.links_only:
            receipt = read_receipt(repository_path(args.receipt, "receipt"))
            print("Checking archived original bytes and SHA256 identities...", flush=True)
            check_snapshot(receipt, checks)
            print("Checking relocated sources and removed component configuration...", flush=True)
            check_relocations(receipt, checks)
        print("Checking historical artifact ignore protection...", flush=True)
        check_ignore_protection(checks)
        print("Checking maintained Markdown links and literal shell roots...", flush=True)
        check_markdown_links(checks)
    except (OSError, ValueError, KeyError, TypeError) as error:
        checks.errors.append(str(error))
    if checks.errors:
        for error in checks.errors:
            print(f"FAIL {error}", file=sys.stderr)
        print(f"Layout verification failed with {len(checks.errors)} problem(s).", file=sys.stderr)
        return 1
    if args.links_only:
        print(
            f"Links verified: {checks.links} maintained Markdown links, "
            f"{checks.shell_roots} literal shell roots; historical artifact trees remain ignored."
        )
    else:
        print(
            f"Layout verified: {checks.files} preserved files, "
            f"{checks.relocations} relocations, {checks.links} maintained Markdown links, "
            f"{checks.shell_roots} literal shell roots; "
            "historical artifact trees remain ignored."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
