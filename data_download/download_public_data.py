#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Download the minimal public NeuroRepFood data stream required to reproduce the
CLIP encoding analyses.

Normal use, from anywhere:
    python data_download/download_public_data.py

The script:
  - resolves the repository root from its own location;
  - verifies the pinned Dataverse dataset version;
  - downloads only files listed in public_data_manifest.csv;
  - creates the required repository-local data/resources tree automatically;
  - resumes interrupted downloads when the server supports HTTP Range;
  - verifies file size and SHA-1 checksum;
  - skips files that are already present and valid;
  - verifies the small parent-RSA derived input bundled with the repository.

It never writes to historical_outputs/.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from html.parser import HTMLParser
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path


SERVER = "https://dataverse.nl"
DATASET_PID = "doi:10.34894/TVPLVR"
DATASET_VERSION = "1.0"

EXPECTED_MANIFEST_ROWS = 201
EXPECTED_CATEGORY_COUNTS = {
    "glmsingle": 75,
    "stimuli": 97,
    "ratings": 25,
    "inherited_feature_inputs": 3,
    "atlas": 1,
}

BUNDLED_PARENT_RSA = (
    "resources/derived_inputs/parent_rsa/model_rdm_vectors.csv"
)
BUNDLED_PARENT_RSA_SHA1_LF = "67ad9a35b8c60a661e47a9b6ff647f912272d425"

USER_AGENT = "NeuroRepFood-CLIP-public-data-downloader/1.1"


def human_bytes(n: int) -> str:
    x = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if x < 1024.0 or unit == "TiB":
            return f"{x:.2f} {unit}"
        x /= 1024.0
    return f"{x:.2f} TiB"


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def sha1_text_normalized_lf(path: Path) -> str:
    """SHA-1 of a text file after normalizing CRLF/CR line endings to LF."""
    data = path.read_bytes()
    data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha1(data).hexdigest()

def request_json(url: str, timeout: int = 120) -> dict:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Dataverse HTTP {e.code}: {body[:1000]}"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Could not contact Dataverse: {e}") from e

    if payload.get("status") not in (None, "OK"):
        raise RuntimeError(f"Unexpected Dataverse response: {payload}")
    return payload


def get_pinned_dataset_files(server: str) -> dict[str, dict]:
    query = urllib.parse.urlencode({"persistentId": DATASET_PID})
    url = (
        f"{server.rstrip('/')}/api/datasets/:persistentId/versions/"
        f"{DATASET_VERSION}?{query}"
    )
    payload = request_json(url)
    data = payload.get("data")
    if not isinstance(data, dict):
        raise RuntimeError("Unexpected Dataverse dataset-version response.")

    observed_version = (
        f"{data.get('versionNumber')}.{data.get('versionMinorNumber')}"
    )
    if observed_version != DATASET_VERSION:
        raise RuntimeError(
            f"Expected Dataverse version {DATASET_VERSION}, got "
            f"{observed_version}."
        )
    if data.get("versionState") != "RELEASED":
        raise RuntimeError(
            f"Dataverse version {DATASET_VERSION} is not RELEASED."
        )

    by_path: dict[str, dict] = {}
    for entry in data.get("files", []):
        df = entry.get("dataFile", {}) or {}
        filename = (
            entry.get("label")
            or df.get("filename")
            or df.get("originalFileName")
            or ""
        )
        directory = entry.get("directoryLabel") or ""
        remote_path = (
            f"{directory}/{filename}" if directory else filename
        ).replace("\\", "/")
        checksum = df.get("checksum") or {}
        by_path[remote_path] = {
            "file_id": int(df["id"]),
            "filesize": int(df.get("filesize") or 0),
            "checksum_type": str(checksum.get("type") or ""),
            "checksum_value": str(checksum.get("value") or "").lower(),
            "restricted": bool(entry.get("restricted", False)),
        }
    return by_path


def load_manifest(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing download manifest: {path}")

    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if len(rows) != EXPECTED_MANIFEST_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_MANIFEST_ROWS} manifest rows, found {len(rows)}."
        )

    counts = Counter(r["category"] for r in rows)
    if dict(counts) != EXPECTED_CATEGORY_COUNTS:
        raise RuntimeError(
            "Unexpected category counts in public_data_manifest.csv.\n"
            f"Expected: {EXPECTED_CATEGORY_COUNTS}\n"
            f"Observed: {dict(counts)}"
        )

    required = {
        "category", "local_path", "discovery_file_id", "dataverse_path",
        "filesize_bytes", "checksum_type", "checksum_value",
    }
    for i, row in enumerate(rows, start=2):
        missing = required - set(row)
        if missing:
            raise RuntimeError(f"Manifest line {i} lacks columns: {missing}")
        if row["checksum_type"].upper() != "SHA-1":
            raise RuntimeError(
                f"Manifest line {i} uses unsupported checksum "
                f"{row['checksum_type']}."
            )
        lp = Path(row["local_path"])
        if lp.is_absolute() or ".." in lp.parts:
            raise RuntimeError(
                f"Unsafe local path in manifest line {i}: {lp}"
            )

    return rows


def validate_manifest_against_dataverse(
    rows: list[dict], remote_files: dict[str, dict]
) -> list[dict]:
    resolved = []
    problems = []

    for row in rows:
        path = row["dataverse_path"]
        remote = remote_files.get(path)
        if remote is None:
            problems.append(f"MISSING: {path}")
            continue

        expected_size = int(row["filesize_bytes"])
        expected_sha1 = row["checksum_value"].lower()

        if remote["restricted"]:
            problems.append(f"RESTRICTED: {path}")
        if remote["filesize"] != expected_size:
            problems.append(
                f"SIZE CHANGED: {path}: expected {expected_size}, "
                f"Dataverse reports {remote['filesize']}"
            )
        if remote["checksum_type"].upper() != "SHA-1":
            problems.append(
                f"CHECKSUM TYPE CHANGED: {path}: "
                f"{remote['checksum_type']}"
            )
        if remote["checksum_value"] != expected_sha1:
            problems.append(
                f"CHECKSUM CHANGED: {path}: expected {expected_sha1}, "
                f"Dataverse reports {remote['checksum_value']}"
            )

        item = dict(row)
        item["resolved_file_id"] = remote["file_id"]
        resolved.append(item)

    if problems:
        raise RuntimeError(
            "Pinned Dataverse version does not match the release manifest:\n  "
            + "\n  ".join(problems)
        )

    return resolved


def existing_file_is_valid(path: Path, expected_size: int, expected_sha1: str) -> bool:
    if not path.is_file():
        return False
    if path.stat().st_size != expected_size:
        return False
    return sha1_file(path).lower() == expected_sha1.lower()



class _DirIndexParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self._href = None
        self._text = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "a":
            self._href = dict(attrs).get("href")
            self._text = []

    def handle_data(self, data):
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag):
        if tag.lower() == "a" and self._href is not None:
            self.links.append((self._href, "".join(self._text).strip()))
            self._href = None
            self._text = []


_DIRINDEX_CACHE = {}


def get_dirindex_download_url(
    server: str,
    remote_path: str,
    timeout: int = 120,
) -> str:
    """
    Resolve Dataverse's canonical download href for one file from the
    version-pinned directory index.

    Dataverse's directory-index API is specifically intended for crawlable,
    restartable downloads and emits the installation's canonical Access-API
    href for each file. This is used as a fallback when the generic
    /api/access/datafile/{id} form returns 404.
    """
    rp = remote_path.replace("\\", "/").strip("/")
    if "/" in rp:
        folder, filename = rp.rsplit("/", 1)
    else:
        folder, filename = "", rp

    cache_key = (server.rstrip("/"), folder)
    if cache_key not in _DIRINDEX_CACHE:
        params = {
            "persistentId": DATASET_PID,
            "version": DATASET_VERSION,
        }
        if folder:
            params["folder"] = folder

        url = (
            f"{server.rstrip('/')}/api/datasets/:persistentId/dirindex?"
            + urllib.parse.urlencode(params)
        )
        req = urllib.request.Request(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html,*/*"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                html = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Directory-index HTTP {e.code} for {folder or '/'}: "
                f"{body[:1000]}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Could not query Dataverse directory index for "
                f"{folder or '/'}: {e}"
            ) from e

        parser = _DirIndexParser()
        parser.feed(html)
        _DIRINDEX_CACHE[cache_key] = parser.links

    matches = [
        href for href, label in _DIRINDEX_CACHE[cache_key]
        if label.rstrip("/") == filename
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Could not uniquely resolve {remote_path!r} in the Dataverse "
            f"directory index (matches={len(matches)})."
        )

    return urllib.parse.urljoin(server.rstrip("/") + "/", matches[0])


def _stream_download_url(
    url: str,
    part: Path,
    expected_size: int,
    start: int,
    timeout: int = 180,
):
    headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}
    if start:
        headers["Range"] = f"bytes={start}-"

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        status = getattr(resp, "status", None) or resp.getcode()

        if start and status == 206:
            mode = "ab"
        else:
            # Some endpoints/redirect targets ignore Range. Restart safely.
            mode = "wb"

        with part.open(mode) as f:
            while True:
                block = resp.read(8 * 1024 * 1024)
                if not block:
                    break
                f.write(block)


def download_one(
    server: str,
    file_id: int,
    remote_path: str,
    destination: Path,
    expected_size: int,
    expected_sha1: str,
    retries: int = 3,
):
    destination.parent.mkdir(parents=True, exist_ok=True)

    if existing_file_is_valid(destination, expected_size, expected_sha1):
        return "SKIP"

    # Do not keep a corrupt completed file.
    if destination.exists():
        destination.unlink()

    part = destination.with_name(destination.name + ".part")
    if part.exists() and part.stat().st_size > expected_size:
        part.unlink()

    generic_url = f"{server.rstrip('/')}/api/access/datafile/{file_id}"
    canonical_url = None

    for attempt in range(1, retries + 1):
        start_byte = part.stat().st_size if part.exists() else 0

        # First try the standard file-id endpoint. If Dataverse returns 404,
        # resolve the canonical href emitted by the version-pinned directory
        # index and retry through that route.
        urls = [generic_url]
        if canonical_url is not None and canonical_url != generic_url:
            urls.append(canonical_url)

        last_error = None
        for url_index, url in enumerate(urls):
            try:
                _stream_download_url(
                    url, part, expected_size, start_byte, timeout=180
                )
                last_error = None
                break
            except urllib.error.HTTPError as e:
                last_error = e

                if e.code == 404 and canonical_url is None:
                    print(
                        "    generic Dataverse file endpoint returned 404; "
                        "resolving canonical directory-index link...",
                        flush=True,
                    )
                    canonical_url = get_dirindex_download_url(
                        server, remote_path
                    )
                    print(
                        f"    canonical link: {canonical_url}",
                        flush=True,
                    )
                    try:
                        start_byte = part.stat().st_size if part.exists() else 0
                        _stream_download_url(
                            canonical_url, part, expected_size, start_byte,
                            timeout=180
                        )
                        last_error = None
                        break
                    except Exception as e2:
                        last_error = e2
                # Otherwise try next candidate URL / retry.
            except Exception as e:
                last_error = e

        if last_error is None:
            if not part.exists() or part.stat().st_size != expected_size:
                last_error = RuntimeError(
                    f"incomplete download: expected {expected_size} bytes, "
                    f"got {part.stat().st_size if part.exists() else 0}"
                )
            else:
                observed = sha1_file(part).lower()
                if observed != expected_sha1.lower():
                    last_error = RuntimeError(
                        f"SHA-1 mismatch: expected {expected_sha1}, "
                        f"got {observed}"
                    )
                else:
                    os.replace(part, destination)
                    return "DOWNLOADED"

        if attempt >= retries:
            raise RuntimeError(
                f"Failed to download file id {file_id} ({remote_path}) to "
                f"{destination} after {retries} attempts: {last_error}"
            ) from last_error

        wait = 2 ** (attempt - 1)
        print(
            f"    attempt {attempt} failed ({last_error}); retrying in "
            f"{wait} s...",
            flush=True,
        )
        time.sleep(wait)

    raise RuntimeError("unreachable")


def verify_bundled_parent_rsa(repo_root: Path):
    path = repo_root / BUNDLED_PARENT_RSA
    if not path.is_file():
        raise FileNotFoundError(
            "The repository-bundled parent-RSA input is missing:\n"
            f"  {path}\n"
            "This file is intentionally distributed with the code release "
            "because the exact 16-column historical input is not present in "
            "Dataverse v1.0."
        )
    observed = sha1_text_normalized_lf(path).lower()
    if observed != BUNDLED_PARENT_RSA_SHA1_LF:
        raise RuntimeError(
            "Bundled parent-RSA input checksum mismatch:\n"
            f"  {path}\n"
            f"Expected normalized SHA-1: {BUNDLED_PARENT_RSA_SHA1_LF}\n"
            f"Observed normalized SHA-1: {observed}"
        )


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    default_repo_root = script_dir.parent

    ap = argparse.ArgumentParser(
        description=(
            "Download the minimal public data needed for the NeuroRepFood "
            "CLIP reproduction."
        )
    )
    ap.add_argument(
        "--repo-root",
        type=Path,
        default=default_repo_root,
        help="Repository root. Defaults to the parent of data_download/.",
    )
    ap.add_argument(
        "--server",
        default=SERVER,
        help=f"Dataverse server (default: {SERVER}).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Resolve and validate the pinned Dataverse files and local tree, "
            "but do not download."
        ),
    )
    args = ap.parse_args()

    repo_root = args.repo_root.resolve()
    manifest_path = script_dir / "public_data_manifest.csv"

    rows = load_manifest(manifest_path)
    total_bytes = sum(int(r["filesize_bytes"]) for r in rows)

    print("=" * 88)
    print("NeuroRepFood CLIP — minimal public-data download")
    print("=" * 88)
    print(f"Repository root : {repo_root}")
    print(f"Dataverse       : {DATASET_PID}")
    print(f"Pinned version  : {DATASET_VERSION}")
    print(f"Manifest files  : {len(rows)}")
    print(f"Download size   : {human_bytes(total_bytes)}")
    print()

    verify_bundled_parent_rsa(repo_root)
    print("Bundled parent-RSA input: OK")

    print("Checking pinned Dataverse dataset and manifest...")
    remote_files = get_pinned_dataset_files(args.server)
    resolved = validate_manifest_against_dataverse(rows, remote_files)
    print("Dataverse manifest: OK")
    print()

    if args.dry_run:
        print("DRY RUN COMPLETE — no public files were downloaded.")
        print("The repository data tree is resolvable from Dataverse v1.0.")
        return 0

    downloaded = 0
    skipped = 0
    n = len(resolved)

    for i, row in enumerate(resolved, start=1):
        rel = Path(row["local_path"])
        dest = repo_root / rel
        size = int(row["filesize_bytes"])
        sha1 = row["checksum_value"].lower()
        fid = int(row["resolved_file_id"])

        print(
            f"[{i:03d}/{n:03d}] {rel} ({human_bytes(size)})",
            flush=True,
        )
        status = download_one(
            args.server, fid, row["dataverse_path"], dest, size, sha1
        )
        if status == "SKIP":
            skipped += 1
            print("    already present; checksum OK", flush=True)
        else:
            downloaded += 1
            print("    downloaded; checksum OK", flush=True)

    print()
    print("=" * 88)
    print("DATASET READY")
    print("=" * 88)
    print(f"Downloaded this run : {downloaded}")
    print(f"Already valid       : {skipped}")
    print(f"Verified public files: {n}/{n}")
    print("Bundled parent-RSA input: verified")
    print()
    print("Created/verified:")
    print("  data/glmsingle/")
    print("  data/stimuli/")
    print("  data/ratings/")
    print("  resources/atlas/")
    print("  resources/derived_inputs/parent_rsa/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
