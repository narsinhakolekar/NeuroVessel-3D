from __future__ import annotations

import csv
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import requests

BASE_URL = "https://huggingface.co/datasets/MedOtter/LUNA16/resolve/main"
MANIFEST_CSV = Path("outputs/metrics/airrc_uid_manifest_with_luna_subset.csv")
PREFLIGHT_CSV = Path("outputs/metrics/airrc_download_preflight.csv")
OUTPUT_ROOT = Path("data/airrc_luna16")
PROGRESS_CSV = Path("outputs/metrics/airrc_download_manifest.csv")
CHUNK_SIZE = 1024 * 1024
MAX_RETRIES = 5
TIMEOUT = (30, 180)

TYPE_BYTES = {
    "MET_CHAR": 1,
    "MET_UCHAR": 1,
    "MET_SHORT": 2,
    "MET_USHORT": 2,
    "MET_INT": 4,
    "MET_UINT": 4,
    "MET_FLOAT": 4,
    "MET_DOUBLE": 8,
}


def parse_mhd(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def expected_bytes_from_header(path: Path) -> int:
    h = parse_mhd(path)
    if h.get("CompressedData", "False").lower() == "true":
        raise ValueError(f"Compressed MetaImage data is not expected: {path}")
    dims = [int(x) for x in h["DimSize"].split()]
    if len(dims) != 3:
        raise ValueError(f"Expected 3D DimSize in {path}, got {dims}")
    element_type = h["ElementType"]
    if element_type not in TYPE_BYTES:
        raise ValueError(f"Unknown ElementType {element_type!r} in {path}")
    return dims[0] * dims[1] * dims[2] * TYPE_BYTES[element_type]


def download_small(session: requests.Session, url: str, dest: Path) -> None:
    part = Path(str(dest) + ".part")
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with session.get(url, stream=True, timeout=TIMEOUT) as response:
                response.raise_for_status()
                with part.open("wb") as f:
                    for chunk in response.iter_content(chunk_size=64 * 1024):
                        if chunk:
                            f.write(chunk)
            if not part.exists() or part.stat().st_size == 0:
                raise IOError("Downloaded header is empty")
            part.replace(dest)
            return
        except Exception:
            if attempt == MAX_RETRIES:
                raise
            time.sleep(min(2 ** attempt, 20))
    raise RuntimeError(f"Could not download {url}")


def download_raw(session: requests.Session, url: str, dest: Path, expected: int) -> None:
    """Stream a raw CT; retain .part on interruption and resume with HTTP Range."""
    part = Path(str(dest) + ".part")
    if dest.exists() and dest.stat().st_size == expected:
        return
    if dest.exists():
        dest.unlink()

    for attempt in range(1, MAX_RETRIES + 1):
        offset = part.stat().st_size if part.exists() else 0
        if offset > expected:
            part.unlink()
            offset = 0
        if offset == expected:
            part.replace(dest)
            return

        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with session.get(url, headers=headers, stream=True, timeout=TIMEOUT) as response:
                if response.status_code == 416 and part.exists() and part.stat().st_size == expected:
                    part.replace(dest)
                    return
                if offset and response.status_code == 206:
                    content_range = response.headers.get("Content-Range", "")
                    match = re.match(r"bytes\s+(\d+)-", content_range)
                    if not match or int(match.group(1)) != offset:
                        raise IOError(f"Unexpected Content-Range for resume: {content_range!r}")
                    mode = "ab"
                elif response.status_code == 200:
                    # Server ignored Range (or this is a fresh request): restart cleanly.
                    mode = "wb"
                    offset = 0
                else:
                    response.raise_for_status()
                    raise IOError(f"Unexpected HTTP status {response.status_code}")

                with part.open(mode) as f:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)

            actual = part.stat().st_size if part.exists() else 0
            if actual == expected:
                part.replace(dest)
                return
            if actual > expected:
                part.unlink()
                raise IOError(f"Raw file too large: {actual} bytes; expected {expected}")
            raise IOError(f"Incomplete raw file: {actual} / {expected} bytes")
        except Exception as exc:
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"Failed after {MAX_RETRIES} attempts; partial file retained at {part}: {exc}") from exc
            print(f"  Retry {attempt}/{MAX_RETRIES - 1}: {exc}", flush=True)
            time.sleep(min(2 ** attempt, 30))


def save_progress(rows: list[dict]) -> None:
    PROGRESS_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(PROGRESS_CSV, index=False)


def main() -> int:
    if not MANIFEST_CSV.exists() or not PREFLIGHT_CSV.exists():
        print("Required manifest/preflight CSV is missing. Run from the NeuroVessel project root.")
        return 2

    manifest = pd.read_csv(MANIFEST_CSV)
    preflight = pd.read_csv(PREFLIGHT_CSV)
    required_manifest = {"series_instance_uid", "patient_id", "luna_subset", "contrast_status"}
    required_preflight = {"series_instance_uid", "http_status", "raw_bytes"}
    if not required_manifest.issubset(manifest.columns):
        raise ValueError(f"Manifest missing columns: {required_manifest - set(manifest.columns)}")
    if not required_preflight.issubset(preflight.columns):
        raise ValueError(f"Preflight missing columns: {required_preflight - set(preflight.columns)}")
    if len(manifest) != 254 or manifest["series_instance_uid"].nunique() != 254:
        raise ValueError("Expected exactly 254 unique AirRC series in the UID manifest")

    sizes = preflight.set_index("series_instance_uid")["raw_bytes"].to_dict()
    statuses = preflight.set_index("series_instance_uid")["http_status"].to_dict()
    session = requests.Session()
    session.headers.update({"User-Agent": "NeuroVessel-AirRC-downloader/1.0"})

    progress: list[dict] = []
    total_bytes = int(preflight.loc[preflight["http_status"] == 200, "raw_bytes"].sum())
    completed_bytes = 0
    print(f"AirRC/LUNA16 download: {len(manifest)} CT series")
    print(f"Expected raw payload: {total_bytes / (1024 ** 3):.2f} GiB")
    print("Destination: data/airrc_luna16/<subset>/")
    print("Press Ctrl+C to stop safely; rerun this script to resume.\n")

    sample_uid = "1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860"
    sample_dir = OUTPUT_ROOT / "sample" / "subset8"

    try:
        for index, row in enumerate(manifest.itertuples(index=False), start=1):
            uid = str(row.series_instance_uid)
            subset = str(row.luna_subset)
            contrast = str(row.contrast_status)
            if int(statuses.get(uid, 0)) != 200:
                raise RuntimeError(f"Preflight did not confirm HTTP 200 for {uid}")
            expected_from_server = int(sizes[uid])
            target_dir = OUTPUT_ROOT / subset
            target_dir.mkdir(parents=True, exist_ok=True)
            mhd_path = target_dir / f"{uid}.mhd"
            raw_path = target_dir / f"{uid}.raw"
            base_url = f"{BASE_URL}/{subset}/{uid}"

            print(f"[{index:03d}/254] {row.patient_id} | {subset} | {contrast}", flush=True)
            try:
                # Reuse the already downloaded and physically validated sample pair.
                if uid == sample_uid:
                    for ext in ("mhd", "raw"):
                        src = sample_dir / f"{uid}.{ext}"
                        dst = target_dir / f"{uid}.{ext}"
                        if not dst.exists() and src.exists():
                            shutil.copy2(src, dst)

                if not mhd_path.exists():
                    download_small(session, f"{base_url}.mhd", mhd_path)
                header = parse_mhd(mhd_path)
                if header.get("ElementDataFile") != f"{uid}.raw":
                    raise ValueError(f"Unexpected ElementDataFile in {mhd_path}: {header.get('ElementDataFile')!r}")
                expected_from_header = expected_bytes_from_header(mhd_path)
                if expected_from_header != expected_from_server:
                    raise ValueError(
                        f"Header size {expected_from_header} does not match preflight {expected_from_server}"
                    )
                if raw_path.exists() and raw_path.stat().st_size != expected_from_server:
                    raw_path.unlink()
                download_raw(session, f"{base_url}.raw", raw_path, expected_from_server)
                if raw_path.stat().st_size != expected_from_server:
                    raise IOError("Final raw-file size check failed")

                completed_bytes += expected_from_server
                progress.append({
                    "series_instance_uid": uid,
                    "patient_id": row.patient_id,
                    "luna_subset": subset,
                    "contrast_status": contrast,
                    "mhd_path": str(mhd_path),
                    "raw_path": str(raw_path),
                    "expected_raw_bytes": expected_from_server,
                    "actual_raw_bytes": raw_path.stat().st_size,
                    "status": "complete",
                    "error": "",
                })
                save_progress(progress)
                print(f"  VERIFIED: {raw_path.stat().st_size:,} bytes", flush=True)
            except Exception as exc:
                progress.append({
                    "series_instance_uid": uid,
                    "patient_id": row.patient_id,
                    "luna_subset": subset,
                    "contrast_status": contrast,
                    "mhd_path": str(mhd_path),
                    "raw_path": str(raw_path),
                    "expected_raw_bytes": expected_from_server,
                    "actual_raw_bytes": raw_path.stat().st_size if raw_path.exists() else 0,
                    "status": "failed",
                    "error": str(exc),
                })
                save_progress(progress)
                print(f"  FAILED: {exc}", flush=True)
                print("Stopping at first failed case. Fix/retry by rerunning; completed files will be skipped.")
                return 1
    except KeyboardInterrupt:
        save_progress(progress)
        print("\nStopped by user. Completed files are saved; partial .part files can resume on rerun.")
        return 130

    save_progress(progress)
    complete = sum(r["status"] == "complete" for r in progress)
    print(f"\nFINISHED: {complete}/254 CT series verified.")
    print(f"Progress manifest: {PROGRESS_CSV}")
    print("Note: this checks byte counts and MetaImage header consistency; it does not replace a full voxel-level geometry audit for every case.")
    return 0 if complete == 254 else 1


if __name__ == "__main__":
    sys.exit(main())
