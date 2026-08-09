from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import urllib.request
from pathlib import Path

import pandas as pd

from src.config import DEFAULT_DATA_PATH, RAW_DIR, SOURCE_FILE_URL


RAW_PATH = RAW_DIR / "finefoods.txt.gz"
MANIFEST_PATH = RAW_DIR / "finefoods_manifest.json"


def download_source() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_PATH.exists():
        return
    request = urllib.request.Request(SOURCE_FILE_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response, RAW_PATH.open("wb") as target:
        while chunk := response.read(1024 * 1024):
            target.write(chunk)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_records(limit: int) -> list[dict]:
    records: list[dict] = []
    current: dict[str, str] = {}
    key_map = {
        "product/productId": "ProductId",
        "review/userId": "UserId",
        "review/profileName": "ProfileName",
        "review/helpfulness": "Helpfulness",
        "review/score": "Score",
        "review/time": "Time",
        "review/summary": "Summary",
        "review/text": "Text",
    }
    with gzip.open(RAW_PATH, "rt", encoding="utf-8", errors="replace") as source:
        for line in source:
            line = line.rstrip("\n")
            if not line:
                if current:
                    records.append(current)
                    current = {}
                    if len(records) >= limit:
                        break
                continue
            key, separator, value = line.partition(": ")
            if separator and key in key_map:
                current[key_map[key]] = value
    return records


def prepare(limit: int) -> None:
    download_source()
    records = parse_records(limit)
    frame = pd.DataFrame(records)
    if frame.empty:
        raise RuntimeError("No review records were parsed from the source file.")
    helpful = frame.pop("Helpfulness").str.split("/", n=1, expand=True)
    frame["HelpfulnessNumerator"] = pd.to_numeric(helpful[0], errors="coerce").fillna(0).astype(int)
    frame["HelpfulnessDenominator"] = pd.to_numeric(helpful[1], errors="coerce").fillna(0).astype(int)
    frame["Score"] = pd.to_numeric(frame["Score"], errors="coerce")
    frame["Time"] = pd.to_numeric(frame["Time"], errors="coerce")

    DEFAULT_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(DEFAULT_DATA_PATH, index=False)
    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "source_url": SOURCE_FILE_URL,
                "raw_file": str(RAW_PATH),
                "sha256": sha256(RAW_PATH),
                "prepared_rows": len(frame),
                "output_file": str(DEFAULT_DATA_PATH),
                "synthetic_reviews": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Prepared {len(frame):,} authentic review records at {DEFAULT_DATA_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare a real Amazon Fine Foods review sample.")
    parser.add_argument("--limit", type=int, default=30_000)
    args = parser.parse_args()
    prepare(args.limit)
