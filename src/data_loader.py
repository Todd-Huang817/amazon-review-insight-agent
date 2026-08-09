from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import BinaryIO, Iterable

import pandas as pd

from src.config import DEFAULT_DATA_PATH


COLUMN_ALIASES = {
    "ASIN": "product_id",
    "ProductId": "product_id",
    "product_id": "product_id",
    "product/productId": "product_id",
    "asin": "product_id",
    "商品ID": "product_id",
    "商品编号": "product_id",
    "parent_asin": "parent_asin",
    "UserId": "user_id",
    "review/userId": "user_id",
    "user_id": "user_id",
    "Score": "rating",
    "score": "rating",
    "stars": "rating",
    "star_rating": "rating",
    "review/score": "rating",
    "overall": "rating",
    "rating": "rating",
    "评分": "rating",
    "星级": "rating",
    "Time": "timestamp",
    "review/time": "timestamp",
    "unixReviewTime": "timestamp",
    "timestamp": "timestamp",
    "date": "timestamp",
    "review_date": "timestamp",
    "日期": "timestamp",
    "评论时间": "timestamp",
    "Summary": "review_title",
    "review/summary": "review_title",
    "summary": "review_title",
    "title": "review_title",
    "review_title": "review_title",
    "评论标题": "review_title",
    "Text": "review_text",
    "review/text": "review_text",
    "reviewText": "review_text",
    "text": "review_text",
    "review": "review_text",
    "review_body": "review_text",
    "review_content": "review_text",
    "content": "review_text",
    "评论": "review_text",
    "评论内容": "review_text",
    "HelpfulnessNumerator": "helpful_votes",
    "helpful_vote": "helpful_votes",
    "helpful_votes": "helpful_votes",
    "verified": "verified_purchase",
    "verified_purchase": "verified_purchase",
}


def _read_dataframe(source: str | Path | BinaryIO, filename: str | None = None) -> pd.DataFrame:
    name = filename or str(source)
    suffix = Path(name).suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(source)
    if suffix in {".jsonl", ".json"}:
        return pd.read_json(source, lines=suffix == ".jsonl")
    return pd.read_csv(source)


def normalize_schema(frame: pd.DataFrame, source_name: str) -> pd.DataFrame:
    data = frame.copy()
    for alias, target in COLUMN_ALIASES.items():
        if alias == target or alias not in data:
            continue
        if target in data:
            data[target] = data[target].combine_first(data[alias])
        else:
            data[target] = data[alias]
        data = data.drop(columns=alias)
    if "product_id" not in data:
        data["product_id"] = pd.Series(pd.NA, index=data.index, dtype="string")
    else:
        data["product_id"] = data["product_id"].astype("string")
    if "parent_asin" not in data:
        data["parent_asin"] = data["product_id"]
    else:
        data["parent_asin"] = data["parent_asin"].astype("string")
    if "user_id" not in data:
        data["user_id"] = pd.Series("", index=data.index, dtype="string")
    else:
        data["user_id"] = data["user_id"].fillna("").astype("string")
    if "rating" not in data:
        data["rating"] = pd.Series(float("nan"), index=data.index, dtype=float)
    else:
        data["rating"] = pd.to_numeric(data["rating"], errors="coerce")
    if "review_title" not in data:
        data["review_title"] = ""
    else:
        data["review_title"] = data["review_title"].fillna("").astype(str)
    if "review_text" not in data:
        data["review_text"] = ""
    else:
        data["review_text"] = data["review_text"].fillna("").astype(str)
    if "helpful_votes" not in data:
        data["helpful_votes"] = 0
    else:
        data["helpful_votes"] = pd.to_numeric(data["helpful_votes"], errors="coerce").fillna(0).astype(int)
    if "verified_purchase" not in data:
        data["verified_purchase"] = False
    else:
        data["verified_purchase"] = data["verified_purchase"].fillna(False).astype(bool)

    timestamp_source = data["timestamp"] if "timestamp" in data else pd.Series(index=data.index, dtype=object)
    raw_timestamp = pd.to_numeric(timestamp_source, errors="coerce")
    if raw_timestamp.notna().any():
        unit = "ms" if raw_timestamp.dropna().median() > 10_000_000_000 else "s"
        numeric_dates = pd.to_datetime(raw_timestamp, unit=unit, utc=True, errors="coerce")
        string_dates = pd.to_datetime(timestamp_source.where(raw_timestamp.isna()), utc=True, errors="coerce")
        data["review_date"] = numeric_dates.fillna(string_dates).dt.tz_localize(None)
    else:
        data["review_date"] = pd.to_datetime(timestamp_source, utc=True, errors="coerce").dt.tz_localize(None)

    identity = (
        data["product_id"].fillna("")
        + "|"
        + data["user_id"].fillna("")
        + "|"
        + timestamp_source.fillna("").astype(str)
        + "|"
        + data["review_text"]
        + "|"
        + data["review_title"]
        + "|"
        + data["rating"].fillna("").astype(str)
    )
    generated_ids = identity.map(lambda value: hashlib.sha1(value.encode("utf-8")).hexdigest()[:16])
    if "review_id" in data:
        existing_ids = data["review_id"].fillna("").astype(str)
        data["review_id"] = existing_ids.where(existing_ids.str.strip().ne(""), generated_ids)
    else:
        data["review_id"] = generated_ids
    data["source_name"] = source_name

    data.loc[~data["rating"].between(1, 5), "rating"] = float("nan")
    data = data.drop_duplicates("review_id", keep="first")
    return data.reset_index(drop=True)


def load_reviews(path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = _read_dataframe(path)
    return normalize_schema(frame, source_name="Stanford SNAP Amazon Fine Foods")


def load_uploaded_reviews(uploaded_file: BinaryIO) -> pd.DataFrame:
    payload = uploaded_file.read()
    buffer = io.BytesIO(payload)
    frame = _read_dataframe(buffer, filename=uploaded_file.name)
    return normalize_schema(frame, source_name=f"Uploaded: {uploaded_file.name}")


def load_uploaded_review_files(uploaded_files: Iterable[BinaryIO]) -> pd.DataFrame:
    frames = [load_uploaded_reviews(uploaded_file) for uploaded_file in uploaded_files]
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    return combined.drop_duplicates("review_id", keep="first").reset_index(drop=True)
