from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DEFAULT_DATA_PATH = PROCESSED_DIR / "finefoods_reviews.parquet"

SOURCE_NAME = "Stanford SNAP Amazon Fine Foods"
SOURCE_URL = "https://snap.stanford.edu/data/web-FineFoods.html"
SOURCE_FILE_URL = "https://snap.stanford.edu/data/finefoods.txt.gz"

ASPECT_LABELS = {
    "taste": "Taste & flavor",
    "texture": "Texture",
    "freshness": "Freshness",
    "packaging": "Packaging",
    "portion_size": "Quantity & portion",
    "price_value": "Price & value",
    "ingredients": "Ingredients & nutrition",
    "delivery": "Delivery condition",
    "overall_quality": "Overall quality",
}

ASPECT_LABELS_ZH = {
    "taste": "口味与风味",
    "texture": "口感",
    "freshness": "新鲜度",
    "packaging": "包装",
    "portion_size": "数量与份量",
    "price_value": "价格与价值",
    "ingredients": "成分与营养",
    "delivery": "配送状态",
    "overall_quality": "整体品质",
}

SENTIMENT_COLORS = {
    "positive": "#2A9D8F",
    "neutral": "#E9C46A",
    "mixed": "#457B9D",
    "negative": "#D85D3F",
}
