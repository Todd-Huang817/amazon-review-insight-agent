from __future__ import annotations

import html
import math
import re
from collections import Counter

import pandas as pd

from src.config import ASPECT_LABELS, ASPECT_LABELS_ZH


ASPECT_RULES = {
    "taste": {
        "taste", "tastes", "tasted", "flavor", "flavour", "sweet", "sweetness",
        "salty", "bitter", "sour", "delicious", "bland", "aftertaste", "yummy",
    },
    "texture": {
        "texture", "crunchy", "crispy", "chewy", "hard", "soft", "stale", "dry",
        "soggy", "mushy", "sticky", "tough", "crumbly",
    },
    "freshness": {
        "fresh", "freshness", "stale", "expired", "expiration", "spoiled", "rancid",
        "mold", "mould", "old", "shelf life",
    },
    "packaging": {
        "package", "packaging", "pack", "bag", "box", "seal", "sealed", "wrapper",
        "leak", "leaking", "broken", "crushed", "damaged", "open", "opened", "resealable",
    },
    "portion_size": {
        "size", "small", "tiny", "portion", "quantity", "amount", "ounces", "oz",
        "count", "pieces", "serving", "half empty",
    },
    "price_value": {
        "price", "priced", "expensive", "cost", "value", "money", "cheap", "overpriced",
        "worth", "deal",
    },
    "ingredients": {
        "ingredient", "ingredients", "sugar", "calorie", "calories", "organic", "natural",
        "gluten", "allergen", "allergy", "vegan", "protein", "nutrition", "artificial",
    },
    "delivery": {
        "delivery", "delivered", "shipping", "shipped", "arrived", "melted", "warm",
        "heat", "late", "warehouse",
    },
}

NEGATIVE_TERMS = {
    "bad", "awful", "terrible", "disappointed", "disappointing", "waste", "poor",
    "hate", "worst", "inedible", "gross", "nasty", "problem", "issue", "wrong",
    "broken", "damaged", "stale", "expired", "melted", "overpriced", "bland",
}

PAIN_TERMS = NEGATIVE_TERMS | {
    "artificial", "bitter", "bland", "crushed", "dry", "expired", "hard", "late",
    "leaking", "melted", "mold", "opened", "overpriced", "rancid", "salty", "small",
    "soggy", "sour", "spoiled", "stale", "sticky", "sweet", "tiny", "tough",
}

POSITIVE_TERMS = {
    "love", "loved", "great", "excellent", "delicious", "perfect", "favorite", "best",
    "good", "fresh", "tasty", "amazing", "wonderful", "recommend", "happy",
}

STOP_WORDS = {
    "the", "and", "this", "that", "was", "were", "with", "for", "you", "but", "not",
    "are", "have", "had", "has", "they", "them", "from", "too", "very", "just", "can",
    "would", "could", "should", "about", "when", "what", "which", "will", "your", "our",
    "out", "one", "all", "get", "got", "did", "does", "than", "then", "there", "their",
    "product", "amazon", "buy", "bought", "order", "ordered", "really", "much", "more",
    "some", "because", "been", "into", "only", "also", "like", "these", "those", "its",
}

NEED_MAP = {
    "taste": "口味平衡且符合商品描述",
    "texture": "在食用时保持理想口感",
    "freshness": "收到新鲜、未过期且品质稳定的食品",
    "packaging": "包装完整、密封可靠并保护食品形态",
    "portion_size": "获得与页面描述一致的数量和份量",
    "price_value": "价格与数量、品质和体验相匹配",
    "ingredients": "清晰了解成分、营养和过敏原信息",
    "delivery": "运输过程中保持食品完好和适宜温度",
    "overall_quality": "获得稳定且符合预期的整体品质",
}

NEED_MAP_EN = {
    "taste": "Flavor complaints",
    "texture": "Texture complaints",
    "freshness": "Freshness complaints",
    "packaging": "Packaging complaints",
    "portion_size": "Quantity and portion complaints",
    "price_value": "Price and value complaints",
    "ingredients": "Ingredient and nutrition complaints",
    "delivery": "Delivery complaints",
    "overall_quality": "Overall quality complaints",
}

NEED_TERM_LABELS_ZH = {
    "package": "包装",
    "packaging": "包装",
    "flavor": "风味",
    "flavour": "风味",
    "taste": "口味",
    "broken": "破损",
    "damaged": "损坏",
    "crushed": "挤压变形",
    "opened": "开封",
    "open": "开封",
    "seal": "密封",
    "sealed": "密封",
    "leaking": "漏液",
    "leak": "漏液",
    "stale": "不新鲜",
    "expired": "过期",
    "spoiled": "变质",
    "mold": "发霉",
    "melted": "融化",
    "bland": "味道寡淡",
    "bitter": "发苦",
    "salty": "过咸",
    "sour": "过酸",
    "sweet": "过甜",
    "dry": "干硬",
    "hard": "太硬",
    "soggy": "受潮",
    "sticky": "粘连",
    "small": "份量偏小",
    "tiny": "份量过小",
    "overpriced": "价格偏高",
    "expensive": "价格偏高",
    "artificial": "人工添加感",
}

NEED_TERM_LABELS_EN = {term: term.replace("_", " ") for term in NEED_TERM_LABELS_ZH}

NEED_ASPECT_LABELS_ZH = {
    "taste": "口味",
    "texture": "口感",
    "freshness": "新鲜度",
    "packaging": "包装",
    "portion_size": "份量",
    "price_value": "价格",
    "ingredients": "成分",
    "delivery": "配送",
    "overall_quality": "质量",
}

NEED_ASPECT_LABELS_EN = {
    "taste": "Taste",
    "texture": "Texture",
    "freshness": "Freshness",
    "packaging": "Packaging",
    "portion_size": "Portion",
    "price_value": "Price",
    "ingredients": "Ingredients",
    "delivery": "Delivery",
    "overall_quality": "Quality",
}
ACTION_MAP = {
    "taste": ("Product", "开展口味分层测试，核查甜度、风味强度与Listing描述是否一致。"),
    "texture": ("Quality", "复核配方与储存条件对口感的影响，并按批次检查一致性。"),
    "freshness": ("Quality", "核查保质期、密封和库存周转，优先审查近期低分批次。"),
    "packaging": ("Packaging", "审查密封与抗压结构，执行模拟运输测试并核对破损集中ASIN。"),
    "portion_size": ("Operations", "核对页面数量、净重和图片比例，降低消费者预期偏差。"),
    "price_value": ("Operations", "评估价格与包装数量的价值感，并强化可验证的差异化卖点。"),
    "ingredients": ("Compliance", "复核成分、营养及过敏原展示，涉及安全信号时转人工合规审核。"),
    "delivery": ("Fulfillment", "按季节和地区检查温控及运输损坏，区分FBA履约与产品包装责任。"),
    "overall_quality": ("Product", "抽样复核低分评论，定位尚未归类的质量问题后再决定改版。"),
}

ACTION_MAP_EN = {
    "taste": ("Product", "Run segmented taste tests and verify that sweetness and flavor intensity match the listing."),
    "texture": ("Quality", "Review formulation and storage effects on texture, then check consistency by batch."),
    "freshness": ("Quality", "Audit shelf life, seals, and inventory rotation, prioritizing recent low-rated batches."),
    "packaging": ("Packaging", "Audit seal and compression strength, run transit simulations, and identify ASINs with concentrated damage."),
    "portion_size": ("Operations", "Align listing quantity, net weight, and image scale to reduce expectation gaps."),
    "price_value": ("Operations", "Review price-to-quantity value and strengthen verifiable differentiated benefits."),
    "ingredients": ("Compliance", "Review ingredient, nutrition, and allergen disclosures; route safety signals to human compliance review."),
    "delivery": ("Fulfillment", "Analyze temperature and transit damage by season and region, separating FBA from packaging responsibility."),
    "overall_quality": ("Product", "Sample low-rated reviews, classify unassigned quality issues, and then decide whether to revise the product."),
}

OWNER_LABELS_ZH = {
    "Product": "产品",
    "Quality": "质量",
    "Packaging": "包装",
    "Operations": "运营",
    "Compliance": "合规",
    "Fulfillment": "履约",
}


def clean_text(value: str) -> str:
    text = html.unescape(str(value))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z][a-z'-]{2,}", text.lower())


def infer_sentiment(rating: float, text: str) -> str:
    if pd.notna(rating):
        numeric_rating = float(rating)
        if numeric_rating >= 5:
            return "positive"
        if numeric_rating >= 3:
            return "neutral"
        if numeric_rating >= 1:
            return "negative"
    tokens = set(_tokens(text))
    positive_hits = len(tokens & POSITIVE_TERMS)
    negative_hits = len(tokens & NEGATIVE_TERMS)
    if positive_hits and negative_hits:
        return "mixed"
    if positive_hits > negative_hits:
        return "positive"
    if negative_hits > positive_hits:
        return "negative"
    return "neutral"


def infer_emotion(sentiment: str, text: str) -> str:
    if pd.isna(sentiment):
        return "none"
    lowered = text.lower()
    if any(term in lowered for term in ("worst", "angry", "furious")):
        return "anger"
    if any(term in lowered for term in ("worry", "concern", "allergy", "mold", "spoiled")):
        return "concern"
    if any(term in lowered for term in ("confused", "misleading", "different than")):
        return "confusion"
    if sentiment == "positive":
        return "delight" if any(term in lowered for term in ("love", "amazing", "best", "favorite")) else "satisfaction"
    if sentiment == "negative":
        return "frustration" if any(term in lowered for term in ("again", "still", "waste")) else "disappointment"
    return "none"


def classify_aspect(text: str) -> str:
    lowered = text.lower()
    scores = {
        aspect: sum(1 for term in terms if term in lowered)
        for aspect, terms in ASPECT_RULES.items()
    }
    top_aspect, top_score = max(scores.items(), key=lambda item: item[1])
    return top_aspect if top_score else "overall_quality"


def enrich_reviews(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    data = frame.copy()
    data["review_text_clean"] = data["review_text"].map(clean_text)
    inferred_sentiment = pd.Series([
        infer_sentiment(rating, text)
        for rating, text in zip(data["rating"], data["review_text_clean"])
    ], index=data.index)
    has_text_or_rating = data["review_text_clean"].str.strip().ne("") | data["rating"].notna()
    if "sentiment" in data:
        existing = data["sentiment"].fillna("").astype(str).str.lower()
        keep_existing = data["rating"].isna() & existing.isin({"positive", "negative", "mixed", "neutral"})
        data["sentiment"] = existing.where(keep_existing, inferred_sentiment)
        has_text_or_rating = has_text_or_rating | existing.isin({"positive", "negative", "mixed", "neutral"})
    else:
        data["sentiment"] = inferred_sentiment
    data.loc[~has_text_or_rating, "sentiment"] = pd.NA

    inferred_emotion = pd.Series([
        infer_emotion(sentiment, text)
        for sentiment, text in zip(data["sentiment"], data["review_text_clean"])
    ], index=data.index)
    if "emotion" in data:
        existing = data["emotion"].fillna("").astype(str)
        data["emotion"] = existing.where(existing.str.strip().ne(""), inferred_emotion)
    else:
        data["emotion"] = inferred_emotion

    has_text = data["review_text_clean"].str.strip().ne("")
    inferred_aspect = data["review_text_clean"].map(classify_aspect)
    if "product_aspect" in data:
        existing = data["product_aspect"].fillna("").astype(str)
        data["product_aspect"] = existing.where(existing.isin(ASPECT_LABELS), inferred_aspect)
    else:
        data["product_aspect"] = inferred_aspect
    data.loc[~has_text & data["product_aspect"].eq("overall_quality"), "product_aspect"] = pd.NA
    data["aspect_label"] = data["product_aspect"].map(ASPECT_LABELS)
    inferred_need = data["product_aspect"].map(NEED_MAP)
    if "customer_need" in data:
        existing = data["customer_need"].fillna("").astype(str)
        data["customer_need"] = existing.where(existing.str.strip().ne(""), inferred_need)
    else:
        data["customer_need"] = inferred_need
    if "analysis_method" not in data:
        data["analysis_method"] = "rating + explainable food-domain rules"
    return data


def negative_signals(frame: pd.DataFrame) -> pd.DataFrame:
    """Return 1-2 star reviews, plus explicitly negative reviews without ratings."""
    ratings = pd.to_numeric(frame["rating"], errors="coerce") if "rating" in frame else pd.Series(index=frame.index, dtype=float)
    sentiment = frame["sentiment"] if "sentiment" in frame else pd.Series(index=frame.index, dtype="string")
    low_rating = ratings.between(1, 2, inclusive="both")
    unrated_negative = ratings.isna() & sentiment.eq("negative")
    mask = low_rating | unrated_negative
    return frame[mask.fillna(False)]


def improvement_signals(frame: pd.DataFrame) -> pd.DataFrame:
    """Return every rated Review below 5 stars, plus unrated negative Reviews."""
    ratings = pd.to_numeric(frame["rating"], errors="coerce") if "rating" in frame else pd.Series(index=frame.index, dtype=float)
    sentiment = frame["sentiment"] if "sentiment" in frame else pd.Series(index=frame.index, dtype="string")
    below_five = ratings.between(1, 5, inclusive="left")
    unrated_negative = ratings.isna() & sentiment.eq("negative")
    return frame[(below_five | unrated_negative).fillna(False)]


def top_negative_keywords(
    frame: pd.DataFrame, limit: int = 15, include_non_five: bool = False
) -> pd.DataFrame:
    negative = improvement_signals(frame) if include_non_five else negative_signals(frame)
    counter: Counter[str] = Counter()
    for text in negative["review_text_clean"]:
        tokens = _tokens(text)
        counter.update(token for token in tokens if token in PAIN_TERMS)
        for left, right in zip(tokens, tokens[1:]):
            if left in PAIN_TERMS or right in PAIN_TERMS:
                if left not in STOP_WORDS or left in {"not", "too"}:
                    if right not in STOP_WORDS or right in {"not", "too"}:
                        counter[f"{left} {right}"] += 1
    rows = [{"keyword": key, "mentions": value} for key, value in counter.most_common(limit)]
    return pd.DataFrame(rows)


def aspect_summary(frame: pd.DataFrame, include_non_five: bool = False) -> pd.DataFrame:
    signals = improvement_signals(frame) if include_non_five else negative_signals(frame)
    negative = signals.dropna(subset=["product_aspect", "aspect_label"])
    total = max(len(frame), 1)
    grouped = (
        negative.groupby(["product_aspect", "aspect_label"], dropna=False)
        .agg(review_count=("review_id", "nunique"), average_rating=("rating", "mean"))
        .reset_index()
    )
    grouped["mention_rate"] = grouped["review_count"] / total
    return grouped.sort_values("review_count", ascending=False)


def need_summary(
    frame: pd.DataFrame, language: str = "en", include_non_five: bool = False
) -> pd.DataFrame:
    signals = improvement_signals(frame) if include_non_five else negative_signals(frame)
    negative = signals.dropna(subset=["product_aspect", "aspect_label"])
    total = max(len(frame), 1)
    grouped = (
        negative.groupby(["product_aspect", "aspect_label"], dropna=False)
        .agg(
            review_count=("review_id", "nunique"),
            product_count=("product_id", "nunique"),
            average_rating=("rating", "mean"),
            evidence_examples=(
                "review_text_clean",
                lambda values: [str(value).strip() for value in values if str(value).strip()][:3],
            ),
        )
        .reset_index()
    )
    grouped["mention_rate"] = grouped["review_count"] / total
    derived_needs = {
        aspect: _comment_driven_need(negative, aspect, language)
        for aspect in grouped["product_aspect"].dropna().unique()
    }
    grouped["customer_need"] = grouped["product_aspect"].map(derived_needs)
    return grouped.sort_values("review_count", ascending=False)


def _comment_driven_need(frame: pd.DataFrame, aspect: str, language: str) -> str:
    """Summarize a need as a concise aspect and the strongest complaint term."""
    texts = frame.loc[frame["product_aspect"].eq(aspect), "review_text_clean"].dropna().astype(str)
    pain_counter: Counter[str] = Counter()
    aspect_counter: Counter[str] = Counter()
    aspect_terms = ASPECT_RULES.get(aspect, set())
    for text in texts:
        tokens = _tokens(text)
        pain_counter.update(token for token in tokens if token in PAIN_TERMS)
        aspect_counter.update(token for token in tokens if token in aspect_terms)
    if pain_counter:
        signal = pain_counter.most_common(1)[0][0]
    elif aspect_counter:
        signal = aspect_counter.most_common(1)[0][0]
    else:
        fallback: Counter[str] = Counter()
        for text in texts:
            fallback.update(token for token in _tokens(text) if token not in STOP_WORDS)
        signal = fallback.most_common(1)[0][0] if fallback else ""
    label = NEED_ASPECT_LABELS_ZH.get(aspect, aspect) if language == "zh" else NEED_ASPECT_LABELS_EN.get(aspect, aspect)
    return f"{label}:{signal}" if signal else label


def recommendation_summary(frame: pd.DataFrame, language: str = "en") -> pd.DataFrame:
    summary = aspect_summary(frame)
    if summary.empty:
        return summary
    max_mentions = max(summary["review_count"].max(), 1)
    rows = []
    actions = ACTION_MAP if language == "zh" else ACTION_MAP_EN
    aspect_labels = ASPECT_LABELS_ZH if language == "zh" else ASPECT_LABELS
    for row in summary.itertuples(index=False):
        owner, action = actions[row.product_aspect]
        reach = row.review_count / max_mentions
        rating_impact = 0.0 if pd.isna(row.average_rating) else max(0.0, (3.0 - row.average_rating) / 2.0)
        priority_score = round(100 * (0.65 * reach + 0.35 * rating_impact))
        priority = "P0" if priority_score >= 75 else "P1" if priority_score >= 50 else "P2"
        rows.append(
            {
                "priority": priority,
                "priority_score": priority_score,
                "product_aspect": row.product_aspect,
                "issue": aspect_labels[row.product_aspect],
                "owner": OWNER_LABELS_ZH.get(owner, owner) if language == "zh" else owner,
                "recommendation": action,
                "review_count": row.review_count,
                "mention_rate": row.mention_rate,
                "average_rating": row.average_rating,
                "validation": _validation_plan(row.product_aspect, language),
            }
        )
    return pd.DataFrame(rows).sort_values(["priority_score", "review_count"], ascending=False)


def _validation_plan(aspect: str, language: str = "en") -> str:
    plans_zh = {
        "taste": "对比改版前后该主题负面提及率，并进行盲测。",
        "texture": "按批次抽检并追踪四周口感负面提及率。",
        "freshness": "核查库存批次，追踪过期/不新鲜提及率。",
        "packaging": "执行跌落与挤压测试，追踪破损评论占比。",
        "portion_size": "更新页面后监测尺寸、数量误解提及率。",
        "price_value": "A/B测试价值表达，监测价格负面提及率。",
        "ingredients": "合规审核后抽检Listing信息完整率。",
        "delivery": "按地区、月份对比融化和破损提及率。",
        "overall_quality": "人工复核代表性评论并建立二级问题标签。",
    }
    plans_en = {
        "taste": "Compare the issue's negative mention rate before and after the change, supported by a blind taste test.",
        "texture": "Inspect batches and track texture-related negative mentions for four weeks.",
        "freshness": "Audit inventory lots and track expired or not-fresh mentions.",
        "packaging": "Run drop and compression tests, then track the share of damage-related reviews.",
        "portion_size": "After updating the listing, track size and quantity misunderstanding mentions.",
        "price_value": "A/B test the value proposition and monitor price-related negative mentions.",
        "ingredients": "After compliance review, sample listings for disclosure completeness.",
        "delivery": "Compare melting and damage mentions by region and month.",
        "overall_quality": "Manually review representative comments and establish second-level issue labels.",
    }
    return (plans_zh if language == "zh" else plans_en)[aspect]


def format_percent(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "0.0%"
    return f"{value:.1%}"
