import io

import pandas as pd

from src.analytics import (
    aspect_summary,
    enrich_reviews,
    improvement_signals,
    negative_signals,
    need_summary,
    recommendation_summary,
)
from src.data_loader import load_uploaded_review_files, load_uploaded_reviews, normalize_schema


def test_normalize_and_enrich_review_schema() -> None:
    source = pd.DataFrame(
        {
            "ProductId": ["B000TEST01"],
            "UserId": ["USER01"],
            "Score": [2],
            "Time": [1_600_000_000],
            "Summary": ["Damaged"],
            "Text": ["The package arrived broken and the bag was open."],
        }
    )
    normalized = normalize_schema(source, "test")
    enriched = enrich_reviews(normalized)

    assert enriched.loc[0, "sentiment"] == "negative"
    assert enriched.loc[0, "product_aspect"] == "packaging"
    assert enriched.loc[0, "review_id"]


def test_recommendations_include_evidence_metrics() -> None:
    source = pd.DataFrame(
        {
            "ProductId": ["B000TEST01", "B000TEST02"],
            "Score": [1, 2],
            "Time": [1_600_000_000, 1_600_000_001],
            "Text": ["The package was broken.", "The seal was open."],
        }
    )
    enriched = enrich_reviews(normalize_schema(source, "test"))
    recommendations = recommendation_summary(enriched)

    assert recommendations.iloc[0]["issue"] == "Packaging"
    assert recommendations.iloc[0]["review_count"] == 2
    assert recommendations.iloc[0]["owner"] == "Packaging"


def test_missing_columns_are_accepted_without_fabricating_values() -> None:
    source = pd.DataFrame({"Score": [5, 2], "unmapped_field": ["a", "b"]})

    normalized = normalize_schema(source, "partial")
    enriched = enrich_reviews(normalized)

    assert len(enriched) == 2
    assert enriched["product_id"].isna().all()
    assert enriched["review_text"].eq("").all()
    assert enriched["rating"].tolist() == [5, 2]
    assert enriched["product_aspect"].isna().all()


def test_recommendations_support_bilingual_business_output() -> None:
    source = pd.DataFrame(
        {
            "Score": [1],
            "Text": ["The package arrived broken and the seal was open."],
        }
    )
    enriched = enrich_reviews(normalize_schema(source, "test"))

    english = recommendation_summary(enriched, language="en").iloc[0]
    chinese = recommendation_summary(enriched, language="zh").iloc[0]

    assert english["issue"] == "Packaging"
    assert "transit" in english["recommendation"]
    assert chinese["issue"] == "包装"
    assert chinese["owner"] == "包装"


def test_uploaded_file_keeps_its_own_lineage() -> None:
    class UploadedCsv(io.BytesIO):
        name = "merchant_reviews.csv"

    uploaded = UploadedCsv(b"Score,Text\n1,The package was broken\n")
    loaded = load_uploaded_reviews(uploaded)
    enriched = enrich_reviews(loaded)

    assert enriched["source_name"].unique().tolist() == ["Uploaded: merchant_reviews.csv"]
    assert enriched.loc[0, "product_aspect"] == "packaging"


def test_multiple_uploaded_datasets_are_combined_with_lineage() -> None:
    class UploadedCsv(io.BytesIO):
        def __init__(self, name: str, payload: bytes) -> None:
            super().__init__(payload)
            self.name = name

    files = [
        UploadedCsv("positive.csv", b"Score,Text\n5,Great taste\n"),
        UploadedCsv("negative.csv", b"Score,Text\n1,Gross flavor\n"),
    ]
    combined = enrich_reviews(load_uploaded_review_files(files))

    assert len(combined) == 2
    assert set(combined["source_name"]) == {"Uploaded: positive.csv", "Uploaded: negative.csv"}
    assert combined["sentiment"].tolist() == ["positive", "negative"]


def test_rating_buckets_drive_sentiment_and_negative_evidence() -> None:
    source = pd.DataFrame(
        {
            "Score": [1, 2, 3, 4, 5],
            "Text": [
                "Excellent product.",
                "Good product.",
                "Terrible product.",
                "Awful product.",
                "Worst product.",
            ],
        }
    )
    enriched = enrich_reviews(normalize_schema(source, "test"))
    negative = negative_signals(enriched)
    improvement = improvement_signals(enriched)

    assert enriched["sentiment"].tolist() == ["negative", "negative", "neutral", "neutral", "positive"]
    assert negative["rating"].tolist() == [1, 2]
    assert improvement["rating"].tolist() == [1, 2, 3, 4]


def test_pain_points_include_one_to_four_stars_but_recommendations_do_not() -> None:
    source = pd.DataFrame(
        {
            "Score": [1, 2, 3, 4, 5],
            "Text": ["bad taste", "bad taste", "bad taste", "bad taste", "great taste"],
        }
    )
    enriched = enrich_reviews(normalize_schema(source, "test"))

    pain = aspect_summary(enriched, include_non_five=True)
    recommendation_scope = aspect_summary(enriched)

    assert pain["review_count"].sum() == 4
    assert recommendation_scope["review_count"].sum() == 2


def test_customer_needs_are_derived_from_current_review_language() -> None:
    source = pd.DataFrame(
        {
            "ProductId": ["B000TEST01", "B000TEST01"],
            "Score": [1, 2],
            "Text": ["The package arrived broken.", "The seal was open."],
        }
    )
    enriched = enrich_reviews(normalize_schema(source, "test"))
    needs = need_summary(enriched, language="zh")

    assert len(needs) == 1
    assert needs.iloc[0]["customer_need"].startswith("包装:")
    assert "broken" in needs.iloc[0]["customer_need"]
    assert "差评" not in needs.iloc[0]["customer_need"]


def test_customer_need_summary_aggregates_all_negative_evidence() -> None:
    source = pd.DataFrame(
        {
            "ProductId": ["A", "B", "C"],
            "Score": [1, 2, 5],
            "Text": ["The package was broken.", "The seal was open.", "Great taste."],
        }
    )
    enriched = enrich_reviews(normalize_schema(source, "test"))
    needs = need_summary(enriched, language="en")
    packaging = needs[needs["product_aspect"] == "packaging"].iloc[0]

    assert packaging["review_count"] == 2
    assert packaging["product_count"] == 2
    assert len(packaging["evidence_examples"]) == 2
    assert packaging["average_rating"] == 1.5


def test_customer_need_uses_only_review_complaint_points() -> None:
    source = pd.DataFrame(
        {
            "Score": [1, 2],
            "Text": ["The flavor is bland.", "It is far too sweet."],
        }
    )
    enriched = enrich_reviews(normalize_schema(source, "test"))
    enriched["customer_need"] = "Balanced flavor that matches the product description"
    need = need_summary(enriched, language="en").iloc[0]["customer_need"]

    assert need == "Taste:bland"
    assert "complaints" not in need
    assert "Balanced flavor" not in need


def test_customer_voice_uses_one_dominant_complaint_term() -> None:
    source = pd.DataFrame(
        {
            "Score": [1],
            "Text": [
                "Not good. The Carmel part is grainy powder and the gummy is gross; "
                "it is egg flavored, not custard flavored."
            ],
        }
    )
    enriched = enrich_reviews(normalize_schema(source, "test"))

    assert need_summary(enriched, language="zh").iloc[0]["customer_need"] == "口味:gross"
