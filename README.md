# Amazon Consumer Insight Agent Demo

A Streamlit portfolio demo that turns real Amazon food reviews into evidence-backed consumer insights, pain points, needs, and product recommendations.

## Data policy

The dashboard is upload-driven. It does not load a bundled or fallback dataset: pain points, customer needs, recommendations, metrics, and evidence are calculated only from the file currently uploaded in the sidebar.

The project does not generate synthetic reviews. Raw and processed review files are excluded from Git.

An optional preparation utility can download the public Stanford SNAP Amazon Fine Foods research dataset and create a local Parquet file for manual upload. The dashboard never loads that file automatically.

Source: [Stanford SNAP Amazon Fine Foods](https://snap.stanford.edu/data/web-FineFoods.html)

The source contains 568,454 Amazon food reviews from October 1999 through October 2012. It is suitable for a historical portfolio demo, not for claims about current Amazon market conditions.

## Setup

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

After the app starts, upload one or more CSV, JSON, JSONL, or Parquet Review files in the sidebar. The files are normalized, combined, and deduplicated by `review_id`; every row retains its uploaded-file lineage. For an optional historical test file, run `python -m scripts.prepare_finefoods --limit 30000` and upload the generated Parquet file manually.

The active uploaded dataset is kept in the Streamlit session, so switching between English and Chinese does not require re-uploading the file. Pain-point, customer-voice, and recommendation views retain a complete table of all negative Review evidence.

Rating sentiment follows the Amazon operations rule used by this demo: 5 stars are positive, 3-4 stars are mid-rating, and 1-2 stars are negative. Pain Point Analysis and Customer Voice use all 1-4 star Reviews; AI Recommendation and the negative-signal KPI remain restricted to 1-2 star Reviews.

## Supported input schemas

The sidebar accepts CSV, JSON, JSONL, and Parquet files. The loader recognizes common fields from:

- Stanford Amazon Fine Foods
- Amazon Reviews 2018
- Amazon Reviews 2023

Partial schemas are supported. The file is imported even when product/ASIN, rating, date, or review text is absent. Filters, metrics, charts, and analysis sections are rendered only when their corresponding field contains valid data; missing fields are never filled with fabricated business values.

Common English and Chinese aliases such as `ASIN`, `商品ID`, `Score`, `评分`, `Text`, `评论内容`, `date`, and `评论时间` are normalized automatically.

## Architecture

```text
app.py                  Streamlit presentation layer
src/data_loader.py      Schema normalization and lineage IDs
src/analytics.py        Sentiment, pain-point, need and action baseline
src/config.py           Taxonomy, paths and visual configuration
scripts/                Reproducible real-data preparation
tests/                  Core analysis contract tests
```

The baseline is deliberately labeled in the UI: it uses rating signals and transparent food-domain rules. Structured outputs from the Review Analysis Agent can replace or enrich these fields without changing the dashboard contract.

## Tests

```powershell
python -m pytest
```
