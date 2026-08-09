from __future__ import annotations

import hashlib

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analytics import (
    aspect_summary,
    enrich_reviews,
    format_percent,
    improvement_signals,
    need_summary,
    recommendation_summary,
    negative_signals,
    top_negative_keywords,
)
from src.config import ASPECT_LABELS, ASPECT_LABELS_ZH, SENTIMENT_COLORS
from src.data_loader import load_uploaded_review_files


ANALYSIS_VERSION = "rating-buckets-v2"
DATASET_WIDGET_KEYS = (
    "rating_filter",
    "sentiment_filter",
    "product_filter",
    "review_date_filter",
)


st.set_page_config(
    page_title="Amazon Consumer Insight Agent",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="auto",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #1E2A2A;
        --muted: #63706E;
        --line: #D8DDD8;
        --surface: #FFFFFF;
        --accent: #D85D3F;
        --teal: #2A9D8F;
        --gold: #E9C46A;
    }
    .stApp { background: #F7F8F6; color: var(--ink); }
    .stAppDeployButton, #MainMenu { display: none !important; }
    [data-testid="stSidebar"] { background: #ECEFEA; border-right: 1px solid var(--line); }
    [data-testid="stHeader"] { background: rgba(247, 248, 246, 0.92); }
    h1, h2, h3 { letter-spacing: 0; color: var(--ink); }
    h1 { font-size: 2rem !important; line-height: 1.15 !important; margin-bottom: .25rem !important; }
    h2 { font-size: 1.25rem !important; margin-top: .8rem !important; }
    h3 { font-size: 1rem !important; }
    [data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 6px;
        padding: 14px 16px;
        min-height: 112px;
    }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stMetricValue"] { font-size: 1.8rem; color: var(--ink); }
    .eyebrow { color: var(--accent); font-size: .76rem; font-weight: 700; text-transform: uppercase; }
    .subtle { color: var(--muted); font-size: .92rem; margin-bottom: 1.2rem; }
    .priority-p0 { color: #A33A2B; font-weight: 700; }
    .priority-p1 { color: #9A6812; font-weight: 700; }
    .priority-p2 { color: #47635F; font-weight: 700; }
    .review-meta { color: var(--muted); font-size: .78rem; margin-bottom: .3rem; }
    div[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 6px; }
    div[data-baseweb="tab-list"] { gap: 1.4rem; border-bottom: 1px solid var(--line); }
    button[data-baseweb="tab"] { padding-left: 0; padding-right: 0; }
    .st-key-language_switcher {
        position: fixed;
        top: .7rem;
        right: 1.1rem;
        z-index: 1000000;
        width: 146px;
        padding: 4px;
        border: 1px solid rgba(42, 157, 143, .28);
        border-radius: 8px;
        background: rgba(255, 255, 255, .94);
        box-shadow: 0 8px 24px rgba(30, 42, 42, .12);
        backdrop-filter: blur(12px);
    }
    .st-key-language_switcher div[data-testid="stSegmentedControl"] {
        width: 100%;
        white-space: nowrap;
    }
    .st-key-language_switcher [role="radiogroup"] {
        display: grid !important;
        grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
        gap: 3px !important;
        width: 100% !important;
    }
    .st-key-language_switcher [role="radio"],
    .st-key-language_switcher [role="radiogroup"] > label {
        min-width: 0 !important;
        border: 0 !important;
        border-radius: 5px !important;
        transition: transform .18s ease, background-color .18s ease, color .18s ease, box-shadow .18s ease !important;
    }
    .st-key-language_switcher [role="radio"]:hover,
    .st-key-language_switcher [role="radiogroup"] > label:hover {
        transform: translateY(-1px);
        background: #F1F5F2 !important;
        box-shadow: 0 3px 8px rgba(30, 42, 42, .09);
    }
    .st-key-language_switcher [role="radio"][aria-checked="true"],
    .st-key-language_switcher [role="radiogroup"] > label:has(input:checked) {
        color: #FFFFFF !important;
        background: var(--teal) !important;
        box-shadow: 0 3px 10px rgba(42, 157, 143, .28);
    }
    @media (max-width: 700px) {
        .st-key-language_switcher {
            top: .55rem;
            right: .65rem;
            width: 132px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


UI_TEXT = {
    "en": {
        "data_workspace": "Data workspace",
        "consumer_insight": "Consumer Insight",
        "review_dataset": "Review datasets",
        "analysis_scope": "Analysis scope",
        "rating": "Rating",
        "sentiment": "Sentiment",
        "top_asins": "Top product ASINs",
        "review_date": "Review date",
        "scope": "Scope",
        "reviews_lower": "reviews",
        "detected_fields": "Detected fields",
        "review_text": "review text",
        "date": "date",
        "none": "none",
        "source": "Source",
        "analysis_caption": "Analysis uses only the currently uploaded datasets; Agent JSON compatible",
        "uploaded_only": "All insights below are calculated from the current uploaded datasets",
        "dataset_count": "{count} datasets",
        "title": "What customers value, where products fail, and what to do next",
        "subtitle": "Uploaded Amazon Reviews translated into evidence-backed product and operations decisions.",
        "no_match": "No reviews match the current analysis scope.",
        "tab_overview": "Overview Dashboard",
        "tab_pain": "Pain Point Analysis",
        "tab_voice": "Customer Voice",
        "tab_recommendation": "AI Recommendation",
        "reviews": "Reviews",
        "average_rating": "Average rating",
        "negative_signal": "Negative signal",
        "products": "Products",
        "sentiment_mix": "Sentiment mix",
        "rating_distribution": "Rating distribution",
        "review_trend": "Review trend",
        "negative_issue_categories": "Negative issue categories",
        "negative_reviews": "Negative reviews",
        "below_five_issue_categories": "Issue categories below 5 stars",
        "below_five_reviews": "Reviews below 5 stars",
        "avg_rating": "Avg rating",
        "frequent_negative_terms": "Frequent issue terms",
        "mentions": "Mentions",
        "issue_evidence": "Issue evidence",
        "issue_category": "Issue category",
        "high_frequency_needs": "High-frequency consumer needs",
        "needs_from_comments": "Consumer needs are synthesized from the current product's Reviews below 5 stars.",
        "need_signals": "Need signals",
        "consumer_need": "Consumer need",
        "mention_rate": "Mention rate",
        "voice_of_customer": "Voice of customer",
        "priority": "Priority",
        "consumer_issue": "Consumer issue",
        "owner": "Owner",
        "recommended_action": "Recommended action",
        "evidence": "Evidence",
        "recommendation_brief": "Recommendation brief",
        "recommendation": "Recommendation",
        "validation": "Validation",
        "priority_score": "Priority score",
        "evidence_reviews": "Evidence reviews",
        "recommendation_caption": "Evidence-based baseline recommendations. Final product and compliance decisions require human review.",
        "no_evidence": "No evidence in the current filter scope.",
        "all_negative_evidence": "All negative review evidence",
        "negative_review_count": "negative reviews retained",
        "all_below_five_evidence": "All Review evidence below 5 stars",
        "below_five_review_count": "Reviews below 5 stars retained",
        "need_basis": "Synthesized from {reviews} Reviews below 5 stars across {products} products.",
        "need_basis_scope": "Synthesized from {reviews} Reviews below 5 stars in the current scope.",
        "evidence_examples": "Representative evidence",
        "clear_dataset": "Clear current dataset",
        "no_text_pain": "No review text column detected. Pain-point analysis is not displayed.",
        "no_negative": "No negative signals in the current filter scope.",
        "no_below_five": "No Reviews below 5 stars in the current filter scope.",
        "no_keywords": "No negative keywords in the current filter scope.",
        "no_text_voice": "No review text column detected. Customer Voice is not displayed.",
        "no_needs": "No consumer needs in the current filter scope.",
        "no_text_recommendation": "No review text column detected. Recommendations are not displayed.",
        "no_recommendation": "No recommendation can be generated for the current filter scope.",
        "stars": "stars",
        "review": "Review",
        "empty_title": "Upload Review data to start",
        "empty_subtitle": "Pain points, customer needs, and AI recommendations are generated only from the files you upload.",
        "empty_info": "CSV, JSONL, and Parquet are supported. Missing fields are accepted; analyses that depend on them stay hidden.",
        "load_error": "Unable to load review data",
        "sentiment_positive": "Positive",
        "sentiment_mixed": "Mixed",
        "sentiment_neutral": "Mid-rating",
        "sentiment_negative": "Negative",
    },
    "zh": {
        "data_workspace": "数据工作区",
        "consumer_insight": "消费者洞察",
        "review_dataset": "Review 数据集",
        "analysis_scope": "分析范围",
        "rating": "评分",
        "sentiment": "情感",
        "top_asins": "热门商品 ASIN",
        "review_date": "Review 日期",
        "scope": "当前范围",
        "reviews_lower": "条 Review",
        "detected_fields": "已识别字段",
        "review_text": "Review 正文",
        "date": "日期",
        "none": "无",
        "source": "数据来源",
        "analysis_caption": "分析仅使用当前上传的数据集；兼容 Agent JSON 输出",
        "uploaded_only": "以下所有洞察根据当前上传的全部数据集计算",
        "dataset_count": "{count} 个数据集",
        "title": "消费者重视什么、产品问题在哪里、下一步如何行动",
        "subtitle": "将上传的 Amazon Review 转化为有证据支持的产品与运营决策。",
        "no_match": "当前分析范围内没有匹配的 Review。",
        "tab_overview": "总览看板",
        "tab_pain": "痛点分析",
        "tab_voice": "消费者之声",
        "tab_recommendation": "AI 优化建议",
        "reviews": "Review 数量",
        "average_rating": "平均评分",
        "negative_signal": "负面信号占比",
        "products": "商品数量",
        "sentiment_mix": "情感分布",
        "rating_distribution": "评分分布",
        "review_trend": "Review 趋势",
        "negative_issue_categories": "负面问题分类",
        "negative_reviews": "负面 Review",
        "below_five_issue_categories": "低于5分的问题分类",
        "below_five_reviews": "低于5分的 Review",
        "avg_rating": "平均评分",
        "frequent_negative_terms": "高频问题词",
        "mentions": "提及次数",
        "issue_evidence": "问题证据",
        "issue_category": "问题分类",
        "high_frequency_needs": "高频消费者需求",
        "needs_from_comments": "消费者需求由当前产品所有低于5分的 Review 综合归纳。",
        "need_signals": "需求信号",
        "consumer_need": "消费者需求",
        "mention_rate": "提及率",
        "voice_of_customer": "消费者原声",
        "priority": "优先级",
        "consumer_issue": "消费者问题",
        "owner": "责任团队",
        "recommended_action": "建议行动",
        "evidence": "证据数",
        "recommendation_brief": "建议简报",
        "recommendation": "优化建议",
        "validation": "验证方式",
        "priority_score": "优先级评分",
        "evidence_reviews": "证据 Review",
        "recommendation_caption": "以下为基于证据的基线建议，最终产品与合规决策需由人工复核。",
        "no_evidence": "当前筛选范围内没有相关证据。",
        "all_negative_evidence": "全部差评评论证据",
        "negative_review_count": "条差评评论已保留",
        "all_below_five_evidence": "全部低于5分的 Review 证据",
        "below_five_review_count": "条低于5分的 Review 已保留",
        "need_basis": "综合 {reviews} 条低于5分的 Review、覆盖 {products} 个商品后得到。",
        "need_basis_scope": "综合当前范围内 {reviews} 条低于5分的 Review 后得到。",
        "evidence_examples": "代表性评论证据",
        "clear_dataset": "清除当前数据集",
        "no_text_pain": "未识别到 Review 正文列，因此不展示痛点分析。",
        "no_negative": "当前筛选范围内没有负面信号。",
        "no_below_five": "当前筛选范围内没有低于5分的 Review。",
        "no_keywords": "当前筛选范围内没有负面关键词。",
        "no_text_voice": "未识别到 Review 正文列，因此不展示消费者之声。",
        "no_needs": "当前筛选范围内没有消费者需求信号。",
        "no_text_recommendation": "未识别到 Review 正文列，因此不展示优化建议。",
        "no_recommendation": "当前筛选范围内无法生成优化建议。",
        "stars": "星",
        "review": "Review",
        "empty_title": "请上传 Review 数据开始分析",
        "empty_subtitle": "痛点、消费者需求和 AI 优化建议只会根据你上传的全部文件生成。",
        "empty_info": "支持 CSV、JSONL 和 Parquet。允许缺少字段；依赖缺失字段的分析将自动隐藏。",
        "load_error": "无法加载 Review 数据",
        "sentiment_positive": "正面",
        "sentiment_mixed": "混合",
        "sentiment_neutral": "中评",
        "sentiment_negative": "负面",
    },
}

with st.container(key="language_switcher"):
    selected_language = st.segmented_control(
        "Language / 语言",
        options=["EN", "中文"],
        default="EN",
        label_visibility="collapsed",
        key="ui_language_selector",
    )
LANGUAGE = "zh" if selected_language == "中文" else "en"


def tr(key: str) -> str:
    return UI_TEXT[LANGUAGE][key]


def format_dataset_label(names: tuple[str, ...]) -> str:
    if len(names) <= 2:
        return ", ".join(names)
    return tr("dataset_count").format(count=len(names))


def localized_aspect_label(aspect: str) -> str:
    labels = ASPECT_LABELS_ZH if LANGUAGE == "zh" else ASPECT_LABELS
    return labels.get(aspect, aspect)


def clear_dataset_dependent_state() -> None:
    for key in DATASET_WIDGET_KEYS:
        st.session_state.pop(key, None)


@st.cache_data(show_spinner=False)
def get_uploaded_datasets(
    datasets: tuple[tuple[str, bytes], ...], analysis_version: str
) -> pd.DataFrame:
    del analysis_version
    class Uploaded:
        def __init__(self, raw: bytes, name: str) -> None:
            self._raw = raw
            self.name = name

        def read(self) -> bytes:
            return self._raw

    uploaded_files = [Uploaded(payload, filename) for filename, payload in datasets]
    return enrich_reviews(load_uploaded_review_files(uploaded_files))


def has_values(frame: pd.DataFrame, column: str) -> bool:
    if column not in frame or frame.empty:
        return False
    values = frame[column]
    if pd.api.types.is_string_dtype(values.dtype) or values.dtype == object:
        return values.fillna("").astype(str).str.strip().ne("").any()
    return values.notna().any()


def chart_layout(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=28, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1E2A2A", size=12),
        legend_title_text="",
        hoverlabel=dict(bgcolor="white"),
    )
    fig.update_xaxes(gridcolor="#E4E8E4", zeroline=False)
    fig.update_yaxes(gridcolor="#E4E8E4", zeroline=False)
    return fig


def render_empty_state() -> None:
    st.markdown('<div class="eyebrow">Amazon Consumer Insight Agent</div>', unsafe_allow_html=True)
    st.title(tr("empty_title"))
    st.markdown(
        f'<div class="subtle">{tr("empty_subtitle")}</div>',
        unsafe_allow_html=True,
    )
    st.info(tr("empty_info"))


def apply_filters(data: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.markdown(f"### {tr('analysis_scope')}")
    filtered = data.copy()
    if has_values(data, "rating"):
        rating_options = sorted(data["rating"].dropna().astype(int).unique().tolist())
        ratings = st.sidebar.multiselect(
            tr("rating"), rating_options, default=rating_options, key="rating_filter"
        )
        rating_mask = filtered["rating"].fillna(-1).astype(int).isin(ratings)
        if set(ratings) == set(rating_options):
            rating_mask = rating_mask | filtered["rating"].isna()
        filtered = filtered[rating_mask]

    if has_values(data, "sentiment"):
        preferred_order = ["positive", "mixed", "neutral", "negative"]
        observed = set(data["sentiment"].dropna().astype(str))
        sentiment_options = [value for value in preferred_order if value in observed]
        sentiments = st.sidebar.multiselect(
            tr("sentiment"),
            sentiment_options,
            default=sentiment_options,
            format_func=lambda value: tr(f"sentiment_{value}"),
            key="sentiment_filter",
        )
        sentiment_mask = filtered["sentiment"].isin(sentiments)
        if set(sentiments) == set(sentiment_options):
            sentiment_mask = sentiment_mask | filtered["sentiment"].isna()
        filtered = filtered[sentiment_mask]

    if has_values(data, "product_id"):
        product_counts = data["product_id"].dropna().value_counts().head(100)
        product_options = product_counts.index.astype(str).tolist()
        products = st.sidebar.multiselect(tr("top_asins"), product_options, key="product_filter")
        if products:
            filtered = filtered[filtered["product_id"].isin(products)]

    valid_dates = data["review_date"].dropna()
    if not valid_dates.empty:
        min_date = valid_dates.min().date()
        max_date = valid_dates.max().date()
        selected = st.sidebar.date_input(
            tr("review_date"),
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="review_date_filter",
        )
        if isinstance(selected, tuple) and len(selected) == 2:
            start, end = pd.Timestamp(selected[0]), pd.Timestamp(selected[1])
            filtered = filtered[filtered["review_date"].between(start, end + pd.Timedelta(days=1), inclusive="left")]
    return filtered


def show_review_evidence(
    frame: pd.DataFrame,
    aspect: str,
    limit: int = 5,
    include_non_five: bool = False,
) -> None:
    evidence = improvement_signals(frame) if include_non_five else negative_signals(frame)
    evidence = evidence[evidence["product_aspect"] == aspect]
    evidence = evidence.sort_values(["helpful_votes", "rating"], ascending=[False, True]).head(limit)
    if evidence.empty:
        st.caption(tr("no_evidence"))
        return
    for row in evidence.itertuples(index=False):
        meta = []
        if pd.notna(row.product_id) and str(row.product_id).strip():
            meta.append(str(row.product_id))
        if pd.notna(row.rating):
            meta.append(f"{row.rating:.0f} {tr('stars')}")
        if pd.notna(row.review_date):
            meta.append(row.review_date.strftime("%Y-%m-%d"))
        meta.append(f"{tr('review')} {row.review_id}")
        with st.container(border=True):
            st.markdown(
                f'<div class="review-meta">{" · ".join(meta)}</div>',
                unsafe_allow_html=True,
            )
            if row.review_title:
                st.markdown(f"**{row.review_title}**")
            excerpt = row.review_text_clean[:520]
            st.write(excerpt + ("..." if len(row.review_text_clean) > 520 else ""))


def show_all_review_evidence(frame: pd.DataFrame, include_non_five: bool = False) -> None:
    evidence = (improvement_signals(frame) if include_non_five else negative_signals(frame)).copy()
    if evidence.empty:
        st.caption(tr("no_evidence"))
        return
    evidence["aspect_display"] = evidence["product_aspect"].map(localized_aspect_label)
    columns = ["review_id", "product_id", "rating", "review_date", "aspect_display", "review_title", "review_text_clean"]
    display = evidence[columns].sort_values(["rating", "review_date"], ascending=[True, False], na_position="last")
    display.columns = [
        tr("review"),
        "ASIN",
        tr("rating"),
        tr("review_date"),
        tr("issue_category"),
        "Title",
        "Review text",
    ]
    title_key = "all_below_five_evidence" if include_non_five else "all_negative_evidence"
    count_key = "below_five_review_count" if include_non_five else "negative_review_count"
    st.subheader(tr(title_key))
    st.caption(f"{len(display):,} {tr(count_key)}")
    st.dataframe(display, hide_index=True, width="stretch", height=520)


def render_overview(frame: pd.DataFrame) -> None:
    metrics = [(tr("reviews"), f"{len(frame):,}")]
    if has_values(frame, "rating"):
        metrics.append((tr("average_rating"), f"{frame['rating'].mean():.2f} / 5"))
    if has_values(frame, "sentiment"):
        negative_share = len(negative_signals(frame)) / max(frame["sentiment"].notna().sum(), 1)
        metrics.append((tr("negative_signal"), format_percent(negative_share)))
    if has_values(frame, "product_id"):
        metrics.append((tr("products"), f"{frame['product_id'].nunique():,}"))
    cols = st.columns(len(metrics))
    for column, (label, value) in zip(cols, metrics):
        column.metric(label, value)

    has_sentiment = has_values(frame, "sentiment")
    has_rating = has_values(frame, "rating")
    chart_columns = st.columns([1, 1.35], gap="large") if has_sentiment and has_rating else [st.container()]
    chart_index = 0
    if has_sentiment:
        with chart_columns[chart_index]:
            chart_index += 1
            st.subheader(tr("sentiment_mix"))
            sentiment = frame["sentiment"].dropna().value_counts().rename_axis("sentiment").reset_index(name="reviews")
            sentiment["sentiment_label"] = sentiment["sentiment"].map(
                lambda value: tr(f"sentiment_{value}")
            )
            fig = px.pie(
                sentiment,
                values="reviews",
                names="sentiment_label",
                hole=0.64,
                color="sentiment_label",
                color_discrete_map={
                    tr(f"sentiment_{key}"): color for key, color in SENTIMENT_COLORS.items()
                },
            )
            fig.update_traces(textposition="inside", textinfo="percent+label", sort=False)
            st.plotly_chart(chart_layout(fig), width="stretch", config={"displayModeBar": False})

    if has_rating:
        with chart_columns[chart_index]:
            st.subheader(tr("rating_distribution"))
            ratings = frame["rating"].dropna().astype(int).value_counts().sort_index().rename_axis("rating").reset_index(name="reviews")
            fig = px.bar(
                ratings,
                x="rating",
                y="reviews",
                text="reviews",
                color="rating",
                color_continuous_scale=["#D85D3F", "#E9C46A", "#2A9D8F"],
            )
            fig.update_layout(coloraxis_showscale=False)
            fig.update_traces(textposition="outside")
            fig.update_xaxes(title=tr("rating"))
            fig.update_yaxes(title=tr("reviews"))
            st.plotly_chart(chart_layout(fig), width="stretch", config={"displayModeBar": False})

    dated = frame.dropna(subset=["review_date"]).copy()
    if not dated.empty:
        st.subheader(tr("review_trend"))
        dated["month"] = dated["review_date"].dt.to_period("M").dt.to_timestamp()
        trend = dated.groupby("month").agg(reviews=("review_id", "nunique"), average_rating=("rating", "mean")).reset_index()
        fig = go.Figure()
        fig.add_bar(x=trend["month"], y=trend["reviews"], name=tr("reviews"), marker_color="#A8B7B2", yaxis="y")
        if has_rating:
            fig.add_scatter(x=trend["month"], y=trend["average_rating"], name=tr("average_rating"), line=dict(color="#D85D3F", width=3), yaxis="y2")
            fig.update_layout(yaxis2=dict(title=tr("average_rating"), overlaying="y", side="right", range=[1, 5], showgrid=False))
        fig.update_layout(yaxis=dict(title=tr("reviews")))
        st.plotly_chart(chart_layout(fig, 330), width="stretch", config={"displayModeBar": False})


def render_pain_points(frame: pd.DataFrame) -> None:
    if not has_values(frame, "review_text_clean"):
        st.info(tr("no_text_pain"))
        return
    issues = aspect_summary(frame, include_non_five=True)
    if not issues.empty:
        issues["aspect_display"] = issues["product_aspect"].map(localized_aspect_label)
    left, right = st.columns([1.2, 1], gap="large")
    with left:
        st.subheader(tr("below_five_issue_categories"))
        if issues.empty:
            st.info(tr("no_below_five"))
        else:
            plot_data = issues.sort_values("review_count")
            chart_options = dict(
                data_frame=plot_data,
                x="review_count",
                y="aspect_display",
                orientation="h",
                text="review_count",
            )
            if has_values(frame, "rating"):
                chart_options.update(
                    color="average_rating",
                    color_continuous_scale=["#D85D3F", "#E9C46A", "#2A9D8F"],
                )
            else:
                chart_options["color_discrete_sequence"] = ["#D85D3F"]
            fig = px.bar(**chart_options)
            if has_values(frame, "rating"):
                fig.update_layout(coloraxis_colorbar_title=tr("avg_rating"))
            fig.update_yaxes(title=None)
            fig.update_xaxes(title=tr("below_five_reviews"))
            st.plotly_chart(chart_layout(fig, 410), width="stretch", config={"displayModeBar": False})

    with right:
        st.subheader(tr("frequent_negative_terms"))
        keywords = top_negative_keywords(frame, include_non_five=True)
        if keywords.empty:
            st.info(tr("no_keywords"))
        else:
            fig = px.bar(
                keywords.sort_values("mentions"),
                x="mentions",
                y="keyword",
                orientation="h",
                text="mentions",
                color="mentions",
                color_continuous_scale=["#E9C46A", "#D85D3F"],
            )
            fig.update_layout(coloraxis_showscale=False)
            fig.update_yaxes(title=None)
            fig.update_xaxes(title=tr("mentions"))
            st.plotly_chart(chart_layout(fig, 410), width="stretch", config={"displayModeBar": False})

    if not issues.empty:
        st.subheader(tr("issue_evidence"))
        options = dict(zip(issues["aspect_display"], issues["product_aspect"]))
        selected_label = st.selectbox(tr("issue_category"), list(options))
        show_review_evidence(frame, options[selected_label], include_non_five=True)
    show_all_review_evidence(frame, include_non_five=True)


def render_customer_voice(frame: pd.DataFrame) -> None:
    if not has_values(frame, "review_text_clean"):
        st.info(tr("no_text_voice"))
        return
    needs = need_summary(frame, LANGUAGE, include_non_five=True)
    if needs.empty:
        st.info(tr("no_needs"))
        show_all_review_evidence(frame, include_non_five=True)
        return

    st.caption(tr("needs_from_comments"))
    left, right = st.columns([1.35, 1], gap="large")
    with left:
        st.subheader(tr("high_frequency_needs"))
        plot_data = needs.head(8).sort_values("review_count")
        chart_options = dict(
            data_frame=plot_data,
            x="review_count",
            y="customer_need",
            orientation="h",
            text="review_count",
        )
        if has_values(frame, "product_id"):
            chart_options.update(
                color="product_count",
                color_continuous_scale=["#E9C46A", "#2A9D8F", "#264653"],
            )
        else:
            chart_options["color_discrete_sequence"] = ["#2A9D8F"]
        fig = px.bar(**chart_options)
        if has_values(frame, "product_id"):
            fig.update_layout(coloraxis_colorbar_title=tr("products"))
        fig.update_yaxes(title=None)
        fig.update_xaxes(title=tr("below_five_reviews"))
        st.plotly_chart(chart_layout(fig, 430), width="stretch", config={"displayModeBar": False})

    with right:
        st.subheader(tr("need_signals"))
        needs["aspect_display"] = needs["product_aspect"].map(localized_aspect_label)
        need_columns = ["customer_need", "aspect_display", "review_count"]
        need_labels = [tr("consumer_need"), tr("issue_category"), tr("reviews")]
        if has_values(frame, "product_id"):
            need_columns.append("product_count")
            need_labels.append(tr("products"))
        if has_values(frame, "rating"):
            need_columns.append("average_rating")
            need_labels.append(tr("avg_rating"))
        need_columns.append("mention_rate")
        need_labels.append(tr("mention_rate"))
        display = needs[need_columns].head(8).copy()
        display["mention_rate"] = display["mention_rate"].map(format_percent)
        if "average_rating" in display:
            display["average_rating"] = display["average_rating"].map(lambda value: f"{value:.2f}")
        display.columns = need_labels
        st.dataframe(display, hide_index=True, width="stretch", height=390)

    st.subheader(tr("voice_of_customer"))
    need_options = needs.index.tolist()
    selected_need_index = st.selectbox(
        tr("consumer_need"),
        need_options,
        format_func=lambda index: needs.loc[index, "customer_need"],
    )
    selected_need = needs.loc[selected_need_index]
    st.markdown(f"### {selected_need['customer_need']}")
    has_products = has_values(frame, "product_id")
    basis_key = "need_basis" if has_products else "need_basis_scope"
    st.caption(
        tr(basis_key).format(
            reviews=int(selected_need["review_count"]),
            products=int(selected_need["product_count"]),
        )
    )
    examples = selected_need.get("evidence_examples", [])
    if examples:
        st.markdown(f"**{tr('evidence_examples')}**")
        for example in examples:
            st.markdown(f"- {example}")
    show_all_review_evidence(frame, include_non_five=True)


def render_recommendations(frame: pd.DataFrame) -> None:
    if not has_values(frame, "review_text_clean"):
        st.info(tr("no_text_recommendation"))
        return
    recommendations = recommendation_summary(frame, LANGUAGE)
    st.caption(tr("recommendation_caption"))
    if recommendations.empty:
        st.info(tr("no_recommendation"))
        show_all_review_evidence(frame)
        return

    display_columns = ["priority", "issue", "owner", "recommendation", "review_count", "mention_rate"]
    output_columns = [
        tr("priority"),
        tr("consumer_issue"),
        tr("owner"),
        tr("recommended_action"),
        tr("evidence"),
        tr("mention_rate"),
    ]
    if has_values(frame, "rating"):
        display_columns.append("average_rating")
        output_columns.append(tr("avg_rating"))
    display = recommendations[display_columns].copy()
    display["mention_rate"] = display["mention_rate"].map(format_percent)
    if "average_rating" in display:
        display["average_rating"] = display["average_rating"].map(lambda value: f"{value:.2f}")
    display.columns = output_columns
    st.dataframe(display, hide_index=True, width="stretch", height=360)

    st.subheader(tr("recommendation_brief"))
    selected_issue = st.selectbox(tr("recommendation"), recommendations["issue"].tolist())
    selected = recommendations[recommendations["issue"] == selected_issue].iloc[0]
    left, right = st.columns([1.4, 1], gap="large")
    with left:
        st.markdown(f"### {selected['priority']} · {selected['issue']}")
        st.write(selected["recommendation"])
        st.markdown(f"**{tr('validation')}:** {selected['validation']}")
    with right:
        st.metric(tr("priority_score"), f"{selected['priority_score']} / 100")
        st.metric(tr("evidence_reviews"), f"{selected['review_count']:,}")
        st.markdown(f"**{tr('owner')}:** {selected['owner']}")
    show_review_evidence(frame, selected["product_aspect"], limit=3)
    show_all_review_evidence(frame)


with st.sidebar:
    st.markdown(f'<div class="eyebrow">{tr("data_workspace")}</div>', unsafe_allow_html=True)
    st.markdown(f"## {tr('consumer_insight')}")
    st.markdown(tr("review_dataset"))
    uploaded = st.file_uploader(
        "Review dataset upload",
        type=["csv", "json", "jsonl", "parquet"],
        key="review_dataset_uploader",
        label_visibility="collapsed",
        accept_multiple_files=True,
    )

uploaded_files = list(uploaded or [])
if uploaded_files:
    uploaded_datasets = tuple((item.name, item.getvalue()) for item in uploaded_files)
    uploaded_key = tuple(
        (filename, hashlib.sha256(payload).hexdigest())
        for filename, payload in uploaded_datasets
    )
    dataset_changed = st.session_state.get("active_upload_key") != uploaded_key
    needs_refresh = (
        dataset_changed
        or st.session_state.get("active_analysis_version") != ANALYSIS_VERSION
    )
    if needs_refresh:
        try:
            parsed_data = get_uploaded_datasets(uploaded_datasets, ANALYSIS_VERSION)
        except Exception as exc:
            st.error(f"{tr('load_error')}: {exc}")
            st.stop()
        st.session_state["active_upload_key"] = uploaded_key
        st.session_state["active_upload_data"] = parsed_data
        st.session_state["active_analysis_version"] = ANALYSIS_VERSION
    st.session_state["active_upload_names"] = tuple(item.name for item in uploaded_files)
    data = st.session_state["active_upload_data"]
elif "active_upload_data" in st.session_state:
    data = st.session_state["active_upload_data"]
    if st.session_state.get("active_analysis_version") != ANALYSIS_VERSION:
        data = enrich_reviews(data)
        st.session_state["active_upload_data"] = data
        st.session_state["active_analysis_version"] = ANALYSIS_VERSION
else:
    render_empty_state()
    st.stop()

active_names = st.session_state.get("active_upload_names", ("uploaded file",))
active_dataset_label = format_dataset_label(tuple(active_names))

active_dataset_key = st.session_state.get("active_upload_key")
if st.session_state.get("rendered_dataset_key") != active_dataset_key:
    clear_dataset_dependent_state()
    st.session_state["rendered_dataset_key"] = active_dataset_key

if data.empty:
    render_empty_state()
    st.stop()

filtered = apply_filters(data)
with st.sidebar:
    st.divider()
    scope_value = f"{len(filtered):,} {tr('reviews_lower')}"
    st.metric(tr("scope"), scope_value)
    detected = [
        label
        for column, label in (
            ("review_text_clean", tr("review_text")),
            ("rating", tr("rating")),
            ("product_id", "ASIN"),
            ("review_date", tr("date")),
        )
        if has_values(data, column)
    ]
    st.caption(f"{tr('detected_fields')}: " + (", ".join(detected) if detected else tr("none")))
    st.caption(f"{tr('source')}: {active_dataset_label}")
    st.caption(tr("analysis_caption"))

st.markdown('<div class="eyebrow">Amazon Consumer Insight Agent</div>', unsafe_allow_html=True)
st.title(tr("title"))
st.markdown(
    f'<div class="subtle">{tr("subtitle")}</div>',
    unsafe_allow_html=True,
)
st.caption(f"{tr('uploaded_only')}: {active_dataset_label}")

if filtered.empty:
    st.warning(tr("no_match"))
    st.stop()

overview_tab, pain_tab, voice_tab, recommendation_tab = st.tabs(
    [tr("tab_overview"), tr("tab_pain"), tr("tab_voice"), tr("tab_recommendation")]
)

with overview_tab:
    render_overview(filtered)
with pain_tab:
    render_pain_points(filtered)
with voice_tab:
    render_customer_voice(filtered)
with recommendation_tab:
    render_recommendations(filtered)
