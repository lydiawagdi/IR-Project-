import json
import importlib
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_SCRIPT_PATH = Path(__file__).resolve()
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pipeline.stopwords import STOPWORDS  # noqa: E402
from analysis.eda import run_eda_from_records  # noqa: E402


def bootstrap_streamlit_runtime() -> None:
    """When run with `python app.py`, re-launch using `python -m streamlit run`."""
    try:
        scriptrunner_module = importlib.import_module("streamlit.runtime.scriptrunner")
        get_script_run_ctx = getattr(scriptrunner_module, "get_script_run_ctx", None)
    except Exception:
        get_script_run_ctx = None

    if get_script_run_ctx and get_script_run_ctx() is not None:
        return

    launch_command = [sys.executable, "-m", "streamlit", "run", str(APP_SCRIPT_PATH)]
    launch_command.extend(sys.argv[1:])
    raise SystemExit(subprocess.call(launch_command, cwd=str(PROJECT_ROOT)))


if __name__ == "__main__":
    bootstrap_streamlit_runtime()


import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_DATASET = Path("data/processed/clean_reviews_ai.json")
FIGURES_DIR = Path("reports/figures")
SUMMARY_JSON = Path("reports/analysis_summary.json")
SUMMARY_MD = Path("reports/analysis_summary.md")
SCRAPER_WALKTHROUGH_STEPS = [
    ("Checking robots.txt permissions", "[ROBOTS]"),
    ("Opening categories page with Selenium", "[CATEGORIES]"),
    ("Crawling category subpages for complaint links", "[SUBPAGES]"),
    ("Searching warranty and return listings", "[SEARCH]"),
    ("Extracting complaint details and filtering relevance", "[EXTRACT]"),
]


def load_records(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def search_records(records: List[Dict], query: str, top_k: int) -> List[Dict]:
    corpus = [r.get("review_text_normalized", "") for r in records]
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=8000,
        stop_words=list(STOPWORDS),
    )
    matrix = vectorizer.fit_transform(corpus)
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, matrix).flatten()
    top_indices = np.argsort(sims)[::-1][:top_k]

    results = []
    for idx in top_indices:
        rec = records[idx].copy()
        rec["similarity"] = round(float(sims[idx]), 4)
        results.append(rec)
    return results


def extract_year(date_value: Optional[str]) -> Optional[int]:
    if not date_value or str(date_value).strip().lower() == "unknown":
        return None
    match = re.search(r"\b(19|20)\d{2}\b", str(date_value))
    return int(match.group(0)) if match else None


def get_available_years(records: List[Dict]) -> List[int]:
    years = set()
    for record in records:
        for field in ("review_date", "review_updated_date"):
            year = extract_year(record.get(field))
            if year:
                years.add(year)
    return sorted(years, reverse=True)


def apply_filters(
    records: List[Dict],
    return_only: bool,
    warranty_only: bool,
    issue_filter: str,
    selected_years: Optional[List[int]],
    include_unknown_year: bool,
) -> List[Dict]:
    filtered = records
    if return_only:
        filtered = [r for r in filtered if r.get("has_return_mention")]
    if warranty_only:
        filtered = [r for r in filtered if r.get("has_warranty_mention")]
    if issue_filter != "all":
        filtered = [r for r in filtered if issue_filter in r.get("issue_labels", [])]
    if selected_years is not None:
        allowed_years = set(selected_years)
        filtered = [
            r
            for r in filtered
            if (extract_year(r.get("review_date")) in allowed_years)
            or (extract_year(r.get("review_date")) is None and include_unknown_year)
        ]
    return filtered


def render_results(results: List[Dict]) -> None:
    if not results:
        st.warning("No matching records found for current query/filters.")
        return

    for i, rec in enumerate(results, start=1):
        with st.container(border=True):
            st.markdown(f"### Result #{i}  |  Similarity: `{rec.get('similarity', 0.0)}`")
            st.write(f"**Company:** {rec.get('company', 'unknown')}")
            st.write(f"**Title:** {rec.get('title', 'unknown')}")
            if rec.get("breadcrumbs"):
                st.write(f"**Breadcrumbs:** {' > '.join(rec.get('breadcrumbs', []))}")
            if rec.get("reviewer_name") or rec.get("reviewer_location"):
                st.write(
                    f"**Reviewer:** {rec.get('reviewer_name', 'N/A')} "
                    f"({rec.get('reviewer_location', 'N/A')})"
                )
            st.write(
                f"**Date:** {rec.get('review_date', 'unknown')}  |  "
                f"**Updated:** {rec.get('review_updated_date', 'N/A')}  |  "
                f"**Rating:** {rec.get('rating', 'N/A')}"
            )
            st.write(
                f"**Helpful count:** {rec.get('helpful_count', 'N/A')}  |  "
                f"**Comment count:** {rec.get('comment_count', 0)}  |  "
                f"**Featured:** {rec.get('is_featured_review', False)}"
            )
            if rec.get("claimed_loss"):
                st.write(f"**Claimed loss:** {rec.get('claimed_loss')}")
            if rec.get("desired_outcome"):
                st.write(f"**Desired outcome:** {rec.get('desired_outcome')}")
            st.write(f"**Issue labels:** {', '.join(rec.get('issue_labels', [])) or 'N/A'}")
            st.write(f"**AI predicted issue:** {rec.get('ai_predicted_issue', 'N/A')}")
            st.write(f"**AI confidence:** {rec.get('ai_confidence', 'N/A')}")
            st.write(
                f"**AI sentiment:** {rec.get('ai_sentiment', 'N/A')} "
                f"({rec.get('ai_sentiment_score', 'N/A')})"
            )
            if rec.get("ai_summary"):
                st.write(f"**AI summary:** {rec.get('ai_summary')}")
            if rec.get("ai_domain_relevance"):
                st.write(f"**AI domain relevance:** {rec.get('ai_domain_relevance')}")
            st.write(f"**Root cause suggestion:** {rec.get('ai_root_cause_suggestion', rec.get('root_cause_suggestion', 'N/A'))}")
            st.write(f"**Source URL:** {rec.get('raw_url', '')}")
            st.caption((rec.get("review_text", "") or "")[:800] + "...")
            comments = rec.get("comments", [])
            if comments:
                st.markdown("**Comments**")
                for comment in comments[:5]:
                    st.write(
                        f"- **{comment.get('author_name', 'Anonymous')}** "
                        f"({comment.get('posted_at', 'unknown')}): "
                        f"{(comment.get('text', '') or '')[:300]}"
                    )


def render_figures(records: List[Dict], dataset_path: Path) -> None:
    st.subheader("Analysis Visualizations")
    st.caption("Charts are generated from your dataset. Use the button below after scraping or changing filters.")

    chart_source = st.radio(
        "Chart data source",
        options=["Current sidebar filters", "Full dataset file"],
        horizontal=True,
        key="chart_data_source",
    )

    if st.button("Update charts", type="primary", width="stretch", key="update_charts_button"):
        if chart_source == "Full dataset file":
            try:
                chart_records = load_records(dataset_path)
            except Exception as exc:
                st.error(f"Could not load dataset at {dataset_path}: {exc}")
                chart_records = []
        else:
            chart_records = records

        if not chart_records:
            st.warning("No records available to build charts.")
        else:
            with st.spinner(f"Generating charts from {len(chart_records)} records..."):
                try:
                    summary = run_eda_from_records(
                        chart_records,
                        FIGURES_DIR,
                        SUMMARY_JSON,
                        SUMMARY_MD,
                    )
                    st.session_state["charts_last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state["charts_record_count"] = len(chart_records)
                    st.session_state["charts_source"] = chart_source
                    st.success(
                        f"Charts updated using {len(chart_records)} records "
                        f"({summary.get('total_records', len(chart_records))} in summary)."
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f"Chart generation failed: {exc}")

    if st.session_state.get("charts_last_updated"):
        st.info(
            f"Last chart update: {st.session_state['charts_last_updated']} "
            f"| Records: {st.session_state.get('charts_record_count', 'N/A')} "
            f"| Source: {st.session_state.get('charts_source', 'N/A')}"
        )

    figure_files = [
        ("top_issue_categories.png", "Top issue categories"),
        ("return_warranty_mentions.png", "Return vs warranty mentions"),
        ("top_keywords.png", "Top keywords"),
    ]
    for fig, caption in figure_files:
        path = FIGURES_DIR / fig
        if path.exists():
            st.image(str(path), caption=caption, width="stretch")
        else:
            st.info(f"Figure not found: {path}. Click **Update charts** to generate it.")


def render_markdown_file(path: Path, title: str) -> None:
    st.markdown(f"### {title}")
    if path.exists():
        st.markdown(path.read_text(encoding="utf-8"))
    else:
        st.info(f"Report not found: {path}. Run the pipeline to generate it.")


def render_rubric_coverage() -> None:
    st.subheader("Rubric Coverage")
    rubric_rows = [
        ("Feature Extraction", "TF-IDF terms, token stats, keyword signals", "src/analysis/feature_extraction.py"),
        ("Product System Implementation", "Streamlit GUI + CLI search", "src/product/app.py, src/product/cli.py"),
        (
            "AI Feature Integration",
            "Classification, sentiment, summarization",
            "src/analysis/ai_enrichment.py",
        ),
        ("AI Feature Relevance", "Warranty/return domain mapping", "ai_domain_relevance field in dataset"),
        ("Insight Generation", "Automated insights report", "reports/project_insights.md"),
        ("Evaluation of Outputs", "Retrieval + AI quality metrics", "reports/evaluation_report.md"),
        (
            "Code Structure & Reproducibility",
            "Modular pipeline scripts + README",
            "src/*, README.md, reports/RUBRIC_COVERAGE.md",
        ),
    ]
    st.table(
        {
            "Requirement": [row[0] for row in rubric_rows],
            "Implementation": [row[1] for row in rubric_rows],
            "Evidence": [row[2] for row in rubric_rows],
        }
    )


def render_insights_tab() -> None:
    st.subheader("Insights & Evaluation")
    st.caption("Project-level interpretation and measurable output quality checks.")

    render_rubric_coverage()
    col1, col2 = st.columns(2)
    with col1:
        render_markdown_file(PROJECT_ROOT / "reports/project_insights.md", "Project Insights")
    with col2:
        render_markdown_file(PROJECT_ROOT / "reports/evaluation_report.md", "Output Evaluation")

    metrics_path = PROJECT_ROOT / "reports/ai_metrics.json"
    feature_path = PROJECT_ROOT / "reports/feature_extraction_summary.json"
    if metrics_path.exists():
        st.markdown("### AI Metrics")
        st.json(json.loads(metrics_path.read_text(encoding="utf-8")))
    if feature_path.exists():
        st.markdown("### Feature Extraction Summary")
        st.json(json.loads(feature_path.read_text(encoding="utf-8")))


def run_command(command: List[str]) -> Dict[str, str]:
    completed = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    return {
        "exit_code": str(completed.returncode),
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def render_scraper_walkthrough() -> None:
    st.markdown("#### Scraper Walkthrough")
    st.caption("Quick preview of what happens behind the scenes before live scraping starts.")

    status_placeholder = st.empty()
    progress_bar = st.progress(0, text="Preparing scraper...")
    animation_frames = ["", ".", "..", "..."]
    total_frames = len(SCRAPER_WALKTHROUGH_STEPS) * len(animation_frames)
    frame_counter = 0

    for step, emoji in SCRAPER_WALKTHROUGH_STEPS:
        for frame in animation_frames:
            frame_counter += 1
            progress_pct = int((frame_counter / total_frames) * 100)
            status_placeholder.info(f"{emoji} {step}{frame}")
            progress_bar.progress(progress_pct, text=f"{step}{frame}")
            time.sleep(0.12)

    status_placeholder.success("Scraper launched: collecting live complaint pages now.")
    progress_bar.progress(100, text="Scraper is running...")


def default_min_year() -> int:
    return datetime.now().year - 4


def run_pipeline_steps(
    run_scraper: bool,
    target_records: int,
    max_categories: int,
    max_search_pages: int,
    crawl_delay: float,
    light_scrape: bool,
    skip_charts: bool,
    min_scrape_year: int,
) -> None:
    steps: List[tuple[str, List[str]]] = []

    if run_scraper:
        scraper_cmd = [
            sys.executable,
            "src/scraper/scrape_complaints.py",
            "--target-records",
            str(int(target_records)),
            "--max-categories",
            str(int(max_categories)),
            "--max-search-pages",
            str(int(max_search_pages)),
            "--crawl-delay",
            str(float(crawl_delay)),
            "--min-year",
            str(int(min_scrape_year)),
        ]
        if light_scrape:
            scraper_cmd.append("--light-mode")
        else:
            scraper_cmd.append("--no-light-mode")
        steps.append(("Scraper", scraper_cmd))

    steps.extend(
        [
            ("Preprocess", [sys.executable, "src/pipeline/preprocess.py"]),
            ("Feature Extraction", [sys.executable, "src/analysis/feature_extraction.py"]),
            ("AI Enrichment", [sys.executable, "src/analysis/ai_enrichment.py"]),
            ("Output Evaluation", [sys.executable, "src/analysis/evaluate_outputs.py"]),
            ("Insights Report", [sys.executable, "src/analysis/generate_insights.py"]),
        ]
    )
    if not skip_charts:
        steps.append(("EDA & Charts", [sys.executable, "src/analysis/eda.py"]))

    all_ok = True
    for step_name, command in steps:
        if step_name == "Scraper":
            render_scraper_walkthrough()
        with st.spinner(f"Running {step_name}..."):
            result = run_command(command)
        success = result["exit_code"] == "0"
        if success:
            st.success(f"{step_name} finished successfully.")
        else:
            st.error(f"{step_name} failed with exit code {result['exit_code']}.")
            all_ok = False

        if result["stdout"]:
            st.caption(f"{step_name} stdout")
            st.code(result["stdout"], language="text")
        if result["stderr"]:
            st.caption(f"{step_name} stderr")
            st.code(result["stderr"], language="text")

        if not success:
            break

    if all_ok:
        st.success("Pipeline completed. Click the button below to reload data in this UI.")
        if st.button("Reload App Data", width="stretch"):
            st.rerun()


def render_pipeline_runner(
    run_scraper: bool,
    target_records: int,
    max_categories: int,
    max_search_pages: int,
    crawl_delay: float,
    light_scrape: bool,
    skip_charts: bool,
    min_scrape_year: int,
) -> None:
    st.subheader("Pipeline Runner")
    st.caption("Run data collection and processing directly from this GUI.")
    st.info(
        "Pipeline order: Scraper (optional) -> Preprocess -> Feature extraction -> "
        "AI enrichment -> Output evaluation -> Insights"
        + ("" if skip_charts else " -> EDA/charts")
        + ". Use light scrape to reduce CPU/RAM usage."
    )

    if st.button("Run Pipeline Now", type="primary", width="stretch"):
        run_pipeline_steps(
            run_scraper,
            target_records,
            max_categories,
            max_search_pages,
            crawl_delay,
            light_scrape,
            skip_charts,
            min_scrape_year,
        )


def main() -> None:
    st.set_page_config(page_title="Warranty & Return Analyzer", layout="wide")
    st.title("Warranty & Return Analyzer - GUI")
    st.caption("Search complaints and explore return/warranty issue insights")

    dataset_path = st.sidebar.text_input("Dataset path", str(DEFAULT_DATASET))
    top_k = st.sidebar.slider("Top K results", min_value=3, max_value=30, value=10, step=1)
    query = st.sidebar.text_input("Search query", "warranty claim denied refund")
    return_only = st.sidebar.checkbox("Only records with return mention", value=False)
    warranty_only = st.sidebar.checkbox("Only records with warranty mention", value=False)

    try:
        records = load_records(Path(dataset_path))
    except Exception as exc:
        st.error(f"Failed to load dataset at {dataset_path}: {exc}")
        return

    available_years = get_available_years(records)
    recent_default_years = [y for y in available_years if y >= default_min_year()] or available_years
    st.sidebar.markdown("---")
    st.sidebar.subheader("Year filter")
    if available_years:
        selected_years = st.sidebar.multiselect(
            "Years to include in search",
            options=available_years,
            default=recent_default_years,
            help="Toggle which complaint years are included in search, charts, and table views.",
            key="year_multiselect",
        )
    else:
        selected_years = []
        st.sidebar.info("No parseable years found in dataset dates.")
    include_unknown_year = st.sidebar.checkbox(
        "Include records with unknown year",
        value=False,
        key="include_unknown_year",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Quick Pipeline Run")
    run_scraper = st.sidebar.checkbox("Run scraper step", value=True, key="pipeline_run_scraper")
    light_scrape = st.sidebar.checkbox(
        "Light scrape (recommended)",
        value=True,
        help="Search-only, smaller browser footprint, fewer pages, periodic Chrome restarts.",
        key="pipeline_light_scrape",
    )
    skip_charts = st.sidebar.checkbox(
        "Skip chart generation",
        value=True,
        help="Skips matplotlib EDA step to save memory after scraping.",
        key="pipeline_skip_charts",
    )
    target_records = st.sidebar.number_input(
        "Target records",
        min_value=10,
        max_value=100,
        value=25,
        step=5,
        key="pipeline_target_records",
    )
    max_categories = st.sidebar.number_input(
        "Max categories",
        min_value=0,
        max_value=10,
        value=2,
        step=1,
        disabled=light_scrape,
        key="pipeline_max_categories",
    )
    max_search_pages = st.sidebar.number_input(
        "Max search pages",
        min_value=1,
        max_value=10,
        value=2,
        step=1,
        key="pipeline_max_search_pages",
    )
    crawl_delay = st.sidebar.number_input(
        "Crawl delay (seconds)",
        min_value=0.5,
        max_value=3.0,
        value=1.0,
        step=0.1,
        key="pipeline_crawl_delay",
    )
    current_year = datetime.now().year
    min_scrape_year = st.sidebar.number_input(
        "Minimum complaint year (scraper)",
        min_value=2000,
        max_value=current_year,
        value=default_min_year(),
        step=1,
        help="Scraper keeps only complaints from this year onward and sorts search by date.",
        key="pipeline_min_scrape_year",
    )
    sidebar_run_pipeline = st.sidebar.button(
        "Run Pipeline Now",
        type="primary",
        width="stretch",
        key="pipeline_sidebar_run",
    )

    issue_values = sorted({label for r in records for label in r.get("issue_labels", [])})
    issue_filter = st.sidebar.selectbox("Issue label filter", ["all"] + issue_values)

    year_filter_values = selected_years if available_years else None
    filtered = apply_filters(
        records,
        return_only,
        warranty_only,
        issue_filter,
        year_filter_values,
        include_unknown_year,
    )
    st.sidebar.write(f"Records after filters: **{len(filtered)}**")
    if year_filter_values:
        st.sidebar.caption(f"Years included: {min(year_filter_values)}–{max(year_filter_values)}")
    elif available_years and not year_filter_values:
        st.sidebar.warning("No years selected. Select at least one year to view results.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total records loaded", len(records))
    col2.metric("Filtered records", len(filtered))
    col3.metric("Distinct companies", len({r.get("company") for r in filtered}))

    if sidebar_run_pipeline:
        st.subheader("Pipeline Runner Output")
        run_pipeline_steps(
            run_scraper,
            int(target_records),
            int(max_categories),
            int(max_search_pages),
            float(crawl_delay),
            light_scrape,
            skip_charts,
            int(min_scrape_year),
        )

    tab_search, tab_data, tab_figures, tab_insights, tab_pipeline = st.tabs(
        ["Search", "Dataset Preview", "Charts", "Insights & Evaluation", "Pipeline Runner"]
    )

    with tab_search:
        if not query.strip():
            st.warning("Please enter a search query in the sidebar.")
        elif not filtered:
            st.warning("No records left after filters.")
        else:
            top_k_effective = min(top_k, len(filtered))
            results = search_records(filtered, query.strip(), top_k_effective)
            render_results(results)

    with tab_data:
        df = pd.DataFrame(filtered)
        st.dataframe(df, width="stretch", height=520)

    with tab_figures:
        render_figures(filtered, Path(dataset_path))

    with tab_insights:
        render_insights_tab()

    with tab_pipeline:
        render_pipeline_runner(
            run_scraper,
            int(target_records),
            int(max_categories),
            int(max_search_pages),
            float(crawl_delay),
            light_scrape,
            skip_charts,
            int(min_scrape_year),
        )


if __name__ == "__main__":
    main()
