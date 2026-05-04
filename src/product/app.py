import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_DATASET = Path("data/processed/clean_reviews_ai.json")
FIGURES_DIR = Path("reports/figures")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_records(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def search_records(records: List[Dict], query: str, top_k: int) -> List[Dict]:
    corpus = [r.get("review_text_normalized", "") for r in records]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=8000)
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


def apply_filters(records: List[Dict], return_only: bool, warranty_only: bool, issue_filter: str) -> List[Dict]:
    filtered = records
    if return_only:
        filtered = [r for r in filtered if r.get("has_return_mention")]
    if warranty_only:
        filtered = [r for r in filtered if r.get("has_warranty_mention")]
    if issue_filter != "all":
        filtered = [r for r in filtered if issue_filter in r.get("issue_labels", [])]
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
            st.write(f"**Issue labels:** {', '.join(rec.get('issue_labels', [])) or 'N/A'}")
            st.write(f"**AI predicted issue:** {rec.get('ai_predicted_issue', 'N/A')}")
            st.write(f"**AI confidence:** {rec.get('ai_confidence', 'N/A')}")
            st.write(f"**Root cause suggestion:** {rec.get('ai_root_cause_suggestion', rec.get('root_cause_suggestion', 'N/A'))}")
            st.write(f"**Date:** {rec.get('review_date', 'unknown')}  |  **Rating:** {rec.get('rating', 'N/A')}")
            st.write(f"**Source URL:** {rec.get('raw_url', '')}")
            st.caption((rec.get("review_text", "") or "")[:800] + "...")


def render_figures() -> None:
    st.subheader("Analysis Visualizations")
    figure_files = [
        "rating_distribution.png",
        "top_issue_categories.png",
        "return_warranty_mentions.png",
        "issue_trend_over_time.png",
        "top_keywords.png",
    ]
    for fig in figure_files:
        path = FIGURES_DIR / fig
        if path.exists():
            st.image(str(path), caption=fig, use_container_width=True)
        else:
            st.info(f"Figure not found: {path}")


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


def render_pipeline_runner() -> None:
    st.subheader("Pipeline Runner")
    st.caption("Run data collection and processing directly from this GUI.")

    run_scraper = st.checkbox("Run scraper step", value=True)
    target_records = st.number_input("Target records", min_value=50, max_value=300, value=80, step=10)
    max_sitemaps = st.number_input("Max sitemaps", min_value=1, max_value=10, value=2, step=1)
    crawl_delay = st.number_input("Crawl delay (seconds)", min_value=0.1, max_value=3.0, value=0.5, step=0.1)

    st.info(
        "Pipeline order: Scraper (optional) -> Preprocess -> AI enrichment -> EDA/charts. "
        "This may take a few minutes."
    )

    if st.button("Run Pipeline Now", type="primary", use_container_width=True):
        steps: List[tuple[str, List[str]]] = []

        if run_scraper:
            steps.append(
                (
                    "Scraper",
                    [
                        sys.executable,
                        "src/scraper/scrape_complaints.py",
                        "--target-records",
                        str(int(target_records)),
                        "--max-sitemaps",
                        str(int(max_sitemaps)),
                        "--crawl-delay",
                        str(float(crawl_delay)),
                    ],
                )
            )
        steps.extend(
            [
                ("Preprocess", [sys.executable, "src/pipeline/preprocess.py"]),
                ("AI Enrichment", [sys.executable, "src/analysis/ai_issue_classifier.py"]),
                ("EDA & Charts", [sys.executable, "src/analysis/eda.py"]),
            ]
        )

        all_ok = True
        for step_name, command in steps:
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
            if st.button("Reload App Data", use_container_width=True):
                st.rerun()


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

    issue_values = sorted({label for r in records for label in r.get("issue_labels", [])})
    issue_filter = st.sidebar.selectbox("Issue label filter", ["all"] + issue_values)

    filtered = apply_filters(records, return_only, warranty_only, issue_filter)
    st.sidebar.write(f"Records after filters: **{len(filtered)}**")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total records loaded", len(records))
    col2.metric("Filtered records", len(filtered))
    col3.metric("Distinct companies", len({r.get("company") for r in filtered}))

    tab_search, tab_data, tab_figures, tab_pipeline = st.tabs(
        ["Search", "Dataset Preview", "Charts", "Pipeline Runner"]
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
        st.dataframe(df, use_container_width=True, height=520)

    with tab_figures:
        render_figures()

    with tab_pipeline:
        render_pipeline_runner()


if __name__ == "__main__":
    main()
