import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_records(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_plot(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def analysis_summary(df: pd.DataFrame) -> Dict:
    issue_counter = Counter()
    for labels in df["issue_labels"]:
        if isinstance(labels, list):
            issue_counter.update(labels)

    top_issues = issue_counter.most_common(10)
    keyword_counter = Counter()
    for tokens in df["tokens"]:
        if isinstance(tokens, list):
            keyword_counter.update(tokens)

    return {
        "total_records": int(len(df)),
        "records_with_return_mentions": int(df["has_return_mention"].sum()),
        "records_with_warranty_mentions": int(df["has_warranty_mention"].sum()),
        "avg_rating": float(df["rating"].dropna().mean()) if not df["rating"].dropna().empty else None,
        "avg_reliability_score": float(df["reliability_score"].mean()),
        "top_issue_categories": top_issues,
        "top_keywords": keyword_counter.most_common(20),
        "company_count": int(df["company"].nunique()),
    }


def make_visuals(df: pd.DataFrame, figures_dir: Path) -> None:
    sns.set_theme(style="whitegrid")

    # 1) Rating distribution
    plt.figure(figsize=(8, 4))
    rating_values = df["rating"].dropna()
    if not rating_values.empty:
        sns.countplot(x=rating_values.astype(int), color="#4c72b0")
        plt.title("Rating Distribution")
        plt.xlabel("Rating")
        plt.ylabel("Count")
    else:
        plt.text(0.5, 0.5, "No rating values available", ha="center")
    save_plot(figures_dir / "rating_distribution.png")

    # 2) Top issue categories
    issue_counter = Counter()
    for labels in df["issue_labels"]:
        if isinstance(labels, list):
            issue_counter.update(labels)
    top_issue_df = pd.DataFrame(issue_counter.most_common(8), columns=["issue", "count"])
    plt.figure(figsize=(9, 5))
    if not top_issue_df.empty:
        sns.barplot(data=top_issue_df, x="count", y="issue", hue="issue", dodge=False, legend=False)
        plt.title("Top Issue Categories")
        plt.xlabel("Count")
        plt.ylabel("Issue Category")
    else:
        plt.text(0.5, 0.5, "No issue labels available", ha="center")
    save_plot(figures_dir / "top_issue_categories.png")

    # 3) Return vs warranty mention rates
    mention_df = pd.DataFrame(
        {
            "type": ["return_mention", "warranty_mention"],
            "count": [int(df["has_return_mention"].sum()), int(df["has_warranty_mention"].sum())],
        }
    )
    plt.figure(figsize=(7, 4))
    sns.barplot(data=mention_df, x="type", y="count", hue="type", dodge=False, legend=False)
    plt.title("Return/Warranty Mention Counts")
    plt.xlabel("Mention Type")
    plt.ylabel("Count")
    save_plot(figures_dir / "return_warranty_mentions.png")

    # 4) Temporal trend
    temporal = df.copy()
    temporal["review_date_parsed"] = pd.to_datetime(temporal["review_date"], errors="coerce")
    temporal = temporal.dropna(subset=["review_date_parsed"])
    plt.figure(figsize=(10, 4))
    if not temporal.empty:
        temporal["month"] = temporal["review_date_parsed"].dt.to_period("M").astype(str)
        by_month = temporal.groupby("month").size().reset_index(name="count").tail(12)
        sns.lineplot(data=by_month, x="month", y="count", marker="o")
        plt.xticks(rotation=45, ha="right")
        plt.title("Issue Volume Over Time (Last 12 Months in Dataset)")
        plt.xlabel("Month")
        plt.ylabel("Complaints Count")
    else:
        plt.text(0.5, 0.5, "No valid date values available", ha="center")
    save_plot(figures_dir / "issue_trend_over_time.png")

    # 5) Top keywords
    keyword_counter = Counter()
    for tokens in df["tokens"]:
        if isinstance(tokens, list):
            keyword_counter.update(tokens)
    top_kw_df = pd.DataFrame(keyword_counter.most_common(15), columns=["keyword", "count"])
    plt.figure(figsize=(10, 5))
    if not top_kw_df.empty:
        sns.barplot(data=top_kw_df, x="count", y="keyword", hue="keyword", dodge=False, legend=False)
        plt.title("Top Complaint Keywords")
        plt.xlabel("Count")
        plt.ylabel("Keyword")
    else:
        plt.text(0.5, 0.5, "No tokenized keywords available", ha="center")
    save_plot(figures_dir / "top_keywords.png")


def write_markdown_report(summary: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 1 EDA Summary",
        "",
        f"- Total records: **{summary['total_records']}**",
        f"- Distinct companies: **{summary['company_count']}**",
        f"- Return mentions: **{summary['records_with_return_mentions']}**",
        f"- Warranty mentions: **{summary['records_with_warranty_mentions']}**",
        f"- Average rating: **{summary['avg_rating']}**",
        f"- Average reliability score: **{summary['avg_reliability_score']}**",
        "",
        "## Top issue categories",
    ]
    for issue, count in summary["top_issue_categories"]:
        lines.append(f"- {issue}: {count}")
    lines.append("")
    lines.append("## Top keywords")
    for kw, count in summary["top_keywords"][:15]:
        lines.append(f"- {kw}: {count}")
    lines.append("")
    lines.append("## Generated figures")
    lines.append("- reports/figures/rating_distribution.png")
    lines.append("- reports/figures/top_issue_categories.png")
    lines.append("- reports/figures/return_warranty_mentions.png")
    lines.append("- reports/figures/issue_trend_over_time.png")
    lines.append("- reports/figures/top_keywords.png")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EDA and visualization on cleaned complaints dataset")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/clean_reviews_ai.json"),
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("reports/figures"),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("reports/analysis_summary.json"),
    )
    parser.add_argument(
        "--summary-md",
        type=Path,
        default=Path("reports/analysis_summary.md"),
    )
    args = parser.parse_args()

    records = load_records(args.input)
    df = pd.DataFrame(records)

    if "rating" not in df.columns:
        df["rating"] = None
    if "reliability_score" not in df.columns:
        df["reliability_score"] = 0.0
    if "issue_labels" not in df.columns:
        df["issue_labels"] = [[] for _ in range(len(df))]
    if "tokens" not in df.columns:
        df["tokens"] = [[] for _ in range(len(df))]
    if "has_return_mention" not in df.columns:
        df["has_return_mention"] = False
    if "has_warranty_mention" not in df.columns:
        df["has_warranty_mention"] = False

    summary = analysis_summary(df)
    make_visuals(df, args.figures_dir)

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown_report(summary, args.summary_md)

    print(f"EDA complete for {len(df)} records")
    print(f"Summary JSON: {args.summary_json}")
    print(f"Summary report: {args.summary_md}")


if __name__ == "__main__":
    main()
