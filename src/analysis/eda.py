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


def prepare_dataframe(records: List[Dict]) -> pd.DataFrame:
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
    if "review_date" not in df.columns:
        df["review_date"] = None
    return df


def run_eda_from_records(
    records: List[Dict],
    figures_dir: Path,
    summary_json: Path,
    summary_md: Path,
) -> Dict:
    df = prepare_dataframe(records)
    summary = analysis_summary(df)
    make_visuals(df, figures_dir)

    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown_report(summary, summary_md)
    return summary


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

    # 1) Top issue categories
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

    # 2) Return vs warranty mention rates
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

    # 3) Top keywords
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
    lines.append("- reports/figures/top_issue_categories.png")
    lines.append("- reports/figures/return_warranty_mentions.png")
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
    summary = run_eda_from_records(records, args.figures_dir, args.summary_json, args.summary_md)

    print(f"EDA complete for {len(df)} records")
    print(f"Summary JSON: {args.summary_json}")
    print(f"Summary report: {args.summary_md}")


if __name__ == "__main__":
    main()
