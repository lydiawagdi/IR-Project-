"""Generate human-readable project insights from processed and AI-enriched data."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List


def load_records(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def top_counter(values: List[str], n: int = 8) -> List[tuple[str, int]]:
    return Counter(values).most_common(n)


def build_insights(records: List[Dict]) -> Dict:
    issue_counter = Counter()
    companies = []
    sentiments = []
    for record in records:
        companies.append(record.get("company", "unknown"))
        sentiments.append(record.get("ai_sentiment", "unknown"))
        for label in record.get("issue_labels", []):
            issue_counter[label] += 1

    top_issues = issue_counter.most_common(8)
    top_companies = top_counter(companies, 8)
    sentiment_dist = Counter(sentiments)

    return_mentions = sum(1 for r in records if r.get("has_return_mention"))
    warranty_mentions = sum(1 for r in records if r.get("has_warranty_mention"))
    negative_sentiment = sentiment_dist.get("negative", 0)

    insights = {
        "total_records": len(records),
        "top_issue_categories": top_issues,
        "top_companies": top_companies,
        "sentiment_distribution": dict(sentiment_dist),
        "return_mention_rate": round(return_mentions / max(len(records), 1), 4),
        "warranty_mention_rate": round(warranty_mentions / max(len(records), 1), 4),
        "negative_sentiment_rate": round(negative_sentiment / max(len(records), 1), 4),
        "key_findings": [],
    }

    if top_issues:
        top_issue, count = top_issues[0]
        insights["key_findings"].append(
            f"The most common issue category is '{top_issue}' ({count} labeled mentions)."
        )
    insights["key_findings"].append(
        f"{return_mentions} complaints mention return/refund language ({insights['return_mention_rate']:.0%} of records)."
    )
    insights["key_findings"].append(
        f"{warranty_mentions} complaints mention warranty/claim language ({insights['warranty_mention_rate']:.0%} of records)."
    )
    insights["key_findings"].append(
        f"{negative_sentiment} records are classified with negative sentiment ({insights['negative_sentiment_rate']:.0%})."
    )
    if top_companies:
        insights["key_findings"].append(
            f"Most frequent company in dataset: {top_companies[0][0]} ({top_companies[0][1]} records)."
        )

    return insights


def write_markdown(insights: Dict, path: Path) -> None:
    lines = [
        "# Project Insights Report",
        "",
        "## Dataset Overview",
        f"- Total analyzed complaints: **{insights['total_records']}**",
        f"- Return/refund mention rate: **{insights['return_mention_rate']}**",
        f"- Warranty/claim mention rate: **{insights['warranty_mention_rate']}**",
        f"- Negative sentiment rate: **{insights['negative_sentiment_rate']}**",
        "",
        "## Top Issue Categories",
    ]
    for issue, count in insights["top_issue_categories"]:
        lines.append(f"- {issue}: {count}")
    lines.extend(["", "## Top Companies"])
    for company, count in insights["top_companies"]:
        lines.append(f"- {company}: {count}")
    lines.extend(["", "## Sentiment Distribution"])
    for label, count in insights["sentiment_distribution"].items():
        lines.append(f"- {label}: {count}")
    lines.extend(["", "## Key Findings"])
    for finding in insights["key_findings"]:
        lines.append(f"- {finding}")
    lines.extend(
        [
            "",
            "## Product Interpretation",
            "- Warranty and return complaints are strongly represented, validating domain focus.",
            "- AI outputs (classification, sentiment, summary) support triage and policy review workflows.",
            "- IR search can prioritize warranty/return language for analyst investigation.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate project insights report")
    parser.add_argument("--input", type=Path, default=Path("data/processed/clean_reviews_ai.json"))
    parser.add_argument("--json-output", type=Path, default=Path("reports/project_insights.json"))
    parser.add_argument("--md-output", type=Path, default=Path("reports/project_insights.md"))
    args = parser.parse_args()

    records = load_records(args.input)
    insights = build_insights(records)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(insights, indent=2), encoding="utf-8")
    write_markdown(insights, args.md_output)

    print(f"Insights generated for {len(records)} records")
    print(f"Saved JSON insights to {args.json_output}")
    print(f"Saved markdown insights to {args.md_output}")


if __name__ == "__main__":
    main()
