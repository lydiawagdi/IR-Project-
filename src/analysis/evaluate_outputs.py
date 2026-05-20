"""Evaluate IR search quality and AI output quality."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pipeline.stopwords import STOPWORDS  # noqa: E402


EVAL_QUERIES = [
    ("warranty claim denied", {"warranty", "claim", "denied"}),
    ("refund not received", {"refund", "return", "not"}),
    ("defective product replacement", {"defective", "product", "replacement"}),
    ("repair under warranty", {"warranty", "repair"}),
]


def load_records(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_retrieval(records: List[Dict], top_k: int = 5) -> Dict:
    corpus = [r.get("review_text_normalized", "") or r.get("review_text", "") for r in records]
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=8000,
        stop_words=list(STOPWORDS),
    )
    matrix = vectorizer.fit_transform(corpus)

    query_results = []
    keyword_hit_rates = []

    for query, expected_terms in EVAL_QUERIES:
        query_vec = vectorizer.transform([query])
        sims = cosine_similarity(query_vec, matrix).flatten()
        top_indices = np.argsort(sims)[::-1][:top_k]
        hits = 0
        selected = []
        for idx in top_indices:
            text = corpus[idx].lower()
            matched = [term for term in expected_terms if term in text]
            if matched:
                hits += 1
            selected.append(
                {
                    "company": records[idx].get("company"),
                    "similarity": round(float(sims[idx]), 4),
                    "matched_terms": matched,
                }
            )
        hit_rate = hits / top_k
        keyword_hit_rates.append(hit_rate)
        query_results.append(
            {
                "query": query,
                "top_k": top_k,
                "keyword_hit_rate": round(hit_rate, 4),
                "top_results": selected,
            }
        )

    return {
        "queries_evaluated": len(EVAL_QUERIES),
        "avg_keyword_hit_rate": round(float(np.mean(keyword_hit_rates)), 4),
        "query_results": query_results,
    }


def evaluate_classification(records: List[Dict]) -> Dict:
    y_true = []
    y_pred = []
    for record in records:
        labels = record.get("issue_labels", [])
        if not labels:
            continue
        y_true.append(labels[0])
        y_pred.append(record.get("ai_predicted_issue", "other"))

    if len(y_true) < 5 or len(set(y_true)) < 2:
        return {
            "status": "insufficient_samples",
            "records_evaluated": len(y_true),
        }

    return {
        "status": "ok",
        "records_evaluated": len(y_true),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
    }


def evaluate_ai_coverage(records: List[Dict]) -> Dict:
    total = len(records)
    if total == 0:
        return {"records": 0}

    return {
        "records": total,
        "sentiment_coverage": round(
            sum(1 for r in records if r.get("ai_sentiment")) / total, 4
        ),
        "summary_coverage": round(
            sum(1 for r in records if r.get("ai_summary")) / total, 4
        ),
        "classification_coverage": round(
            sum(1 for r in records if r.get("ai_predicted_issue")) / total, 4
        ),
        "negative_sentiment_rate": round(
            sum(1 for r in records if r.get("ai_sentiment") == "negative") / total, 4
        ),
    }


def write_markdown_report(payload: Dict, path: Path) -> None:
    lines = [
        "# Output Evaluation Report",
        "",
        "## IR Retrieval Evaluation",
        f"- Queries evaluated: **{payload['retrieval']['queries_evaluated']}**",
        f"- Average keyword hit rate @Top5: **{payload['retrieval']['avg_keyword_hit_rate']}**",
        "",
        "## AI Classification Evaluation",
    ]
    cls = payload["classification"]
    if cls.get("status") == "ok":
        lines.extend(
            [
                f"- Records evaluated: **{cls['records_evaluated']}**",
                f"- Accuracy vs rule-based primary label: **{cls['accuracy']}**",
                f"- Macro F1: **{cls['macro_f1']}**",
            ]
        )
    else:
        lines.append(f"- Status: **{cls.get('status', 'unknown')}**")

    lines.extend(
        [
            "",
            "## AI Output Coverage",
            f"- Sentiment coverage: **{payload['ai_coverage']['sentiment_coverage']}**",
            f"- Summarization coverage: **{payload['ai_coverage']['summary_coverage']}**",
            f"- Classification coverage: **{payload['ai_coverage']['classification_coverage']}**",
            f"- Negative sentiment rate: **{payload['ai_coverage']['negative_sentiment_rate']}**",
            "",
            "## Interpretation",
            "- Higher keyword hit rate means TF-IDF retrieval returns warranty/return-relevant complaints.",
            "- Classification metrics compare AI labels against rule-based labels from preprocessing.",
            "- Sentiment and summary coverage should be close to 1.0 after AI enrichment.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_evaluation(records: List[Dict]) -> Dict:
    payload = {
        "retrieval": evaluate_retrieval(records),
        "classification": evaluate_classification(records),
        "ai_coverage": evaluate_ai_coverage(records),
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate IR and AI outputs")
    parser.add_argument("--input", type=Path, default=Path("data/processed/clean_reviews_ai.json"))
    parser.add_argument("--json-output", type=Path, default=Path("reports/evaluation_report.json"))
    parser.add_argument("--md-output", type=Path, default=Path("reports/evaluation_report.md"))
    args = parser.parse_args()

    records = load_records(args.input)
    payload = run_evaluation(records)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown_report(payload, args.md_output)

    print(f"Evaluation complete for {len(records)} records")
    print(f"Saved JSON report to {args.json_output}")
    print(f"Saved markdown report to {args.md_output}")


if __name__ == "__main__":
    main()
