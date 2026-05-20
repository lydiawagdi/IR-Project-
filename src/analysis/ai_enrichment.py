"""AI enrichment: sentiment analysis, issue classification, and extractive summarization."""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from analysis.ai_issue_classifier import load_records, save_json, train_and_predict  # noqa: E402

NEGATIVE_CUES = {
    "denied",
    "refused",
    "failed",
    "broken",
    "defective",
    "scam",
    "fraud",
    "terrible",
    "awful",
    "worst",
    "unacceptable",
    "delay",
    "delayed",
    "ignored",
    "useless",
    "disappointed",
    "angry",
    "frustrated",
    "not honor",
    "would not",
    "won't",
    "cant",
    "cannot",
}
POSITIVE_CUES = {
    "resolved",
    "fixed",
    "replaced",
    "refunded",
    "helpful",
    "satisfied",
    "approved",
    "honored",
    "completed",
    "successful",
    "thank",
    "thanks",
    "working",
}


def split_sentences(text: str) -> List[str]:
    chunks = re.split(r"(?<=[.!?])\s+", text or "")
    return [c.strip() for c in chunks if c and len(c.strip()) > 20]


def sentiment_score(text: str) -> Tuple[str, float]:
    lower = (text or "").lower()
    neg = sum(1 for cue in NEGATIVE_CUES if cue in lower)
    pos = sum(1 for cue in POSITIVE_CUES if cue in lower)
    score = (pos - neg) / max(pos + neg, 1)
    if score > 0.15:
        label = "positive"
    elif score < -0.15:
        label = "negative"
    else:
        label = "neutral"
    return label, round(score, 4)


def summarize_text(text: str, max_sentences: int = 2) -> str:
    sentences = split_sentences(text)
    if not sentences:
        cleaned = " ".join((text or "").split())
        return cleaned[:280] + ("..." if len(cleaned) > 280 else "")

    focus_terms = {"warranty", "return", "refund", "claim", "replacement", "repair", "defective"}

    def score_sentence(sentence: str) -> int:
        lower = sentence.lower()
        return sum(2 for term in focus_terms if term in lower) + min(len(sentence.split()), 40) // 10

    ranked = sorted(sentences, key=score_sentence, reverse=True)
    selected = ranked[:max_sentences]
    summary = " ".join(selected).strip()
    return summary[:400] + ("..." if len(summary) > 400 else "")


def domain_relevance_note(record: Dict) -> str:
    issue = record.get("ai_predicted_issue", "other")
    mapping = {
        "warranty_claim_issue": "Directly supports warranty claim workflow analysis.",
        "refund_return_issue": "Directly supports return/refund policy optimization.",
        "defective_product": "Supports product quality and replacement decision-making.",
        "delivery_issue": "Supports post-purchase logistics and customer recovery.",
        "support_issue": "Supports customer support escalation and response quality.",
        "billing_issue": "Supports billing dispute and refund validation.",
        "other": "General complaint signal; may need additional domain labeling.",
    }
    return mapping.get(issue, mapping["other"])


def enrich_with_sentiment_and_summary(records: List[Dict]) -> None:
    for record in records:
        text = record.get("review_text", "") or record.get("full_text_with_comments", "")
        label, score = sentiment_score(text)
        record["ai_sentiment"] = label
        record["ai_sentiment_score"] = score
        record["ai_summary"] = summarize_text(text)
        record["ai_domain_relevance"] = domain_relevance_note(record)


def run_ai_enrichment(records: List[Dict]) -> Dict:
    metrics = train_and_predict(records)
    enrich_with_sentiment_and_summary(records)
    metrics["ai_features"] = ["issue_classification", "sentiment_analysis", "extractive_summarization"]
    metrics["sentiment_distribution"] = {
        label: sum(1 for r in records if r.get("ai_sentiment") == label)
        for label in ["negative", "neutral", "positive"]
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AI enrichment pipeline for complaint records")
    parser.add_argument("--input", type=Path, default=Path("data/processed/clean_reviews.json"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/clean_reviews_ai.json"))
    parser.add_argument("--metrics-output", type=Path, default=Path("reports/ai_metrics.json"))
    args = parser.parse_args()

    records = load_records(args.input)
    metrics = run_ai_enrichment(records)
    save_json(args.output, records)
    save_json(args.metrics_output, metrics)

    print(f"AI enrichment complete for {len(records)} records")
    print(f"Saved enriched dataset to {args.output}")
    print(f"Saved AI metrics to {args.metrics_output}")


if __name__ == "__main__":
    main()
