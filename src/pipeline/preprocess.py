import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


EXTRA_STOPWORDS = {
    "would",
    "could",
    "also",
    "one",
    "two",
    "get",
    "got",
    "im",
    "ive",
    "cant",
    "didnt",
    "dont",
    "doesnt",
    "company",
    "service",
    "customer",
}

STOPWORDS = set(ENGLISH_STOP_WORDS).union(EXTRA_STOPWORDS)

RETURN_KEYWORDS = {
    "return",
    "returned",
    "returning",
    "refund",
    "refunded",
    "chargeback",
}
WARRANTY_KEYWORDS = {
    "warranty",
    "guarantee",
    "claim",
    "claimed",
}

ISSUE_KEYWORDS = {
    "delivery_issue": ["late", "delivery", "shipped", "shipping", "arrive", "arrival"],
    "defective_product": ["defect", "defective", "broken", "malfunction", "stopped", "damaged", "faulty"],
    "refund_return_issue": ["refund", "return", "returned", "replacement", "replace"],
    "warranty_claim_issue": ["warranty", "guarantee", "claim", "coverage", "covered"],
    "support_issue": ["support", "agent", "representative", "response", "unhelpful", "rude"],
    "billing_issue": ["charged", "charge", "billing", "payment", "invoice", "fee"],
}


def load_records(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(text: str) -> str:
    text = text or ""
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]


def classify_issues(text: str) -> List[str]:
    labels = []
    lower = text.lower()
    for label, keywords in ISSUE_KEYWORDS.items():
        if any(k in lower for k in keywords):
            labels.append(label)
    if not labels:
        labels.append("other")
    return labels


def build_root_cause_suggestion(issue_labels: List[str]) -> str:
    if "defective_product" in issue_labels:
        return "Improve pre-shipment quality checks and strengthen supplier QA controls."
    if "warranty_claim_issue" in issue_labels:
        return "Clarify warranty terms and simplify claim verification and approval steps."
    if "refund_return_issue" in issue_labels:
        return "Shorten refund turnaround SLA and provide proactive return-status updates."
    if "delivery_issue" in issue_labels:
        return "Audit logistics partners and enforce stricter on-time delivery metrics."
    if "support_issue" in issue_labels:
        return "Train support agents and add escalation paths for unresolved complaints."
    if "billing_issue" in issue_labels:
        return "Add billing validation checks and transparent fee breakdowns at checkout."
    return "Collect more examples to identify a stable root-cause pattern."


def reliability_score(rating, issue_labels: List[str]) -> float:
    base = 100.0
    if rating is not None:
        base -= max(0, (5.0 - float(rating)) * 12)
    base -= min(30, (len(issue_labels) - 1) * 8)
    if "defective_product" in issue_labels:
        base -= 8
    if "warranty_claim_issue" in issue_labels:
        base -= 6
    return round(max(0.0, min(100.0, base)), 2)


def deduplicate_records(records: List[Dict]) -> Tuple[List[Dict], int]:
    seen = set()
    cleaned = []
    duplicates = 0

    for record in records:
        raw_text = record.get("review_text", "")
        normalized = normalize_text(raw_text)
        key = f"{record.get('raw_url', '')}|{hashlib.md5(normalized.encode('utf-8')).hexdigest()}"
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        cleaned.append(record)
    return cleaned, duplicates


def preprocess_records(records: List[Dict]) -> Tuple[List[Dict], Dict]:
    records, duplicate_count = deduplicate_records(records)
    processed = []
    missing_fields_counter = Counter()

    for idx, record in enumerate(records, start=1):
        text = normalize_text(record.get("review_text", ""))
        tokens = tokenize(text)
        issue_labels = classify_issues(text)

        company = record.get("company") or "unknown"
        review_date = record.get("review_date") or "unknown"
        rating = record.get("rating")
        if rating == "":
            rating = None

        if company == "unknown":
            missing_fields_counter["company"] += 1
        if review_date == "unknown":
            missing_fields_counter["review_date"] += 1
        if rating is None:
            missing_fields_counter["rating"] += 1

        lower = text.lower()
        has_return = any(k in lower for k in RETURN_KEYWORDS)
        has_warranty = any(k in lower for k in WARRANTY_KEYWORDS)

        processed.append(
            {
                "record_id": record.get("record_id") or f"record_{idx}",
                "source": record.get("source", "complaintsboard"),
                "company": company,
                "category": record.get("category", "appliances-electronics-and-technology"),
                "title": record.get("title", "unknown"),
                "review_text": record.get("review_text", ""),
                "review_text_normalized": text,
                "tokens": tokens,
                "review_date": review_date,
                "rating": rating,
                "has_return_mention": has_return,
                "has_warranty_mention": has_warranty,
                "issue_labels": issue_labels,
                "root_cause_suggestion": build_root_cause_suggestion(issue_labels),
                "reliability_score": reliability_score(rating, issue_labels),
                "raw_url": record.get("raw_url", ""),
            }
        )

    summary = {
        "total_input_records": len(records) + duplicate_count,
        "total_output_records": len(processed),
        "duplicates_removed": duplicate_count,
        "missing_fields": dict(missing_fields_counter),
    }
    return processed, summary


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess raw complaints records")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/raw_reviews.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/clean_reviews.json"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("data/processed/preprocessing_summary.json"),
    )
    args = parser.parse_args()

    raw_records = load_records(args.input)
    cleaned_records, summary = preprocess_records(raw_records)
    save_json(args.output, cleaned_records)
    save_json(args.summary_output, summary)

    print(f"Loaded {len(raw_records)} raw records")
    print(f"Saved {len(cleaned_records)} cleaned records to {args.output}")
    print(f"Preprocessing summary written to {args.summary_output}")


if __name__ == "__main__":
    main()
