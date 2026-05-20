"""IR feature extraction: token stats, keyword signals, and corpus TF-IDF terms."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pipeline.stopwords import STOPWORDS  # noqa: E402

RETURN_KEYWORDS = {"return", "refund", "returned", "replacement", "chargeback"}
WARRANTY_KEYWORDS = {"warranty", "guarantee", "claim", "coverage", "repair"}


def load_records(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def keyword_hits(text: str, keywords: set) -> int:
    lower = (text or "").lower()
    return sum(1 for kw in keywords if kw in lower)


def enrich_record_features(record: Dict) -> Dict:
    text = record.get("review_text_normalized") or record.get("review_text", "")
    tokens = record.get("tokens") or text.split()
    record["feature_token_count"] = len(tokens)
    record["feature_return_hits"] = keyword_hits(text, RETURN_KEYWORDS)
    record["feature_warranty_hits"] = keyword_hits(text, WARRANTY_KEYWORDS)
    record["feature_focus_score"] = record["feature_return_hits"] + record["feature_warranty_hits"]
    return record


def extract_corpus_features(records: List[Dict], top_k: int = 25) -> Dict:
    corpus = [r.get("review_text_normalized", "") or r.get("review_text", "") for r in records]
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=8000,
        stop_words=list(STOPWORDS),
    )
    matrix = vectorizer.fit_transform(corpus)
    mean_weights = np.asarray(matrix.mean(axis=0)).flatten()
    terms = vectorizer.get_feature_names_out()
    top_indices = mean_weights.argsort()[::-1][:top_k]
    top_terms = [
        {"term": terms[i], "avg_tfidf_weight": round(float(mean_weights[i]), 6)} for i in top_indices
    ]
    return {
        "record_count": len(records),
        "vocabulary_size": int(len(terms)),
        "matrix_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "top_corpus_terms": top_terms,
    }


def run_feature_extraction(records: List[Dict]) -> Dict:
    enriched = [enrich_record_features(dict(r)) for r in records]
    corpus_features = extract_corpus_features(enriched)
    issue_counter = Counter()
    for record in enriched:
        for label in record.get("issue_labels", []):
            issue_counter[label] += 1

    summary = {
        "feature_extraction_method": "tfidf + keyword frequency + token statistics",
        "corpus_features": corpus_features,
        "issue_label_distribution": dict(issue_counter),
        "avg_token_count": round(
            sum(r.get("feature_token_count", 0) for r in enriched) / max(len(enriched), 1), 2
        ),
        "records_with_return_signal": sum(1 for r in enriched if r.get("feature_return_hits", 0) > 0),
        "records_with_warranty_signal": sum(1 for r in enriched if r.get("feature_warranty_hits", 0) > 0),
    }
    return enriched, summary


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract IR features for warranty/return complaints")
    parser.add_argument("--input", type=Path, default=Path("data/processed/clean_reviews.json"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/clean_reviews.json"))
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("reports/feature_extraction_summary.json"),
    )
    args = parser.parse_args()

    records = load_records(args.input)
    enriched, summary = run_feature_extraction(records)
    save_json(args.output, enriched)
    save_json(args.summary_output, summary)

    print(f"Feature extraction complete for {len(enriched)} records")
    print(f"Updated dataset: {args.output}")
    print(f"Feature summary: {args.summary_output}")


if __name__ == "__main__":
    main()
