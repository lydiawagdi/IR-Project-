import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_records(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def search(records: List[Dict], query: str, top_k: int = 5) -> List[Dict]:
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


def print_results(results: List[Dict]) -> None:
    if not results:
        print("No matching records found.")
        return

    for i, rec in enumerate(results, start=1):
        print("=" * 72)
        print(f"Result #{i} | Similarity: {rec.get('similarity')}")
        print(f"Company: {rec.get('company')}")
        print(f"Title: {rec.get('title')}")
        print(f"AI Predicted Issue: {rec.get('ai_predicted_issue')}")
        print(f"AI Confidence: {rec.get('ai_confidence')}")
        print(f"Root Cause Suggestion: {rec.get('ai_root_cause_suggestion')}")
        print(f"Rating: {rec.get('rating')} | Date: {rec.get('review_date')}")
        print(f"URL: {rec.get('raw_url')}")
        print("-" * 72)
        print((rec.get("review_text", "") or "")[:700])


def main() -> None:
    parser = argparse.ArgumentParser(description="Warranty & Return Analyzer CLI Search")
    parser.add_argument("query", type=str, help="Search query for complaint retrieval")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/clean_reviews_ai.json"),
    )
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    records = load_records(args.input)
    results = search(records, args.query, top_k=args.top_k)
    print_results(results)


if __name__ == "__main__":
    main()
