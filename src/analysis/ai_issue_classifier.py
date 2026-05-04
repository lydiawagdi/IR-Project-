import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def load_records(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def primary_label(issue_labels: List[str]) -> str:
    if not issue_labels:
        return "other"
    return issue_labels[0]


def suggestion_from_label(label: str) -> str:
    mapping = {
        "defective_product": "Strengthen manufacturing QA and pre-dispatch product testing.",
        "warranty_claim_issue": "Simplify claim workflow and make warranty conditions clearer.",
        "refund_return_issue": "Automate refund approvals for validated return scenarios.",
        "delivery_issue": "Improve courier SLA monitoring and proactive customer notifications.",
        "support_issue": "Introduce support quality audits and escalation checkpoints.",
        "billing_issue": "Add billing anomaly detection before payment capture.",
        "other": "Collect additional training examples for more precise categorization.",
    }
    return mapping.get(label, mapping["other"])


def build_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=8000)),
            ("clf", LogisticRegression(max_iter=600, class_weight="balanced")),
        ]
    )


def train_and_predict(records: List[Dict]) -> Dict:
    texts = [r.get("review_text_normalized", "") or r.get("review_text", "") for r in records]
    labels = [primary_label(r.get("issue_labels", [])) for r in records]

    unique_labels = sorted(set(labels))
    class_counts = Counter(labels)
    min_class_size = min(class_counts.values()) if class_counts else 0

    if len(unique_labels) < 2 or len(records) < 30 or min_class_size < 2:
        for r in records:
            label = primary_label(r.get("issue_labels", []))
            r["ai_predicted_issue"] = label
            r["ai_confidence"] = 0.55
            r["ai_root_cause_suggestion"] = suggestion_from_label(label)
        return {
            "model_used": "fallback_rule_based",
            "reason": "insufficient label diversity, class size, or sample size",
            "labels": unique_labels,
            "class_counts": dict(class_counts),
        }

    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    model = build_model()
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    pred_probs = model.predict_proba(texts)
    pred_all = model.predict(texts)
    class_labels = model.named_steps["clf"].classes_

    report = classification_report(y_test, pred, output_dict=True, zero_division=0)

    for idx, record in enumerate(records):
        label = pred_all[idx]
        confidence = float(np.max(pred_probs[idx]))
        record["ai_predicted_issue"] = label
        record["ai_confidence"] = round(confidence, 4)
        record["ai_root_cause_suggestion"] = suggestion_from_label(label)

    return {
        "model_used": "tfidf_logreg",
        "train_size": len(x_train),
        "test_size": len(x_test),
        "labels": unique_labels,
        "macro_f1": round(report.get("macro avg", {}).get("f1-score", 0.0), 4),
        "weighted_f1": round(report.get("weighted avg", {}).get("f1-score", 0.0), 4),
        "accuracy": round(report.get("accuracy", 0.0), 4),
    }


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train explainable issue classifier and enrich records")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/clean_reviews.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/clean_reviews_ai.json"),
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("reports/ai_metrics.json"),
    )
    args = parser.parse_args()

    records = load_records(args.input)
    metrics = train_and_predict(records)
    save_json(args.output, records)
    save_json(args.metrics_output, metrics)

    print(f"AI enrichment complete for {len(records)} records")
    print(f"Saved enriched dataset to {args.output}")
    print(f"Saved model metrics to {args.metrics_output}")


if __name__ == "__main__":
    main()
