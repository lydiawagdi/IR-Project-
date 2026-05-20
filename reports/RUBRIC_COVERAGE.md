# Rubric Coverage Map

This document maps course rubric requirements to concrete project artifacts.

## 1) Feature Extraction

- **Implementation:** `src/analysis/feature_extraction.py`
- **Outputs:** `reports/feature_extraction_summary.json`, per-record fields:
  - `feature_token_count`
  - `feature_return_hits`
  - `feature_warranty_hits`
  - `feature_focus_score`
- **Methods:** TF-IDF corpus term weighting, keyword frequency features, token statistics.

## 2) Product System Implementation

- **GUI:** `src/product/app.py` (search, filters, dataset preview, charts, pipeline runner, insights)
- **CLI:** `src/product/cli.py` (TF-IDF complaint search)
- **Pipeline orchestration:** sidebar and tab controls in GUI.

## 3) AI Feature Integration (sentiment, classification, summarization)

- **Implementation:** `src/analysis/ai_enrichment.py`
- **Features added to each record:**
  - `ai_predicted_issue`, `ai_confidence`
  - `ai_sentiment`, `ai_sentiment_score`
  - `ai_summary`
  - `ai_root_cause_suggestion`

## 4) AI Feature Relevance to Product

- **Field:** `ai_domain_relevance` explains why each AI output supports warranty/return analysis.
- **Domain labels and keyword rules:** warranty/return issue taxonomy in preprocessing and classifier.

## 5) Insight Generation & Interpretation Quality

- **Implementation:** `src/analysis/generate_insights.py`
- **Outputs:** `reports/project_insights.md`, `reports/project_insights.json`
- **EDA support:** `src/analysis/eda.py` + charts in GUI.

## 6) Evaluation of Outputs

- **Implementation:** `src/analysis/evaluate_outputs.py`
- **Outputs:** `reports/evaluation_report.md`, `reports/evaluation_report.json`
- **Checks:**
  - TF-IDF retrieval keyword hit-rate on benchmark queries
  - AI classification accuracy vs rule labels
  - AI output coverage (sentiment/summary/classification)

## 7) Code Structure, Modularity & Reproducibility

- Modular folders:
  - `src/scraper/`
  - `src/pipeline/`
  - `src/analysis/`
  - `src/product/`
- Reproducible commands documented in `README.md`.
- Deterministic scripts with CLI arguments and saved JSON/MD reports.

## Recommended Run Order

```bash
python src/scraper/scrape_complaints.py --target-records 25 --light-mode
python src/pipeline/preprocess.py
python src/analysis/feature_extraction.py
python src/analysis/ai_enrichment.py
python src/analysis/evaluate_outputs.py
python src/analysis/generate_insights.py
python src/analysis/eda.py
python src/product/app.py
```
