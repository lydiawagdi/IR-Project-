# Warranty & Return Analyzer (IR Project - Phase 1 & 2)

This project implements a complete pipeline for the **Warranty & Return Analyzer** domain:

1. **Selenium web scraping** from ComplaintsBoard pages (categories, subpages, and search)
2. Structured JSON storage
3. Cleaning and preprocessing
4. Data quality handling (missing/noisy/duplicate)
5. Exploratory analysis + visualizations
6. Feature extraction (TF-IDF + keyword/token signals)
7. AI enrichment (classification, sentiment, summarization, domain relevance)
8. Output evaluation + automated insight reports
9. Exploratory analysis + visualizations
10. Product-facing CLI and Streamlit GUI (IR search + AI + dashboards)

## Data Source and Ethics

- Source domain: [ComplaintsBoard](https://www.complaintsboard.com)
- Primary entry points (web navigation, not XML feeds):
  - Categories hub: [https://www.complaintsboard.com/categories](https://www.complaintsboard.com/categories)
  - Warranty/return search: [https://www.complaintsboard.com/?search=warranty+and+return](https://www.complaintsboard.com/?search=warranty+and+return)
- Scraper engine: **Selenium + Chrome** (handles JavaScript-rendered pages)
- Robots policy is checked before crawling each URL.
- A local copy of `robots.txt` is saved to `data/raw/robots.txt`.

## Scraping Workflow

1. Open categories page and discover subcategory links.
2. Crawl subcategory listing pages and collect complaint URLs.
3. Crawl warranty/return search result pages (with pagination).
4. Visit each complaint detail page and extract structured fields.
5. Keep only warranty/return relevant records for the dataset.
6. Sort search results by date and drop complaints older than `--min-year` (default: current year − 4).

## Dataset Schema

Each record follows a consistent JSON schema:

- `record_id`: stable hash ID
- `source`: source website
- `company`: complaint target company
- `category`: project category
- `title`: complaint/review title
- `review_text`: original extracted complaint text
- `review_text_normalized`: cleaned version
- `tokens`: tokenized and stopword-filtered tokens
- `review_date`: extracted date (if available)
- `review_updated_date`: last update timestamp (if available)
- `reviewer_name` / `reviewer_location`: complaint author metadata
- `breadcrumbs`: full category path from page layout
- `comment_count`: number of comments shown on page
- `helpful_count`: number of users who marked review helpful
- `claimed_loss`: structured "Claimed loss" field when present
- `desired_outcome`: structured "Desired outcome" field when present
- `is_featured_review`: featured badge flag
- `comments`: list of comment objects (`author_name`, `posted_at`, `text`, `helpful_count`, ...)
- `full_text_with_comments`: combined text used for relevance/IR analysis
- `rating`: extracted rating (if available)
- `has_return_mention`: boolean keyword flag
- `has_warranty_mention`: boolean keyword flag
- `issue_labels`: rule-based issue categories
- `root_cause_suggestion`: explainable recommendation
- `ai_predicted_issue`: model-predicted issue label
- `ai_confidence`: prediction confidence
- `ai_sentiment` / `ai_sentiment_score`: lexicon-based sentiment
- `ai_summary`: extractive complaint summary
- `ai_domain_relevance`: why AI outputs support warranty/return analysis
- `ai_root_cause_suggestion`: AI-feature suggestion
- `feature_*`: token/keyword/TF-IDF feature fields from feature extraction
- `reliability_score`: derived reliability score
- `discovery_source`: `search` or `category_navigation`
- `raw_url`: source URL

## Project Structure

```text
src/
  scraper/scrape_complaints.py
  pipeline/preprocess.py
  analysis/feature_extraction.py
  analysis/ai_enrichment.py
  analysis/evaluate_outputs.py
  analysis/generate_insights.py
  analysis/ai_issue_classifier.py
  analysis/eda.py
  product/cli.py
  product/app.py
data/
  raw/
  processed/
reports/
  figures/
```

## Setup

```bash
python -m pip install -r requirements.txt
```

Requires Google Chrome installed (used by Selenium via `webdriver-manager`).

### One-command Windows bootstrap

```powershell
powershell -ExecutionPolicy Bypass -File .\bootstrap_windows.ps1
```

## Run Pipeline

1) Scrape raw records (Selenium web navigation):

```bash
python src/scraper/scrape_complaints.py --target-records 120 --max-categories 8 --max-search-pages 5 --crawl-delay 1.0
```

2) Clean and preprocess:

```bash
python src/pipeline/preprocess.py
```

3) Feature extraction:

```bash
python src/analysis/feature_extraction.py
```

4) AI enrichment (classification + sentiment + summarization):

```bash
python src/analysis/ai_enrichment.py
```

5) Evaluate outputs (retrieval + AI quality):

```bash
python src/analysis/evaluate_outputs.py
```

6) Generate project insights:

```bash
python src/analysis/generate_insights.py
```

7) EDA and visualizations:

```bash
python src/analysis/eda.py
```

8) Product-layer CLI search:

```bash
python src/product/cli.py "refund denied after warranty claim" --top-k 5
```

9) Product-layer GUI (Streamlit dashboard):

```bash
python src/product/app.py
```

Inside the GUI, use **Run Pipeline Now** (sidebar or Pipeline Runner tab) to execute:
- Selenium scraper (optional)
- preprocessing
- feature extraction
- AI enrichment (classification, sentiment, summarization)
- output evaluation
- insights report
- EDA/chart generation (optional)

Open the **Insights & Evaluation** tab for rubric coverage, `project_insights.md`, and `evaluation_report.md`.

## Rubric Coverage

See `reports/RUBRIC_COVERAGE.md` for a requirement-by-requirement map (feature extraction, product system, AI integration, relevance, insights, evaluation, reproducibility).

## Phase 2 Alignment

- **Phase 1 feedback applied**: real website navigation with Selenium instead of XML sitemap-only scraping.
- **IR techniques**: TF-IDF ranking, keyword statistics, indexing over processed corpus.
- **AI model**: scikit-learn issue classifier trained on collected data and used in GUI/CLI.
- **GUI**: interactive search, dataset preview, EDA charts, and pipeline workflow controls.

## Outputs

- Raw dataset: `data/raw/raw_reviews.json`
- Crawl evidence/logs: `data/raw/crawl_log.json`, `data/raw/crawl_meta.json`, `data/raw/robots.txt`
- Clean dataset: `data/processed/clean_reviews.json`
- AI-enriched dataset: `data/processed/clean_reviews_ai.json`
- Preprocessing summary: `data/processed/preprocessing_summary.json`
- AI metrics: `reports/ai_metrics.json`
- Feature extraction summary: `reports/feature_extraction_summary.json`
- Evaluation report: `reports/evaluation_report.md`, `reports/evaluation_report.json`
- Project insights: `reports/project_insights.md`, `reports/project_insights.json`
- Rubric map: `reports/RUBRIC_COVERAGE.md`
- EDA summary: `reports/analysis_summary.json`, `reports/analysis_summary.md`
- Charts in `reports/figures/`
