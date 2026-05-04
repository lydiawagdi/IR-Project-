# Warranty & Return Analyzer (IR Project - Phase 1)

This project implements a complete Phase 1 pipeline for the **Warranty & Return Analyzer** domain:

1. Web crawling/scraping from real web pages
2. Structured JSON storage
3. Cleaning and preprocessing
4. Data quality handling (missing/noisy/duplicate)
5. Exploratory analysis + visualizations
6. AI-powered issue categorization + root-cause suggestion
7. Product-facing CLI retrieval interface

## Data Source and Ethics

- Source domain: `https://www.complaintsboard.com`
- Entry points:
  - `https://www.complaintsboard.com/sitemap.xml`
  - complaint sitemap files: `/sitemap/complaints-*.xml`
- Robots policy is checked before crawling each sitemap and each complaint URL.
- A local copy of `robots.txt` is saved to `data/raw/robots.txt`.

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
- `rating`: extracted rating (if available)
- `has_return_mention`: boolean keyword flag
- `has_warranty_mention`: boolean keyword flag
- `issue_labels`: rule-based issue categories
- `root_cause_suggestion`: explainable recommendation
- `ai_predicted_issue`: model-predicted issue label
- `ai_confidence`: prediction confidence
- `ai_root_cause_suggestion`: AI-feature suggestion
- `reliability_score`: derived reliability score
- `raw_url`: source URL

## Project Structure

```text
src/
  scraper/scrape_complaints.py
  pipeline/preprocess.py
  analysis/ai_issue_classifier.py
  analysis/eda.py
  product/cli.py
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

### One-command Windows bootstrap

If you are setting up on a new Windows machine, use:

```powershell
powershell -ExecutionPolicy Bypass -File .\bootstrap_windows.ps1
```

This script:
- installs Python 3.12 via `winget` if missing,
- creates `.venv`,
- upgrades `pip`,
- installs all dependencies from `requirements.txt` (including Streamlit).

## Run Pipeline

1) Scrape raw records (multi-page crawl via sitemaps):

```bash
python src/scraper/scrape_complaints.py --target-records 120 --max-sitemaps 2 --crawl-delay 1.0
```

2) Clean and preprocess:

```bash
python src/pipeline/preprocess.py
```

3) AI integration (explainable classifier):

```bash
python src/analysis/ai_issue_classifier.py
```

4) EDA and visualizations:

```bash
python src/analysis/eda.py
```

5) Product-layer CLI search:

```bash
python src/product/cli.py "refund denied after warranty claim" --top-k 5
```

6) Product-layer GUI (Streamlit dashboard):

```bash
streamlit run src/product/app.py
```

Inside the GUI, open the **Pipeline Runner** tab and click **Run Pipeline Now** to execute:
- scraper (optional)
- preprocessing
- AI enrichment
- EDA/chart generation

## Outputs

- Raw dataset: `data/raw/raw_reviews.json`
- Crawl evidence/logs: `data/raw/crawl_log.json`, `data/raw/crawl_meta.json`, `data/raw/robots.txt`
- Clean dataset: `data/processed/clean_reviews.json`
- AI-enriched dataset: `data/processed/clean_reviews_ai.json`
- Preprocessing summary: `data/processed/preprocessing_summary.json`
- AI metrics: `reports/ai_metrics.json`
- EDA summary: `reports/analysis_summary.json`, `reports/analysis_summary.md`
- Charts:
  - `reports/figures/rating_distribution.png`
  - `reports/figures/top_issue_categories.png`
  - `reports/figures/return_warranty_mentions.png`
  - `reports/figures/issue_trend_over_time.png`
  - `reports/figures/top_keywords.png`

## Rubric Mapping (Phase 1)

- **Domain & idea clarity**: Warranty & Return Analyzer scope and objectives
- **Scraping/crawling**: `src/scraper/scrape_complaints.py`
- **Robots compliance**: robots parser checks + saved robots evidence
- **Storage design**: structured JSON schema in `data/raw/` and `data/processed/`
- **Cleaning/preprocessing**: `src/pipeline/preprocess.py`
- **Data quality handling**: dedupe + missing/noisy handling in preprocessing summary
- **EDA**: `src/analysis/eda.py` + generated insights
- **Visualization**: charts in `reports/figures/`
- **AI feature**: `src/analysis/ai_issue_classifier.py`
- **Product interface**:
  - searchable CLI in `src/product/cli.py`
  - Streamlit GUI dashboard in `src/product/app.py`