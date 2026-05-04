# Phase 1 Submission Notes (Warranty & Return Analyzer)

## 1) Project Domain Definition & Product Idea Clarity

- Domain: review/complaint intelligence for return and warranty issues.
- Product objective: identify patterns behind refund denials, warranty claim friction, and defect trends.
- Product output: searchable complaint insights + AI-assisted issue categorization and root-cause suggestions.

## 2) Web Scraping & Crawling Implementation

- Multi-page crawling starts from sitemap index and complaint sitemap files.
- Script: `src/scraper/scrape_complaints.py`
- Key implementation points:
  - fetch sitemap index
  - crawl complaint URLs
  - parse structured complaint blocks (`reviewBody`, `ratingValue`, dates, titles)
  - collect minimum target records

## 3) Robots.txt Compliance & Ethical Crawling

- `RobotFileParser` validates crawl permission before every sitemap and detail page request.
- `crawl_delay` is applied between requests.
- Evidence saved in:
  - `data/raw/robots.txt`
  - `data/raw/crawl_meta.json`
  - `data/raw/crawl_log.json`

## 4) Data Storage Design

- Raw extraction: `data/raw/raw_reviews.json`
- Processed dataset: `data/processed/clean_reviews.json`
- AI-enriched dataset: `data/processed/clean_reviews_ai.json`
- Consistent schema across all records; see `README.md` Dataset Schema section.

## 5) Data Cleaning & Preprocessing

- Script: `src/pipeline/preprocess.py`
- Steps:
  - normalization (lowercasing, punctuation cleanup, URL removal)
  - tokenization + stopword removal
  - issue-label tagging
  - mention flags (`has_return_mention`, `has_warranty_mention`)

## 6) Data Quality Handling

- Duplicate handling: URL + normalized text hash strategy.
- Missing value handling:
  - default `unknown` for missing metadata
  - explicit counters in `preprocessing_summary.json`

## 7) EDA (Patterns, Stats, Insights)

- Script: `src/analysis/eda.py`
- Summary outputs:
  - `reports/analysis_summary.json`
  - `reports/analysis_summary.md`
- Includes records count, mention rates, issue frequencies, keyword trends, and reliability averages.

## 8) Data Visualization & Interpretation

- Generated figures:
  - `reports/figures/rating_distribution.png`
  - `reports/figures/top_issue_categories.png`
  - `reports/figures/return_warranty_mentions.png`
  - `reports/figures/issue_trend_over_time.png`
  - `reports/figures/top_keywords.png`

## 9) AI Integration Layer

- Script: `src/analysis/ai_issue_classifier.py`
- Explainable AI feature:
  - TF-IDF + Logistic Regression issue prediction
  - confidence score
  - model-driven root-cause suggestion
- Metrics stored in `reports/ai_metrics.json`

## 10) Product Layer

- Script: `src/product/cli.py`
- Functional interface:
  - query-based retrieval over complaint corpus
  - ranked results
  - issue label and root-cause suggestion shown per result
