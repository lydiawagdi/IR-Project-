import argparse
import hashlib
import json
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


BASE_URL = "https://www.complaintsboard.com"
ROBOTS_URL = f"{BASE_URL}/robots.txt"
SITEMAP_INDEX = f"{BASE_URL}/sitemap.xml"
USER_AGENT = "IR-Project-WarrantyReturnAnalyzer/1.0 (+course-project)"
SITEMAP_NS = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}

RETURN_KEYWORDS = {
    "return",
    "returned",
    "returning",
    "refund",
    "refunded",
    "refunds",
}
WARRANTY_KEYWORDS = {
    "warranty",
    "warranties",
    "guarantee",
    "guaranteed",
    "claim",
    "claims",
}
FOCUS_KEYWORDS = RETURN_KEYWORDS.union(WARRANTY_KEYWORDS).union(
    {"refund", "replacement", "defect", "defective", "repair", "broken", "damaged"}
)


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    retry = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def load_robots_parser(session: requests.Session) -> RobotFileParser:
    response = session.get(ROBOTS_URL, timeout=30)
    response.raise_for_status()
    parser = RobotFileParser()
    parser.set_url(ROBOTS_URL)
    parser.parse(response.text.splitlines())
    return parser


def parse_sitemap_urls(xml_text: str) -> List[str]:
    root = ET.fromstring(xml_text)
    urls = []
    for loc in root.findall(".//s:loc", SITEMAP_NS):
        if loc.text:
            urls.append(loc.text.strip())
    return urls


def get_complaint_sitemaps(session: requests.Session, max_sitemaps: int) -> List[str]:
    response = session.get(SITEMAP_INDEX, timeout=40)
    response.raise_for_status()
    all_sitemaps = parse_sitemap_urls(response.text)
    complaint_sitemaps = [u for u in all_sitemaps if "/sitemap/complaints-" in u]
    return complaint_sitemaps[:max_sitemaps]


def normalize_whitespace(value: str) -> str:
    return " ".join(value.split()).strip()


def parse_company_name(soup: BeautifulSoup) -> str:
    title_tag = soup.select_one("h1")
    if title_tag:
        title_text = normalize_whitespace(title_tag.get_text(" ", strip=True))
        # Example: "Cognizant review: manager's harassment!"
        match = re.match(r"(.+?)\s+review\s*:", title_text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()

    tag = soup.select_one("span[itemprop=name]")
    if tag and tag.get_text(strip=True):
        candidate = normalize_whitespace(tag.get_text(" ", strip=True))
        if candidate and candidate.upper() != "CB":
            return candidate

    ld_json = soup.select_one('script[type="application/ld+json"]')
    if ld_json and ld_json.get_text(strip=True):
        try:
            payload = json.loads(ld_json.get_text())
            if isinstance(payload, dict) and payload.get("name"):
                return normalize_whitespace(str(payload["name"]))
        except json.JSONDecodeError:
            pass
    return "unknown"


def parse_review_date(container: BeautifulSoup) -> Optional[str]:
    date_tag = container.select_one('[itemprop="datePublished"]')
    if not date_tag:
        return None

    content = date_tag.get("content", "").strip()
    if content:
        return content

    text_value = normalize_whitespace(date_tag.get_text(" ", strip=True))
    return text_value or None


def parse_category(soup: BeautifulSoup) -> str:
    breadcrumb_links = soup.select(".breadcrumbs a[href]")
    if breadcrumb_links:
        candidates = [normalize_whitespace(a.get_text(" ", strip=True)) for a in breadcrumb_links]
        candidates = [c for c in candidates if c and c.lower() not in {"home", "complaints", "companies"}]
        if candidates:
            return candidates[-1].lower().replace(" ", "-")
    return "general-complaints"


def extract_review_record(url: str, html: str) -> Optional[Dict]:
    soup = BeautifulSoup(html, "html.parser")
    body_tag = soup.select_one('[itemprop="reviewBody"]')
    if not body_tag:
        return None

    container = body_tag.find_parent(class_="complaint-main__content")
    if container is None:
        container = body_tag.parent

    title_tag = container.select_one(".complaint-main__header-name") or soup.select_one("h1")
    title = normalize_whitespace(title_tag.get_text(" ", strip=True)) if title_tag else "unknown"

    rating_tag = container.select_one('[itemprop="ratingValue"]')
    rating_raw = rating_tag.get("content") if rating_tag else None
    rating = None
    if rating_raw:
        try:
            rating = float(rating_raw)
        except ValueError:
            rating = None

    review_text = normalize_whitespace(body_tag.get_text(" ", strip=True))
    review_date = parse_review_date(container)
    if not review_date:
        fallback_date = soup.select_one('[itemprop="datePublished"]')
        if fallback_date:
            review_date = (
                fallback_date.get("content", "").strip()
                or normalize_whitespace(fallback_date.get_text(" ", strip=True))
                or None
            )
    source_company = parse_company_name(soup)
    category = parse_category(soup)

    lower_text = review_text.lower()
    has_return = any(k in lower_text for k in RETURN_KEYWORDS)
    has_warranty = any(k in lower_text for k in WARRANTY_KEYWORDS)
    relevance_score = sum(1 for k in FOCUS_KEYWORDS if k in lower_text)
    is_relevant = relevance_score > 0

    record_id = hashlib.md5(url.encode("utf-8")).hexdigest()
    return {
        "record_id": record_id,
        "source": "complaintsboard",
        "company": source_company,
        "category": category,
        "title": title,
        "review_text": review_text,
        "review_date": review_date,
        "rating": rating,
        "has_return_mention": has_return,
        "has_warranty_mention": has_warranty,
        "is_warranty_return_relevant": is_relevant,
        "relevance_score": relevance_score,
        "raw_url": url,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
    }


def scrape_records(
    target_records: int,
    max_sitemaps: int,
    crawl_delay: float,
) -> Dict[str, List[Dict]]:
    session = build_session()
    robots = load_robots_parser(session)

    complaint_sitemaps = get_complaint_sitemaps(session, max_sitemaps=max_sitemaps)
    all_records: List[Dict] = []
    crawl_log: List[Dict] = []
    visited_urls = set()

    for sitemap_url in complaint_sitemaps:
        if len(all_records) >= target_records:
            break

        if not robots.can_fetch(USER_AGENT, sitemap_url):
            crawl_log.append(
                {
                    "url": sitemap_url,
                    "status": "skipped_by_robots",
                    "reason": "robots_disallow",
                }
            )
            continue

        try:
            sitemap_response = session.get(sitemap_url, timeout=40)
            sitemap_response.raise_for_status()
            complaint_urls = parse_sitemap_urls(sitemap_response.text)
        except Exception as exc:
            crawl_log.append(
                {
                    "url": sitemap_url,
                    "status": "error",
                    "reason": f"sitemap_fetch_failed: {exc}",
                }
            )
            continue

        for complaint_url in complaint_urls:
            if len(all_records) >= target_records:
                break

            parsed = urlparse(complaint_url)
            if parsed.netloc and parsed.netloc != urlparse(BASE_URL).netloc:
                continue

            absolute_url = complaint_url if complaint_url.startswith("http") else urljoin(BASE_URL, complaint_url)
            if absolute_url in visited_urls:
                continue
            visited_urls.add(absolute_url)

            if not robots.can_fetch(USER_AGENT, absolute_url):
                crawl_log.append(
                    {
                        "url": absolute_url,
                        "status": "skipped_by_robots",
                        "reason": "robots_disallow",
                    }
                )
                continue

            try:
                response = session.get(absolute_url, timeout=25)
                response.raise_for_status()
                record = extract_review_record(absolute_url, response.text)
                if record:
                    if record["is_warranty_return_relevant"]:
                        all_records.append(record)
                        crawl_log.append({"url": absolute_url, "status": "ok_relevant"})
                    else:
                        crawl_log.append({"url": absolute_url, "status": "skipped_not_relevant"})
                else:
                    crawl_log.append({"url": absolute_url, "status": "no_review_body"})
            except Exception as exc:
                crawl_log.append({"url": absolute_url, "status": "error", "reason": str(exc)})

            time.sleep(crawl_delay)

    return {"records": all_records, "crawl_log": crawl_log, "sitemaps": complaint_sitemaps}


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape complaints/reviews for Warranty & Return Analyzer")
    parser.add_argument("--target-records", type=int, default=120)
    parser.add_argument("--max-sitemaps", type=int, default=2)
    parser.add_argument("--crawl-delay", type=float, default=1.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
    )
    args = parser.parse_args()

    result = scrape_records(
        target_records=args.target_records,
        max_sitemaps=args.max_sitemaps,
        crawl_delay=args.crawl_delay,
    )

    raw_path = args.output_dir / "raw_reviews.json"
    log_path = args.output_dir / "crawl_log.json"
    meta_path = args.output_dir / "crawl_meta.json"
    robots_path = args.output_dir / "robots.txt"

    save_json(raw_path, result["records"])
    save_json(log_path, result["crawl_log"])
    save_json(
        meta_path,
        {
            "target_records": args.target_records,
            "records_collected": len(result["records"]),
            "max_sitemaps": args.max_sitemaps,
            "sitemaps_used": result["sitemaps"],
            "user_agent": USER_AGENT,
            "source": BASE_URL,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    session = build_session()
    robots_txt = session.get(ROBOTS_URL, timeout=20).text
    robots_path.parent.mkdir(parents=True, exist_ok=True)
    robots_path.write_text(robots_txt, encoding="utf-8")

    print(f"Collected {len(result['records'])} records")
    print(f"Saved raw data to {raw_path}")
    print(f"Saved crawl log to {log_path}")


if __name__ == "__main__":
    main()
