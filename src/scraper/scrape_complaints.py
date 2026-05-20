import argparse
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager


BASE_URL = "https://www.complaintsboard.com"
CATEGORIES_URL = f"{BASE_URL}/categories"
SEARCH_URL = f"{BASE_URL}/?search=warranty+and+return"
ROBOTS_URL = f"{BASE_URL}/robots.txt"
USER_AGENT = "IR-Project-WarrantyReturnAnalyzer/2.0 (+course-project; selenium)"
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


def normalize_whitespace(value: str) -> str:
    return " ".join(value.split()).strip()


def extract_year_from_date(date_value: Optional[str]) -> Optional[int]:
    if not date_value or str(date_value).strip().lower() == "unknown":
        return None
    match = re.search(r"\b(19|20)\d{2}\b", str(date_value))
    return int(match.group(0)) if match else None


def default_min_year() -> int:
    return datetime.now(timezone.utc).year - 4


def record_year(record: Dict) -> Optional[int]:
    years: List[int] = []
    for field in ("review_date", "review_updated_date"):
        year = extract_year_from_date(record.get(field))
        if year:
            years.append(year)
    return max(years) if years else None


def is_recent_enough(record: Dict, min_year: int) -> bool:
    year = record_year(record)
    if year is None:
        return False
    return year >= min_year


def load_robots_parser() -> RobotFileParser:
    response = requests.get(ROBOTS_URL, timeout=30, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    parser = RobotFileParser()
    parser.set_url(ROBOTS_URL)
    parser.parse(response.text.splitlines())
    return parser


def build_driver(headless: bool, light_mode: bool = False) -> webdriver.Chrome:
    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-notifications")
    options.add_argument(f"--user-agent={USER_AGENT}")

    if light_mode:
        options.page_load_strategy = "eager"
        options.add_argument("--blink-settings=imagesEnabled=false")
        options.add_argument("--disable-background-networking")
        options.add_argument("--disable-background-timer-throttling")
        options.add_argument("--disable-renderer-backgrounding")
        options.add_argument("--window-size=1280,720")
        options.add_experimental_option(
            "prefs",
            {
                "profile.managed_default_content_settings.images": 2,
                "profile.default_content_setting_values.notifications": 2,
            },
        )
    else:
        options.add_argument("--window-size=1920,1080")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.set_page_load_timeout(25 if light_mode else 40)
    return driver


def can_fetch(robots: RobotFileParser, url: str) -> bool:
    return robots.can_fetch(USER_AGENT, url)


def absolute_url(href: str) -> str:
    return urljoin(BASE_URL, href.strip())


def is_complaint_url(url: str) -> bool:
    path = urlparse(url.split("?")[0]).path
    return bool(re.search(r"-c\d+$", path, re.I))


def normalize_complaint_url(url: str) -> str:
    parsed = urlparse(url.split("?")[0])
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")


def parse_company_name(soup: BeautifulSoup) -> str:
    title_tag = soup.select_one("h1")
    if title_tag:
        title_text = normalize_whitespace(title_tag.get_text(" ", strip=True))
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


def parse_breadcrumbs(soup: BeautifulSoup) -> List[str]:
    crumbs: List[str] = []
    for selector in (".breadcrumbs a[href]", "nav.breadcrumb a[href]", '[itemprop="breadcrumb"] a[href]'):
        links = soup.select(selector)
        if links:
            crumbs = [normalize_whitespace(a.get_text(" ", strip=True)) for a in links]
            crumbs = [c for c in crumbs if c]
            if crumbs:
                break
    return crumbs


def parse_category(soup: BeautifulSoup) -> str:
    crumbs = parse_breadcrumbs(soup)
    if crumbs:
        filtered = [c for c in crumbs if c.lower() not in {"home", "complaints", "companies", "cb"}]
        if filtered:
            return filtered[-1].lower().replace(" ", "-")
    return "general-complaints"


def parse_comment_count(title: str, soup: BeautifulSoup) -> int:
    title_match = re.search(r"\(\s*(\d+)\s*comments?\s*\)", title or "", flags=re.I)
    if title_match:
        return int(title_match.group(1))

    page_match = re.search(r"(\d+)\s+comments?", soup.get_text(" ", strip=True), flags=re.I)
    if page_match:
        return int(page_match.group(1))
    return 0


def parse_helpful_count(soup: BeautifulSoup, scope: Optional[BeautifulSoup] = None) -> Optional[int]:
    root = scope if scope is not None else soup
    text = root.get_text(" ", strip=True)
    match = re.search(r"Helpful\s+(\d+)", text, flags=re.I)
    if match:
        return int(match.group(1))
    return None


def parse_labeled_field(soup: BeautifulSoup, label: str) -> Optional[str]:
    pattern = re.compile(rf"{re.escape(label)}\s*:?\s*", flags=re.I)
    for node in soup.find_all(string=pattern):
        parent = node.parent
        if not parent:
            continue
        combined = normalize_whitespace(parent.get_text(" ", strip=True))
        match = pattern.search(combined)
        if match:
            value = combined[match.end() :].strip()
            if value:
                return value

    full_text = soup.get_text("\n", strip=True)
    for line in full_text.splitlines():
        line = normalize_whitespace(line)
        match = pattern.search(line)
        if match:
            value = line[match.end() :].strip()
            if value:
                return value
    return None


def parse_reviewer_info(soup: BeautifulSoup, container: BeautifulSoup) -> Dict[str, Optional[str]]:
    reviewer_name = None
    reviewer_location = None

    author_tag = container.select_one('[itemprop="author"]') or soup.select_one('[itemprop="author"]')
    if author_tag:
        reviewer_name = normalize_whitespace(author_tag.get_text(" ", strip=True))

    meta_text = ""
    for selector in (
        ".complaint-main__user",
        ".complaint-main__info",
        ".review-author",
        ".complaint-author",
    ):
        block = container.select_one(selector) or soup.select_one(selector)
        if block:
            meta_text = normalize_whitespace(block.get_text(" ", strip=True))
            break

    if not meta_text:
        date_tag = container.select_one('[itemprop="datePublished"]') or soup.select_one('[itemprop="datePublished"]')
        if date_tag and date_tag.parent:
            meta_text = normalize_whitespace(date_tag.parent.get_text(" ", strip=True))

    if meta_text:
        location_match = re.search(r"\bof\s+([A-Z]{2,3}|[A-Za-z][A-Za-z\s]+)\b", meta_text)
        if location_match:
            reviewer_location = location_match.group(1).strip()
        name_match = re.match(
            r"^([A-Za-z][A-Za-z .'-]+?)(?:\s+of\s+[A-Z]{2,3}|\s*\||\s*Jan|\s*Feb|\s*Mar|\s*Apr|\s*May|\s*Jun|\s*Jul|\s*Aug|\s*Sep|\s*Oct|\s*Nov|\s*Dec)",
            meta_text,
        )
        if name_match and not reviewer_name:
            reviewer_name = name_match.group(1).strip()

    return {
        "reviewer_name": reviewer_name,
        "reviewer_location": reviewer_location,
    }


def parse_review_updated_date(soup: BeautifulSoup) -> Optional[str]:
    text = soup.get_text(" ", strip=True)
    match = re.search(r"Review updated:\s*([A-Za-z]{3,9}\s+\d{1,2},\s+\d{4})", text, flags=re.I)
    if match:
        return match.group(1).strip()
    return None


def parse_is_featured_review(soup: BeautifulSoup) -> bool:
    text = soup.get_text(" ", strip=True).lower()
    return "featured review" in text


def parse_comment_block(block: BeautifulSoup) -> Optional[Dict]:
    text_tag = (
        block.select_one('[itemprop="commentText"]')
        or block.select_one(".comment-text")
        or block.select_one(".comment-body")
        or block.select_one("p")
    )
    comment_text = normalize_whitespace(text_tag.get_text(" ", strip=True)) if text_tag else ""
    if not comment_text or len(comment_text) < 8:
        full_text = normalize_whitespace(block.get_text(" ", strip=True))
        if len(full_text) < 12:
            return None
        comment_text = full_text

    author_tag = block.select_one('[itemprop="author"]') or block.select_one(".comment-author")
    author_name = normalize_whitespace(author_tag.get_text(" ", strip=True)) if author_tag else None

    date_tag = block.select_one('[itemprop="datePublished"]') or block.select_one("time")
    posted_at = None
    if date_tag:
        posted_at = date_tag.get("content", "").strip() or normalize_whitespace(date_tag.get_text(" ", strip=True))

    location_match = re.search(r"\bof\s+([A-Z]{2,3}|[A-Za-z][A-Za-z\s]+)\b", comment_text)
    author_location = location_match.group(1).strip() if location_match else None

    comment_id_source = f"{author_name}|{posted_at}|{comment_text[:120]}"
    return {
        "comment_id": hashlib.md5(comment_id_source.encode("utf-8")).hexdigest(),
        "author_name": author_name,
        "author_location": author_location,
        "posted_at": posted_at,
        "text": comment_text,
        "helpful_count": parse_helpful_count(block),
        "is_company_response": "company response" in comment_text.lower(),
    }


def parse_comments(soup: BeautifulSoup) -> List[Dict]:
    comments: List[Dict] = []
    seen_ids: Set[str] = set()

    selectors = [
        '[itemprop="comment"]',
        ".complaint-comment",
        ".comment-item",
        ".comments-list .comment",
        ".comment-list .comment",
        "article.comment",
        ".comment-block",
    ]
    blocks: List[BeautifulSoup] = []
    for selector in selectors:
        found = soup.select(selector)
        if found:
            blocks.extend(found)

    if not blocks:
        comments_header = soup.find(string=re.compile(r"comments?", flags=re.I))
        if comments_header:
            section = comments_header.find_parent(["section", "div", "article"])
            if section:
                blocks = section.select("article, li, .comment, .reply")

    for block in blocks:
        parsed = parse_comment_block(block)
        if not parsed:
            continue
        if parsed["comment_id"] in seen_ids:
            continue
        seen_ids.add(parsed["comment_id"])
        comments.append(parsed)

    return comments


def parse_rating(soup: BeautifulSoup, container: BeautifulSoup) -> Optional[float]:
    rating_tag = container.select_one('[itemprop="ratingValue"]') or soup.select_one('[itemprop="ratingValue"]')
    if rating_tag:
        rating_raw = rating_tag.get("content") or rating_tag.get_text(" ", strip=True)
        if rating_raw:
            try:
                value = float(str(rating_raw).strip())
                if 0 < value <= 5:
                    return value
            except ValueError:
                pass

    for script in soup.select('script[type="application/ld+json"]'):
        try:
            payload = json.loads(script.get_text())
        except json.JSONDecodeError:
            continue
        candidates = payload if isinstance(payload, list) else [payload]
        for item in candidates:
            if not isinstance(item, dict):
                continue
            rating_block = item.get("reviewRating") or item.get("aggregateRating")
            if isinstance(rating_block, dict) and rating_block.get("ratingValue") is not None:
                try:
                    value = float(rating_block["ratingValue"])
                    if 0 < value <= 5:
                        return value
                except (TypeError, ValueError):
                    continue

    return None


def build_full_text(review_text: str, comments: List[Dict], desired_outcome: Optional[str]) -> str:
    chunks = [review_text]
    if desired_outcome:
        chunks.append(desired_outcome)
    for comment in comments:
        chunks.append(comment.get("text", ""))
    return normalize_whitespace(" ".join(chunk for chunk in chunks if chunk))


def extract_review_record(url: str, html: str, discovery_source: str = "unknown") -> Optional[Dict]:
    soup = BeautifulSoup(html, "html.parser")
    body_tag = soup.select_one('[itemprop="reviewBody"]')
    if not body_tag:
        return None

    container = body_tag.find_parent(class_="complaint-main__content")
    if container is None:
        container = body_tag.parent

    title_tag = container.select_one(".complaint-main__header-name") or soup.select_one("h1")
    title = normalize_whitespace(title_tag.get_text(" ", strip=True)) if title_tag else "unknown"

    rating = parse_rating(soup, container)

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
    breadcrumbs = parse_breadcrumbs(soup)
    reviewer_info = parse_reviewer_info(soup, container)
    review_updated_date = parse_review_updated_date(soup)
    comment_count = parse_comment_count(title, soup)
    helpful_count = parse_helpful_count(soup, container)
    claimed_loss = parse_labeled_field(soup, "Claimed loss")
    desired_outcome = parse_labeled_field(soup, "Desired outcome")
    is_featured_review = parse_is_featured_review(soup)
    comments = parse_comments(soup)
    full_text_with_comments = build_full_text(review_text, comments, desired_outcome)

    lower_text = full_text_with_comments.lower()
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
        "breadcrumbs": breadcrumbs,
        "title": title,
        "review_text": review_text,
        "review_date": review_date,
        "review_updated_date": review_updated_date,
        "reviewer_name": reviewer_info["reviewer_name"],
        "reviewer_location": reviewer_info["reviewer_location"],
        "rating": rating,
        "comment_count": comment_count,
        "helpful_count": helpful_count,
        "claimed_loss": claimed_loss,
        "desired_outcome": desired_outcome,
        "is_featured_review": is_featured_review,
        "comments": comments,
        "full_text_with_comments": full_text_with_comments,
        "has_return_mention": has_return,
        "has_warranty_mention": has_warranty,
        "is_warranty_return_relevant": is_relevant,
        "relevance_score": relevance_score,
        "raw_url": url,
        "discovery_source": discovery_source,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
    }


def extract_links_from_html(html: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    links: List[str] = []
    for anchor in soup.select("a[href]"):
        href = anchor.get("href", "").strip()
        if not href or href.startswith("#") or href.lower().startswith("javascript:"):
            continue
        url = absolute_url(href)
        if urlparse(url).netloc.replace("www.", "") != "complaintsboard.com":
            continue
        if is_complaint_url(url):
            links.append(normalize_complaint_url(url))
    return links


def is_subcategory_url(url: str) -> bool:
    path = urlparse(url).path.lower().rstrip("/")
    if path in {"/categories", ""}:
        return False
    if path.startswith("/categories/"):
        return True
    # Category landing pages often use slug paths from the categories hub.
    blocked = {
        "/login",
        "/signup",
        "/about",
        "/contact",
        "/terms",
        "/privacy",
        "/faq",
        "/business",
    }
    return path not in blocked and not is_complaint_url(url) and "-c" not in path


def wait_for_page(driver: webdriver.Chrome, timeout: int = 20) -> None:
    WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.TAG_NAME, "body")))


def apply_search_date_sort(driver: webdriver.Chrome, crawl_log: List[Dict]) -> bool:
    """Prefer newest complaints first on warranty/return search results."""
    selectors = [
        (By.LINK_TEXT, "Date"),
        (By.PARTIAL_LINK_TEXT, "Date"),
        (By.XPATH, "//a[contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'date')]"),
        (By.XPATH, "//button[contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'date')]"),
    ]
    for by, value in selectors:
        try:
            elements = driver.find_elements(by, value)
            for element in elements:
                if not element.is_displayed():
                    continue
                label = normalize_whitespace(element.text).lower()
                if "date" not in label or "update" in label:
                    continue
                element.click()
                time.sleep(1.0)
                crawl_log.append({"status": "search_sorted_by_date", "selector": f"{by}={value}"})
                return True
        except Exception:
            continue

    crawl_log.append({"status": "search_sort_by_date_unavailable"})
    return False


def safe_get(
    driver: webdriver.Chrome,
    url: str,
    robots: RobotFileParser,
    crawl_log: List[Dict],
    delay: float,
    scroll_for_comments: bool = False,
) -> Optional[str]:
    if not can_fetch(robots, url):
        crawl_log.append({"url": url, "status": "skipped_by_robots", "reason": "robots_disallow"})
        return None
    try:
        driver.get(url)
        wait_for_page(driver)
        if scroll_for_comments:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(max(delay, 0.8))
        time.sleep(delay)
        return driver.page_source
    except (TimeoutException, WebDriverException) as exc:
        crawl_log.append({"url": url, "status": "error", "reason": str(exc)})
        return None


def collect_category_subcategory_urls(
    driver: webdriver.Chrome,
    robots: RobotFileParser,
    crawl_log: List[Dict],
    max_categories: int,
    max_subcategories_per_category: int,
    crawl_delay: float,
) -> List[str]:
    html = safe_get(driver, CATEGORIES_URL, robots, crawl_log, crawl_delay)
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    subcategory_urls: List[str] = []
    seen: Set[str] = set()

    # Each category card contains subcategory links and a "Go to category" link.
    category_blocks = soup.select("section, article, div")
    cards = []
    for block in category_blocks:
        heading = block.select_one("h2, h3, .h3, strong")
        sub_links = block.select("a[href]")
        if heading and len(sub_links) >= 2:
            cards.append(block)

    if not cards:
        cards = [soup]

    for card in cards[:max_categories]:
        links = []
        for anchor in card.select("a[href]"):
            href = anchor.get("href", "").strip()
            text = normalize_whitespace(anchor.get_text(" ", strip=True)).lower()
            if not href:
                continue
            url = absolute_url(href)
            if text == "go to category" or is_subcategory_url(url):
                links.append(url)

        unique_links = []
        for url in links:
            if url not in seen:
                seen.add(url)
                unique_links.append(url)
        subcategory_urls.extend(unique_links[:max_subcategories_per_category])

    # Fallback: collect category/subcategory links from full categories page.
    if len(subcategory_urls) < max_categories:
        for anchor in soup.select("a[href]"):
            url = absolute_url(anchor.get("href", "").strip())
            if is_subcategory_url(url) and url not in seen:
                seen.add(url)
                subcategory_urls.append(url)
            if len(subcategory_urls) >= max_categories * max_subcategories_per_category:
                break

    crawl_log.append(
        {
            "url": CATEGORIES_URL,
            "status": "categories_discovered",
            "subcategory_count": len(subcategory_urls),
        }
    )
    return subcategory_urls


def collect_complaint_urls_from_listing(
    driver: webdriver.Chrome,
    listing_url: str,
    robots: RobotFileParser,
    crawl_log: List[Dict],
    crawl_delay: float,
    max_pages: int,
) -> List[str]:
    collected: List[str] = []
    seen_pages: Set[str] = set()
    page_url = listing_url

    for _ in range(max_pages):
        if page_url in seen_pages:
            break
        seen_pages.add(page_url)

        html = safe_get(driver, page_url, robots, crawl_log, crawl_delay)
        if not html:
            break

        if "search=" in listing_url.lower() and page_url == listing_url and len(seen_pages) == 1:
            apply_search_date_sort(driver, crawl_log)
            html = driver.page_source

        links = extract_links_from_html(html)
        for link in links:
            if link not in collected:
                collected.append(link)

        soup = BeautifulSoup(html, "html.parser")
        next_url = None
        for anchor in soup.select("a[href]"):
            text = normalize_whitespace(anchor.get_text(" ", strip=True))
            if text in {"Next", "›", "»"} or re.fullmatch(r"\d+", text):
                candidate = absolute_url(anchor.get("href", ""))
                if candidate not in seen_pages and candidate != page_url:
                    if text == "Next" or text in {"›", "»"}:
                        next_url = candidate
                        break
                    if re.fullmatch(r"\d+", text):
                        # Use the next numbered page when available.
                        current_page_match = re.search(r"[?&]page=(\d+)", page_url)
                        current_page = int(current_page_match.group(1)) if current_page_match else 1
                        if text == str(current_page + 1):
                            next_url = candidate

        if not next_url:
            break
        page_url = next_url

    crawl_log.append(
        {
            "url": listing_url,
            "status": "listing_scraped",
            "complaint_links_found": len(collected),
        }
    )
    return collected


def scrape_records(
    target_records: int,
    max_categories: int,
    max_subcategories_per_category: int,
    max_search_pages: int,
    crawl_delay: float,
    headless: bool,
    use_categories: bool,
    use_search: bool,
    light_mode: bool = False,
    min_year: Optional[int] = None,
) -> Dict:
    min_year = min_year or default_min_year()
    robots = load_robots_parser()
    driver = build_driver(headless=headless, light_mode=light_mode)
    all_records: List[Dict] = []
    crawl_log: List[Dict] = []
    visited_complaints: Set[str] = set()
    queued_complaints: List[str] = []
    queue_cap = int(target_records * (1.5 if light_mode else 3))
    scroll_comments = not light_mode
    listing_pages_per_subcategory = 1 if light_mode else 2
    driver_restart_every = 12 if light_mode else 25
    pages_since_restart = 0

    def restart_driver_if_needed() -> None:
        nonlocal driver, pages_since_restart
        if pages_since_restart < driver_restart_every:
            return
        try:
            driver.quit()
        except Exception:
            pass
        driver = build_driver(headless=headless, light_mode=light_mode)
        pages_since_restart = 0
        crawl_log.append({"status": "driver_restarted", "reason": "memory_safety"})

    try:
        if use_search and can_fetch(robots, SEARCH_URL):
            search_links = collect_complaint_urls_from_listing(
                driver,
                SEARCH_URL,
                robots,
                crawl_log,
                crawl_delay,
                max_pages=max_search_pages,
            )
            for link in search_links:
                if link not in queued_complaints:
                    queued_complaints.append(link)
            crawl_log.append(
                {
                    "url": SEARCH_URL,
                    "status": "search_seed_complete",
                    "queued_complaints": len(queued_complaints),
                }
            )

        if use_categories and can_fetch(robots, CATEGORIES_URL):
            subcategory_urls = collect_category_subcategory_urls(
                driver,
                robots,
                crawl_log,
                max_categories=max_categories,
                max_subcategories_per_category=max_subcategories_per_category,
                crawl_delay=crawl_delay,
            )
            for sub_url in subcategory_urls:
                if len(queued_complaints) >= queue_cap:
                    break
                listing_links = collect_complaint_urls_from_listing(
                    driver,
                    sub_url,
                    robots,
                    crawl_log,
                    crawl_delay,
                    max_pages=listing_pages_per_subcategory,
                )
                for link in listing_links:
                    if link not in queued_complaints:
                        queued_complaints.append(link)

        for complaint_url in queued_complaints:
            if len(all_records) >= target_records:
                break
            if complaint_url in visited_complaints:
                continue
            visited_complaints.add(complaint_url)
            restart_driver_if_needed()

            html = safe_get(
                driver,
                complaint_url,
                robots,
                crawl_log,
                crawl_delay,
                scroll_for_comments=scroll_comments,
            )
            pages_since_restart += 1
            if not html:
                continue

            discovery_source = "search" if "search=warranty" in complaint_url else "category_navigation"
            record = extract_review_record(complaint_url, html, discovery_source=discovery_source)
            if not record:
                crawl_log.append({"url": complaint_url, "status": "no_review_body"})
                continue

            if not record["is_warranty_return_relevant"]:
                crawl_log.append({"url": complaint_url, "status": "skipped_not_relevant"})
                continue

            if not is_recent_enough(record, min_year):
                crawl_log.append(
                    {
                        "url": complaint_url,
                        "status": "skipped_old_year",
                        "review_date": record.get("review_date"),
                        "min_year": min_year,
                    }
                )
                continue

            all_records.append(record)
            crawl_log.append(
                {
                    "url": complaint_url,
                    "status": "ok_relevant",
                    "review_year": record_year(record),
                }
            )
    finally:
        try:
            driver.quit()
        except Exception:
            pass

    return {
        "records": all_records,
        "crawl_log": crawl_log,
        "entry_points": {
            "categories_url": CATEGORIES_URL,
            "search_url": SEARCH_URL,
        },
        "queued_complaint_urls": len(queued_complaints),
        "light_mode": light_mode,
        "min_year": min_year,
    }


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Selenium web scraper for Warranty & Return complaints (ComplaintsBoard)"
    )
    parser.add_argument("--target-records", type=int, default=30)
    parser.add_argument("--max-categories", type=int, default=2)
    parser.add_argument("--max-subcategories-per-category", type=int, default=2)
    parser.add_argument("--max-search-pages", type=int, default=2)
    parser.add_argument("--crawl-delay", type=float, default=1.0)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--light-mode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-categories", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-search", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--min-year",
        type=int,
        default=None,
        help="Only keep complaints from this year or newer (default: current year - 4).",
    )
    args = parser.parse_args()
    if args.min_year is None:
        args.min_year = default_min_year()

    if args.light_mode:
        args.target_records = min(args.target_records, 50)
        args.max_categories = min(args.max_categories, 2)
        args.max_subcategories_per_category = min(args.max_subcategories_per_category, 2)
        args.max_search_pages = min(args.max_search_pages, 3)
        args.use_categories = False
        args.crawl_delay = max(args.crawl_delay, 0.8)

    result = scrape_records(
        target_records=args.target_records,
        max_categories=args.max_categories,
        max_subcategories_per_category=args.max_subcategories_per_category,
        max_search_pages=args.max_search_pages,
        crawl_delay=args.crawl_delay,
        headless=args.headless,
        use_categories=args.use_categories,
        use_search=args.use_search,
        light_mode=args.light_mode,
        min_year=args.min_year,
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
            "max_categories": args.max_categories,
            "max_subcategories_per_category": args.max_subcategories_per_category,
            "max_search_pages": args.max_search_pages,
            "entry_points": result["entry_points"],
            "queued_complaint_urls": result["queued_complaint_urls"],
            "scraper_engine": "selenium",
            "light_mode": args.light_mode,
            "min_year": args.min_year,
            "user_agent": USER_AGENT,
            "source": BASE_URL,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    robots_txt = requests.get(ROBOTS_URL, timeout=20, headers={"User-Agent": USER_AGENT}).text
    robots_path.parent.mkdir(parents=True, exist_ok=True)
    robots_path.write_text(robots_txt, encoding="utf-8")

    print(f"Collected {len(result['records'])} warranty/return relevant records")
    print(f"Saved raw data to {raw_path}")
    print(f"Saved crawl log to {log_path}")


if __name__ == "__main__":
    main()
