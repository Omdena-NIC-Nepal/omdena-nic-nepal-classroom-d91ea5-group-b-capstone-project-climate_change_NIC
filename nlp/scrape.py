import os
import time
import logging
import re

from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from newspaper import Article
from bs4 import BeautifulSoup
import feedparser

# Setup logging
logging.basicConfig(level=logging.INFO, format="INFO: %(message)s")

# Directory to save articles
SAVE_DIR = "articles"
os.makedirs(SAVE_DIR, exist_ok=True)

# RSS feed of climate-related news
RSS_FEED = "https://news.google.com/rss/search?q=climate+change+Nepal&hl=en-NE&gl=NP&ceid=NP:en"

# Setup headless Chrome WebDriver
def get_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

# Clean extracted HTML content using BeautifulSoup
def clean_article_content(raw_html: str) -> str:
    soup = BeautifulSoup(raw_html, 'html.parser')

    # Remove unnecessary sections
    for tag in soup.find_all(['script', 'style', 'nav', 'footer', 'aside']):
        tag.decompose()

    for a in soup.find_all('a', string=re.compile(r'(Facebook|LinkedIn|X|Twitter|Share)', re.IGNORECASE)):
        a.decompose()

    for div in soup.find_all('div', class_=re.compile(r'download|attachments|related', re.IGNORECASE)):
        div.decompose()

    # Collect meaningful blocks
    blocks = soup.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'li', 'blockquote'])

    # Join text with paragraph spacing
    content = '\n\n'.join(block.get_text(strip=True) for block in blocks if block.get_text(strip=True))
    return re.sub(r'\s{3,}', '\n\n', content).strip()

# Extract article content using Selenium and fallback to newspaper3k
def extract_article_content(driver, url):
    try:
        driver.get(url)
        time.sleep(3)

        selectors = [
            '//article',
            '//div[contains(@class, "article-content")]',
            '//div[@id="content"]',
            '//section[contains(@class, "content")]',
            '//div[@class="post-content"]'
        ]

        raw_html = ""
        for selector in selectors:
            try:
                elem = driver.find_element(By.XPATH, selector)
                raw_html = elem.get_attribute('innerHTML')
                logging.info(f"✅ Extracted using XPath: {selector}")
                break
            except:
                continue

        # If no specific selector worked, fallback to full page source
        if not raw_html:
            logging.info("⚠️  XPath selectors failed. Using full page source.")
            raw_html = driver.page_source

        # Clean content
        text = clean_article_content(raw_html)
        if text:
            return text

        # Final fallback: newspaper3k
        logging.info("⚠️  Falling back to newspaper3k for: " + url)
        article = Article(url)
        article.download()
        article.parse()
        return article.text.strip() if article.text else None

    except Exception as e:
        logging.warning(f"❌ Exception extracting from {url}: {e}")
        return None

# Sanitize filename
def clean_filename(title):
    return ''.join(c if c.isalnum() or c in (' ', '_') else '_' for c in title).replace(' ', '_')[:80] + ".txt"

def main():
    logging.info("Fetching article metadata...")
    feed = feedparser.parse(RSS_FEED)
    entries = feed.entries[:50]  # Limit to top 50
    logging.info(f"Found {len(entries)} articles. Starting content extraction...")

    failed_articles = []
    driver = get_driver()

    for entry in tqdm(entries):
        title = entry.title
        link = entry.link
        published = entry.published
        source = entry.source.title if 'source' in entry else 'Unknown'

        logging.info(f"📄 Processing: {title}")

        content = extract_article_content(driver, link)
        if not content:
            logging.info(f"🚫 Could not extract content: {title}")
            failed_articles.append(link)
            continue

        filename = clean_filename(title)
        filepath = os.path.join(SAVE_DIR, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"Title: {title}\nPublished: {published}\nSource: {source}\n\n")
            f.write(content)

        logging.info(f"✅ Saved: {filename}")

    driver.quit()

    if failed_articles:
        logging.info("\n🧾 Failed Articles:")
        for bad_url in failed_articles:
            logging.info(" - " + bad_url)
    else:
        logging.info("✅ All articles processed successfully.")

if __name__ == "__main__":
    main()
