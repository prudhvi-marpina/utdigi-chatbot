# scrapy_academics.py — Full UTD Crawler (Optimized)

import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy_playwright.page import PageMethod
from pymongo import MongoClient
from urllib.parse import urlparse
from datetime import datetime
import logging

# ✅ MongoDB connection
client = MongoClient("mongodb://localhost:27017/")
collection = client["utdallas_db_mini"]["scraped_data"]

def clean_text(texts):
    return list(set(t.strip() for t in texts if t.strip()))

class UTDSpider(scrapy.Spider):
    name = "utd_full_site"
    allowed_domains = ["utdallas.edu"]
    start_urls = ["https://www.utdallas.edu/"]
    visited_urls = set()

    custom_settings = {
        "DOWNLOAD_DELAY": 1,
        "PLAYWRIGHT_BROWSER_TYPE": "chromium",
        "PLAYWRIGHT_DEFAULT_NAVIGATION_TIMEOUT": 60000,
        "PLAYWRIGHT_LAUNCH_OPTIONS": {"headless": True},
        "CLOSESPIDER_PAGECOUNT": 2000,  # ⬅️ Increase this to crawl more
        "TWISTED_REACTOR": "twisted.internet.asyncioreactor.AsyncioSelectorReactor",
        "LOG_LEVEL": "INFO",
    }

    def start_requests(self):
        for url in self.start_urls:
            yield scrapy.Request(
                url,
                callback=self.parse,
                meta={
                    "playwright": True,
                    "playwright_include_page": True,
                    "playwright_page_methods": [PageMethod("wait_for_load_state", "networkidle")],
                },
            )

    def parse(self, response):
        try:
            url = response.url.split("#")[0].rstrip("/")
            if url in self.visited_urls:
                return
            self.visited_urls.add(url)

            # ✅ Extract content
            data = {
                "url": url,
                "title": response.css("title::text").get(),
                "headings": clean_text(response.css("h1::text, h2::text, h3::text").getall()),
                "paragraphs": clean_text(response.css("p::text").getall()),
                "divs": clean_text(response.css("div::text").getall()),
                "spans": clean_text(response.css("span::text").getall()),
                "scraped_at": datetime.utcnow().isoformat(),
                "metadata": {
                    "content_length": len(response.text),
                    "status": response.status
                }
            }
            collection.insert_one(data)
            self.logger.info(f"✅ Crawled: {url} ({len(response.text)} chars)")

            # ✅ Follow internal UTD links
            for href in response.css("a::attr(href)").getall():
                full_url = response.urljoin(href).split("#")[0].rstrip("/")
                parsed = urlparse(full_url)

                if (
                    "utdallas.edu" in parsed.netloc
                    and full_url not in self.visited_urls
                    and not any(social in full_url for social in ["facebook", "twitter", "linkedin", "instagram", "youtube", "mailto:", "tel:"])
                ):
                    yield scrapy.Request(
                        full_url,
                        callback=self.parse,
                        meta={
                            "playwright": True,
                            "playwright_include_page": True,
                            "playwright_page_methods": [PageMethod("wait_for_load_state", "networkidle")],
                        },
                    )

        except Exception as e:
            self.logger.error(f"❌ Failed to process {response.url}: {e}")

if __name__ == "__main__":
    logging.getLogger("scrapy").setLevel(logging.INFO)

    process = CrawlerProcess()
    process.crawl(UTDSpider)
    process.start()

