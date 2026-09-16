"""
Google Indexing API - Submit URLs for crawling/indexing.

Submits all pages from mimran-khan.github.io to Google's priority crawl queue.
Reads episodes.json dynamically so new episodes are auto-included.

Prerequisites:
  1. Service account created and JSON key downloaded
  2. Indexing API enabled in Google Cloud Console
  3. Service account added as Owner in Google Search Console

Usage:
  source .venv/bin/activate
  python submit_indexing.py
"""

import json
import time
from pathlib import Path

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

SERVICE_ACCOUNT_FILE = Path(__file__).parent / "indexingproject-501323-553b360d71bb.json"
SCOPES = ["https://www.googleapis.com/auth/indexing"]
BASE = "https://mimran-khan.github.io"

EPISODES_JSON = Path.home() / "Documents" / "mimran-khan.github.io" / "src" / "data" / "episodes.json"

STATIC_URLS = [
    f"{BASE}/",
    f"{BASE}/blog/loop-engineering",
    f"{BASE}/blog/agents-all-the-way-down",
    f"{BASE}/blog/deltaforge",
    f"{BASE}/series/demystifying-ai",
]

def build_url_list():
    urls = list(STATIC_URLS)
    if EPISODES_JSON.exists():
        episodes = json.loads(EPISODES_JSON.read_text())
        for ep in episodes:
            if ep.get("published"):
                urls.append(f"{BASE}/series/demystifying-ai/{ep['slug']}")
    return urls

SITE_URLS = build_url_list()


def get_service():
    credentials = service_account.Credentials.from_service_account_file(
        str(SERVICE_ACCOUNT_FILE), scopes=SCOPES
    )
    return build("indexing", "v3", credentials=credentials)


def submit_url(service, url: str, action: str = "URL_UPDATED") -> dict:
    """Submit a single URL for indexing. Retries on 5xx errors."""
    body = {"url": url, "type": action}
    for attempt in range(3):
        try:
            response = service.urlNotifications().publish(body=body).execute()
            return response
        except HttpError as e:
            if e.resp.status >= 500:
                wait = 2**attempt
                print(f"  Server error, retrying in {wait}s...")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError(f"Failed after retries: {url}")


def main():
    print("=" * 60)
    print(" Google Indexing API - URL Submission")
    print("=" * 60)
    print(f"\nService Account: {SERVICE_ACCOUNT_FILE.name}")
    print(f"URLs to submit: {len(SITE_URLS)}\n")

    service = get_service()

    results = {"success": [], "failed": []}

    for i, url in enumerate(SITE_URLS, 1):
        print(f"[{i}/{len(SITE_URLS)}] Submitting: {url}")
        try:
            response = submit_url(service, url)
            notify_time = response.get("urlNotificationMetadata", {}).get(
                "latestUpdate", {}
            ).get("notifyTime", "unknown")
            print(f"       ✓ Queued for crawl (notifyTime: {notify_time})")
            results["success"].append(url)
        except HttpError as e:
            error_detail = json.loads(e.content.decode()).get("error", {})
            print(f"       ✗ Error {e.resp.status}: {error_detail.get('message', str(e))}")
            results["failed"].append({"url": url, "error": str(e.resp.status)})
        except Exception as e:
            print(f"       ✗ Error: {e}")
            results["failed"].append({"url": url, "error": str(e)})

        time.sleep(1)

    print("\n" + "=" * 60)
    print(" RESULTS")
    print("=" * 60)
    print(f" Submitted successfully: {len(results['success'])}")
    print(f" Failed:                 {len(results['failed'])}")
    if results["failed"]:
        print("\n Failed URLs:")
        for item in results["failed"]:
            print(f"   - {item['url']} ({item['error']})")
    print("=" * 60)


if __name__ == "__main__":
    main()
