"""
Downloads CVE records from the NVD REST API (v2.0) and saves them as raw
JSON pages in data/raw/. Run this ONCE (or whenever you want a fresh/larger
corpus) before scripts/ingest_corpus.py.

NVD's public rate limit without an API key is 5 requests / 30 seconds.
With a free API key (https://nvd.nist.gov/developers/request-an-api-key)
it's 50 requests / 30 seconds. Set NVD_API_KEY in .env to use one -- it's
optional, this script works without it, just slower.

Usage:
    python scripts/download_nvd_corpus.py --total 1000
"""
import argparse
import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

NVD_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def fetch_page(start_index: int, results_per_page: int, api_key: str | None) -> dict:
    headers = {"apiKey": api_key} if api_key else {}
    params = {"startIndex": start_index, "resultsPerPage": results_per_page}
    resp = requests.get(NVD_URL, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total", type=int, default=500, help="Total CVEs to download")
    parser.add_argument("--page-size", type=int, default=100, help="Results per page (NVD max: 2000)")
    args = parser.parse_args()

    api_key = os.getenv("NVD_API_KEY") or None
    sleep_s = 6 if not api_key else 0.7  # stay comfortably under the rate limit

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    fetched = 0
    start_index = 0
    page_num = 0

    print(f"Downloading {args.total} CVEs from NVD (api_key={'set' if api_key else 'not set'})...")

    while fetched < args.total:
        page_size = min(args.page_size, args.total - fetched)
        data = fetch_page(start_index, page_size, api_key)

        out_path = RAW_DIR / f"nvd_page_{page_num:04d}.json"
        with open(out_path, "w") as f:
            json.dump(data, f)

        n_this_page = len(data.get("vulnerabilities", []))
        fetched += n_this_page
        start_index += n_this_page
        page_num += 1

        print(f"  page {page_num}: +{n_this_page} CVEs (total so far: {fetched}) -> {out_path.name}")

        if n_this_page == 0:
            print("  NVD returned no more results, stopping early.")
            break

        time.sleep(sleep_s)

    print(f"Done. {fetched} CVEs saved across {page_num} files in {RAW_DIR}")


if __name__ == "__main__":
    main()
