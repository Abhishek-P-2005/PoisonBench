"""
Reads the raw NVD JSON pages from data/raw/, transforms each CVE record into
a document, and bulk-loads them through the SAME chunking/embedding/storage
functions the API uses (imported directly, not called over HTTP).

Why direct import instead of hitting POST /ingest in a loop:
bulk-loading thousands of documents one HTTP request at a time is slow and
adds nothing -- the API endpoint exists so each STAGE can be tested in
isolation (see api/routers/ingest.py docstring), not so bulk loads have to
go through the network. This script reuses api.pipeline.chunking /
embedding / vectorstore directly, so there is exactly one implementation of
chunking/embedding/storage logic -- the API and this script can never drift
out of sync.

Usage (run from the PoisonBench/ repo root, with .env configured so
CHROMA_HOST/PORT point at your running chromadb container -- see .env.example):

    python scripts/ingest_corpus.py
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

from api.pipeline.chunking import chunk_text          # noqa: E402
from api.pipeline.vectorstore import upsert_chunks     # noqa: E402

RAW_DIR = REPO_ROOT / "data" / "raw"
BATCH_SIZE = 64  # chunks per upsert call, keeps embedding calls batched


def extract_documents_from_page(page: dict) -> list[dict]:
    """Turns one NVD API page into a list of {doc_id, text, source, timestamp}."""
    docs = []
    for item in page.get("vulnerabilities", []):
        cve = item.get("cve", {})
        cve_id = cve.get("id")
        if not cve_id:
            continue

        descriptions = cve.get("descriptions", [])
        eng_desc = next((d["value"] for d in descriptions if d.get("lang") == "en"), None)
        if not eng_desc:
            continue

        published = cve.get("published", "")

        docs.append({
            "doc_id": cve_id,
            "text": f"{cve_id}: {eng_desc}",
            "source": "NVD",
            "timestamp": published,
            "is_poisoned": False,   # this is the CLEAN corpus loader; the attack
                                     # module sets this True for injected docs
        })
    return docs


def main():
    raw_files = sorted(RAW_DIR.glob("nvd_page_*.json"))
    if not raw_files:
        print(f"No raw NVD files found in {RAW_DIR}. Run download_nvd_corpus.py first.")
        return

    all_docs = []
    for f in raw_files:
        with open(f) as fh:
            page = json.load(fh)
        all_docs.extend(extract_documents_from_page(page))

    print(f"Parsed {len(all_docs)} documents from {len(raw_files)} raw file(s).")

    total_chunks_buffer = []
    total_stored = 0

    for doc in all_docs:
        doc_chunks = chunk_text(text=doc["text"], doc_id=doc["doc_id"])
        for c in doc_chunks:
            c["doc_id"] = doc["doc_id"]
            c["source"] = doc["source"]
            c["timestamp"] = doc["timestamp"]
            c["is_poisoned"] = doc["is_poisoned"]
        total_chunks_buffer.extend(doc_chunks)

        if len(total_chunks_buffer) >= BATCH_SIZE:
            total_stored += upsert_chunks(total_chunks_buffer)
            print(f"  stored {total_stored} chunks so far...")
            total_chunks_buffer = []

    if total_chunks_buffer:
        total_stored += upsert_chunks(total_chunks_buffer)

    print(f"Done. {len(all_docs)} documents -> {total_stored} chunks stored in ChromaDB.")


if __name__ == "__main__":
    main()
