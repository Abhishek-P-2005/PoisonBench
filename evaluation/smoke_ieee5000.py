"""
Smoke test for the ieee_5000 integration. No Docker, no Chroma server, no Ollama, no model download.

    python -m evaluation.smoke_ieee5000                   # full corpus (5,000 docs), about a minute
    python -m evaluation.smoke_ieee5000 --fast            # 400 docs, 12 queries, a few seconds

What it does: reads the REAL texts in ieee_5000/ (raw NVD pages, the three poison files, the
benchmark), builds the four collections in an in-process ChromaDB, and embeds with a deterministic
hash embedder instead of MiniLM. Then it checks the integration and runs the whole evaluation chain
(evaluation.run_ieee5000) end to end.

What it proves: the code runs, the plumbing is right, defenses never read labels, the provenance
tiers behave as documented, the corpus-reference stages work, reports are written.
What it does NOT prove: any detection rate. Hash embeddings are not MiniLM, so retrieval ranks and
D1 scores differ from the real pipeline. Numbers printed here are never results. Run the real thing
with `python -m evaluation.run_ieee5000` once Docker/Chroma/MiniLM are up.

Prints PASS / FAIL per check and exits 1 if any check fails.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.ieee5000_source import (ATTACK_CODES, BASE, CLEAN_COLLECTION, COLLECTIONS, FEATURES_PATH,   # noqa: E402
                                        HARNESS_ATTACKS, POISON_FILES, build_source, claimed_cve,
                                        load_benchmark, preflight, split_query_ids)

DIM = 384
_TOKEN = re.compile(r"[a-z0-9]+")


class HashEmbedder:
    """Deterministic signed bag-of-words+bigrams hashed to 384 dims, unit length. Plumbing only."""

    def embed(self, text: str) -> list[float]:
        vec = np.zeros(DIM)
        tokens = _TOKEN.findall(text.lower())
        for feature in tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]:
            h = int(hashlib.md5(feature.encode("utf-8")).hexdigest(), 16)
            vec[h % DIM] += 1.0 if (h >> 64) & 1 else -1.0
        norm = np.linalg.norm(vec)
        return (vec / norm if norm else vec).tolist()


def load_documents() -> list[dict]:
    """Clean corpus documents exactly as scripts/ingest_corpus_ieee.py parses them."""
    docs = []
    for page_file in sorted((BASE / "raw").glob("nvd_page_*.json")):
        page = json.loads(page_file.read_text(encoding="utf-8-sig"))
        for item in page.get("vulnerabilities", []):
            cve = item.get("cve", {})
            cve_id = cve.get("id")
            english = next((d["value"] for d in cve.get("descriptions", []) if d.get("lang") == "en"), None)
            if cve_id and english:
                docs.append({"doc_id": cve_id, "text": f"{cve_id}: {english}", "source": "NVD",
                             "timestamp": cve.get("published", "")})
    return docs


def load_poison(query_ids: set[str] | None = None) -> dict[str, list[dict]]:
    out = {}
    for code, path in POISON_FILES.items():
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        if query_ids is not None:
            rows = [r for r in rows if r["query_id"].lower() in query_ids]
        out[code] = rows
    return out


def _upsert(collection, ids, texts, metas, embedder, batch: int = 500):
    for start in range(0, len(ids), batch):
        sl = slice(start, start + batch)
        collection.upsert(ids=ids[sl], documents=texts[sl], metadatas=metas[sl],
                          embeddings=[embedder.embed(t) for t in texts[sl]])


def build_collections(client, embedder, n_docs: int | None = None, n_queries: int | None = None):
    """Create the four ieee_5000 collections in `client`. Returns (queries_used, expected_counts)."""
    from api.pipeline.chunking import chunk_text    # same chunker the ingestion scripts use

    benchmark = load_benchmark()
    if n_queries:
        benchmark = benchmark[:n_queries]
    docs = load_documents()
    if n_docs:
        need = {b["expected_doc_id"] for b in benchmark}
        keep = [d for d in docs if d["doc_id"] in need]
        keep += [d for d in docs if d["doc_id"] not in need][: max(n_docs - len(keep), 0)]
        docs = keep
    query_ids = {b["query_id"].lower() for b in benchmark}

    for name in [CLEAN_COLLECTION, *COLLECTIONS.values()]:
        try:
            client.delete_collection(name)
        except Exception:   # noqa: BLE001
            pass

    ids, texts, metas = [], [], []
    for doc in docs:
        for chunk in chunk_text(doc["text"], doc["doc_id"]):
            ids.append(chunk["chunk_id"]); texts.append(chunk["text"])
            metas.append({"doc_id": doc["doc_id"], "source": doc["source"], "timestamp": doc["timestamp"],
                          "is_poisoned": False, "position": chunk["position"]})
    clean = client.create_collection(CLEAN_COLLECTION, metadata={"hnsw:space": "cosine"})
    _upsert(clean, ids, texts, metas, embedder)
    n_clean = clean.count()

    poison = load_poison(query_ids if (n_docs or n_queries) else None)
    expected = {CLEAN_COLLECTION: (n_clean, 0)}
    for code, name in COLLECTIONS.items():
        col = client.create_collection(name, metadata={"hnsw:space": "cosine"})
        _upsert(col, ids, texts, metas, embedder)
        p_ids, p_texts, p_metas = [], [], []
        for rec in poison[code]:
            for chunk in chunk_text(rec["poison_text"], rec["poison_doc_id"]):
                p_ids.append(chunk["chunk_id"]); p_texts.append(chunk["text"])
                p_metas.append({"doc_id": rec["poison_doc_id"], "source": rec["source"], "timestamp": "",
                                "is_poisoned": True, "position": chunk["position"]})
        _upsert(col, p_ids, p_texts, p_metas, embedder)
        expected[name] = (n_clean + len(p_ids), len(p_ids))
    return query_ids, expected


class Checks:
    def __init__(self):
        self.rows = []

    def check(self, name: str, ok: bool, detail: str = ""):
        self.rows.append((name, bool(ok), detail))
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))
        return ok

    @property
    def failed(self):
        return [r for r in self.rows if not r[1]]


def run(fast: bool = False, keep: bool = False, verbose: bool = False) -> int:
    import chromadb
    warnings.filterwarnings("ignore")
    checks = Checks()
    work = Path(tempfile.mkdtemp(prefix="pb_ieee_smoke_"))
    embedder = HashEmbedder()
    n_docs, n_queries = (400, 12) if fast else (None, None)
    print(f"ieee_5000 smoke test ({'fast: 400 docs, 12 queries' if fast else 'full corpus'}); temp dir {work}")

    print("\n[1] data files")
    needed = [FEATURES_PATH, BASE / "processed" / "benchmark_100.jsonl", BASE / "processed" / "defense_split_5300.csv",
              *POISON_FILES.values()]
    checks.check("ieee_5000 data files present", all(p.is_file() for p in needed),
                 ", ".join(p.name for p in needed if not p.is_file()))
    raw_pages = sorted((BASE / "raw").glob("nvd_page_*.json"))
    checks.check("50 raw NVD pages present", len(raw_pages) == 50, f"{len(raw_pages)} found")
    if not all(p.is_file() for p in needed) or not raw_pages:
        print("\nCannot continue without the data files (run from the repo root, ieee_5000/ must exist).")
        return 1

    print("\n[2] text-feature parity (training features == features computed on live chunk text)")
    import pandas as pd
    from api.pipeline.chunking import chunk_text
    from defenses.defense4_xgboost.feature_engineering import text_stat_features
    frame = pd.read_csv(FEATURES_PATH).set_index("chunk_id")
    docs = {d["doc_id"]: d for d in load_documents()}
    sample = []
    for doc_id in list(docs)[:5]:
        chunk = chunk_text(docs[doc_id]["text"], doc_id)[0]
        sample.append((chunk["chunk_id"], chunk["text"]))
    for code, rows in load_poison().items():
        rec = rows[0]
        chunk = chunk_text(rec["poison_text"], rec["poison_doc_id"])[0]
        sample.append((chunk["chunk_id"], chunk["text"]))
    cols = ["text_length_chars", "text_length_words", "avg_word_length", "char_entropy", "digit_ratio", "punctuation_ratio"]
    worst = 0.0
    for chunk_id, text in sample:
        live = text_stat_features(text)
        worst = max(worst, max(abs(float(live[c]) - float(frame.loc[chunk_id, c])) for c in cols))
    checks.check("text features recomputed from chunk text match defense_features_5300.csv", worst < 1e-6, f"max diff {worst:.2e}")

    print("\n[3] build collections in an in-process ChromaDB (hash embedder)")
    client = chromadb.EphemeralClient()
    query_ids, expected = build_collections(client, embedder, n_docs, n_queries)
    rows = preflight(client)
    sizes_ok = all(r["chunks"] == expected[r["collection"]][0] and r["poisoned"] == expected[r["collection"]][1] for r in rows)
    checks.check("4 collections exist with the expected sizes", all(r["exists"] for r in rows) and sizes_ok,
                 ", ".join(f"{r['collection'].split('_')[-1]}={r['chunks']}" for r in rows))
    if not fast:
        checks.check("preflight() agrees with the build scripts (5000 / 5100 chunks, 100 poison)", all(r["ok"] for r in rows))

    print("\n[4] D4: six-feature models train on the repo's split, checksum, load, score a live chunk")
    from evaluation import ieee5000_d4
    d4_dir = work / "d4"
    report = ieee5000_d4.train(d4_dir, verbose=False)
    checks.check("both models trained and F1 is not degenerate", all(report["metrics"][k]["f1"] > 0.5 for k in ieee5000_d4.KINDS),
                 ", ".join(f"{k} F1={report['metrics'][k]['f1']:.2f}" for k in ieee5000_d4.KINDS))
    model = ieee5000_d4.load_ieee_text_model("random_forest", d4_dir)
    probs = model.predict_proba([{"text": sample[0][1]}, {"text": sample[-1][1]}])
    checks.check("model scores a raw chunk (poison scores above clean)", len(probs) == 2 and probs[1] > probs[0],
                 f"clean={probs[0]:.2f} poison={probs[1]:.2f}")
    tampered = d4_dir / ieee5000_d4.FILES["random_forest"]
    tampered.write_bytes(tampered.read_bytes() + b"x")
    try:
        ieee5000_d4.load_ieee_text_model("random_forest", d4_dir)
        refused = False
    except RuntimeError:
        refused = True
    checks.check("loader refuses a model file whose checksum changed", refused)
    ieee5000_d4.train(d4_dir, overwrite=True, verbose=False)   # restore a valid file for the full run below

    print("\n[5] source, provenance tiers, label-blindness")
    from evaluation.config import make_config
    from evaluation.pipeline import DefensePipeline, strip_labels
    from evaluation.reference import CorpusReference, set_reference
    from evaluation.stages import build_stages
    source = build_source(client, embedder.embed, tier="as_stored", query_ids=query_ids)
    embed_many = lambda texts: [embedder.embed(t) for t in texts]   # noqa: E731
    reference = CorpusReference.from_collection(client.get_collection(CLEAN_COLLECTION), embed=embed_many)
    set_reference(reference)
    checks.check("corpus reference built from the clean collection", reference.n_chunks > 0 and len(reference.registry) > 0,
                 f"{reference.n_chunks} chunks via {reference.method} (re-embedded = this Chroma refused to return embeddings; still fine)")
    sq = source.iter_queries("naive", 0.0, 0)
    checks.check("every query returns 10 candidates", all(len(q.candidates) == 10 for q in sq))
    n_poison_raw = sum(any(c["metadata"]["is_poisoned"] for c in q.candidates[:5]) for q in sq)
    checks.check("poison is retrievable at all (plumbing; hash embedder, not a finding)", n_poison_raw > 0, f"{n_poison_raw}/{len(sq)} queries")
    clean_q = source.iter_queries("none", 0.0, 0)
    checks.check("clean collection returns no poison", not any(c["metadata"]["is_poisoned"] for q in clean_q for c in q.candidates))

    def blocked_split(tier, attack, stage_name):
        src = source.with_tier(tier)
        pipe = DefensePipeline(build_stages(make_config((stage_name,))), 5)
        bp = bc = fp = fc = 0
        for q in src.iter_queries(attack, 0.0, 0):
            truth = {c["chunk_id"]: c["metadata"]["is_poisoned"] for c in q.candidates}
            res = pipe.run(q.candidates)
            bp += sum(truth[i] for i in res.blocked_ids); bc += sum(not truth[i] for i in res.blocked_ids)
            flagged = res.flagged_ids | res.blocked_ids
            fp += sum(truth[i] for i in flagged); fc += sum(not truth[i] for i in flagged)
        n_p = sum(sum(c["metadata"]["is_poisoned"] for c in q.candidates) for q in src.iter_queries(attack, 0.0, 0))
        return bp, bc, fp, fc, n_p

    bp, bc, fp, fc, n_p = blocked_split("spoofed", "naive", "d3")
    checks.check("tier 'spoofed': D3 blocks every retrieved poison chunk and no clean chunk", n_p > 0 and bp == n_p and bc == 0, f"blocked {bp}/{n_p}, clean blocked {bc}")
    bp, bc, fp, fc, n_p = blocked_split("forged", "naive", "d3")
    checks.check("tier 'forged': D3 cannot see poison (blocks 0, flags 0, clean unaffected)", bp == 0 and fp == 0 and bc == 0, f"blocked {bp}, flagged {fp}")
    bp, bc, fp, fc, n_p = blocked_split("as_stored", "naive", "d3")
    checks.check("tier 'as_stored': D3 flags poison only via the empty timestamp (documented artifact)", n_p > 0 and fp == n_p and fc == 0, f"flagged {fp}/{n_p}, clean flagged {fc}")

    bp, bc, fp, fc, n_p = blocked_split("forged", "optimized", "ref")
    checks.check("registry ('ref') blocks every poison chunk that reuses a known CVE id, and no clean chunk", n_p > 0 and bp == n_p and bc == 0, f"blocked {bp}/{n_p}, clean blocked {bc}")
    from evaluation.reference import RegistryStage
    invented = [{"chunk_id": "x::chunk0", "text": "CVE-2099-0001: invented advisory text", "metadata": {}, "similarity": 0.9, "embedding": [0.0] * DIM}]
    verdict = RegistryStage().run(invented)
    checks.check("registry only FLAGS an invented CVE id (documented evasion)", not verdict.blocked and "x::chunk0" in verdict.flagged_ids)

    d4_for_leak = ieee5000_d4.load_ieee_text_model("random_forest", d4_dir)
    config = make_config(("d1", "d2", "d3", "d4", "d1c", "ref"))
    leak_free = True
    for tier in ("as_stored", "forged"):
        src = source.with_tier(tier)
        for q in src.iter_queries("injection", 0.0, 0)[:5]:
            pipe = DefensePipeline(build_stages(config, d4_for_leak), 5)
            a = pipe.run(q.candidates)
            flipped = [{**c, "metadata": {**c["metadata"], "is_poisoned": not c["metadata"]["is_poisoned"], "attack_type": "zzz"}} for c in q.candidates]
            b = pipe.run(flipped)
            same = (a.blocked_ids == b.blocked_ids and a.flagged_ids == b.flagged_ids and
                    [c["chunk_id"] for c in a.final_context] == [c["chunk_id"] for c in b.final_context])
            leak_free &= same
    checks.check("flipping every is_poisoned label changes no defense decision (label-blind)", leak_free)
    stripped = strip_labels(sq[0].candidates[0])
    checks.check("DefensePipeline strips is_poisoned / attack_type before defenses", "is_poisoned" not in stripped["metadata"])

    print("\n[6] defended API path accepts the new stage names and model kinds")
    from api.pipeline import defended
    set_reference(None)
    cfg, _, skipped = defended.plan_pipeline(["d1c", "ref"], False, "stored", "none")
    checks.check("d1c/ref are skipped (not crashed) when no reference is loaded", {s["stage"] for s in skipped} == {"d1c", "ref"} and not cfg.stages)
    set_reference(reference)
    cfg, _, skipped = defended.plan_pipeline(["d1c", "ref"], False, "stored", "none")
    checks.check("d1c/ref run once a reference is loaded", cfg.stages == ("d1c", "ref") and not skipped)
    model_obj, reason = defended.load_d4("ieee_nonexistent")
    checks.check("unknown D4 kind is reported as unavailable, not raised", model_obj is None and reason)

    print("\n[7] full chain: python -m evaluation.run_ieee5000 (all tiers, held-out queries)")
    from evaluation import run_ieee5000
    out = work / "out"
    argv = ["--d4-dir", str(d4_dir), "--out", str(out), "--max-eval-queries", "6" if fast else "20", "--repeats", "1"]
    if verbose:
        argv.append("--verbose")
    rc = run_ieee5000.main(argv, client=client, embed_query=embedder.embed, fake_embedder=True)
    checks.check("run_ieee5000 completed with no failed sections", rc == 0, f"exit code {rc}")
    summary_path, report_path = out / "summary.csv", out / "report.md"
    checks.check("summary.csv and report.md written", summary_path.is_file() and report_path.is_file())
    if summary_path.is_file():
        summary = pd.read_csv(summary_path)
        checks.check("all three tiers are in the summary", set(summary["tier"]) == {"as_stored", "spoofed", "forged"})
        none = summary[(summary["config"] == "none") & (summary["attack"] != "none")]
        checks.check("'none' config leaves attack success unchanged (defended == raw)", bool((none["defended_asr"] == none["raw_asr"]).all()))
        ref = summary[(summary["config"] == "ref") & (summary["attack"] != "none")]
        checks.check("registry baseline drives defended ASR to 0 on all three attacks", bool((ref["defended_asr"] == 0).all()) and len(ref) >= 9)
        forged_d3 = summary[(summary["tier"] == "forged") & (summary["config"] == "d3") & (summary["attack"] != "none")]
        checks.check("on the forged tier D3 alone does not reduce attack success", bool((forged_d3["defended_asr"] == forged_d3["raw_asr"]).all()))
        clean_rows = summary[(summary["attack"] == "none") & (summary["config"] == "d1>d2>d3>d4")]
        checks.check("clean baseline rows exist (false-positive accounting)", len(clean_rows) > 0)
        print("\n  Sample of what the report looks like (HASH EMBEDDINGS, NOT RESULTS):")
        view = summary[(summary["tier"] == "forged") & (summary["config"].isin(["none", "d3", "ref", "d4", "d1>d2>d3>d4"])) & (summary["attack"] == "naive")]
        print("  " + view[["config", "attack", "raw_asr", "defended_asr", "legit_kept_rate", "empty_context_rate"]].round(2).to_string(index=False).replace("\n", "\n  "))
    if not keep:
        shutil.rmtree(work, ignore_errors=True)
    else:
        print(f"\n(kept outputs in {work})")

    print("\n" + "=" * 70)
    if checks.failed:
        print(f"{len(checks.failed)} of {len(checks.rows)} checks FAILED:")
        for name, _, detail in checks.failed:
            print(f"  - {name}  {detail}")
        return 1
    print(f"all {len(checks.rows)} checks passed")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fast", action="store_true", help="400 documents and 12 queries instead of the full corpus")
    ap.add_argument("--keep", action="store_true", help="keep the temp output folder")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)
    return run(fast=args.fast, keep=args.keep, verbose=args.verbose)


if __name__ == "__main__":
    sys.exit(main())
