"""
evaluation/ieee5000_source.py
-----------------------------
Runs the harness on the 5,000-record benchmark (folder ieee_5000/) instead of the
original 316-chunk one. Nothing under ieee_5000/ is modified; this module only reads it.

  queries           ieee_5000/processed/benchmark_100.jsonl           (100 queries)
  clean collection  poisonbench_ieee_5000_clean                       (5,000 chunks)
  attacked          poisonbench_ieee_5000_{sm|obea|rpi}               (clean + 100 poison chunks)
  held-out split    ieee_5000/processed/defense_split_5300.csv        (20 held-out target queries)

Differences from evaluation/real_source.py (the 316-chunk source), and why they matter:

1. ONE poisoning ratio. Each attacked collection holds 100 poison chunks among 5,100,
   i.e. 1.96%. The 1/5/10/20% sweep of the old benchmark does not exist here, so the
   ratio is fixed (IEEE_RATIO) and is logged as such.

2. Provenance tiers. The stored chunks carry only doc_id/source/timestamp/is_poisoned/position.
   Clean chunks say source="NVD" with a real date; poison chunks say source="controlled_*"
   with an EMPTY timestamp. Run D2/D3 on those labels and they separate poison from clean
   because of how the experiment named its own attack data, not because the defenses are
   any good. The tiers make the attacker model explicit instead:

     thin       metadata untouched; D2, D3 and the router are skipped
     as_stored  derived from the stored labels (exploratory, leaky; see above)
     spoofed    poison claims to be official NVD but its source_id is not on the allow-list
                (what D3's identity check is designed to catch)
     forged     poison carries a fully valid, allow-listed identity, plausible dates, a correct
                content hash and the CVE id of its target: a compromised trusted source.
                D2 and D3 cannot separate it from clean data; only content defenses can.

   The attacker model is applied here, by the experiment, the same way the attack module
   writes the poison. The defenses never see is_poisoned (DefensePipeline strips it).

3. D4. The repo's IEEE models use 12 transductive features and cannot score a live chunk, so
   this source is meant to be used with the six-feature text models from ieee5000_d4.py.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from defenses.defense1_outlier_detection.filter import from_chroma_query_result
from evaluation.config import CLEAN_ATTACK, N_RETRIEVE
from evaluation.metadata_bridge import chunk_hash, enrich_metadata
from evaluation.scenarios import ChromaScenarioSource, ScenarioQuery, _embeddings_as_lists

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "ieee_5000"
BENCHMARK_PATH = BASE / "processed" / "benchmark_100.jsonl"
SPLIT_PATH = BASE / "processed" / "defense_split_5300.csv"
FEATURES_PATH = BASE / "processed" / "defense_features_5300.csv"
POISON_FILES = {
    "sm": BASE / "poisoned" / "semantic_mimicry_100.jsonl",
    "obea": BASE / "poisoned" / "optimized_embedding_100.jsonl",
    "rpi": BASE / "poisoned" / "retrieved_prompt_injection_100.jsonl",
}

CLEAN_COLLECTION = "poisonbench_ieee_5000_clean"
COLLECTIONS = {
    "sm": "poisonbench_ieee_5000_sm",
    "obea": "poisonbench_ieee_5000_obea",
    "rpi": "poisonbench_ieee_5000_rpi",
}
ATTACK_CODES = {"naive": "sm", "optimized": "obea", "injection": "rpi",
                "sm": "sm", "obea": "obea", "rpi": "rpi"}
HARNESS_ATTACKS = ("naive", "optimized", "injection")   # names the analysis modules use

N_CLEAN_CHUNKS = 5000
N_POISON_PER_ATTACK = 100
IEEE_RATIO = N_POISON_PER_ATTACK / (N_CLEAN_CHUNKS + N_POISON_PER_ATTACK)   # 0.0196
RPI_MARKER = "POISONBENCH_RPI_SUCCESS"

TIERS = ("thin", "as_stored", "spoofed", "forged")
ANALYSIS_TIERS = ("as_stored", "spoofed", "forged")    # tiers D2/D3 can be run on
ATTACKER_DATE = "2019-01-01"                           # plausible date the attacker stamps on poison
_CVE_RE = re.compile(r"(CVE-\d{4}-\d{4,7})")   # first CVE id anywhere: OBEA poison puts keywords before it


# ── benchmark and split ───────────────────────────────────────────────────────

def load_benchmark(path: Path = BENCHMARK_PATH) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            row = json.loads(line)
            rows.append({"query_id": row["query_id"], "query_text": row["query"],
                         "expected_doc_id": row.get("expected_doc_id")})
    return rows


def heldout_query_ids(split_path: Path = SPLIT_PATH) -> set[str]:
    """Targets (queries) whose documents D4 never trained on: 'target:q013' -> 'q013'."""
    split = pd.read_csv(split_path)
    held = split[(split["split"] == "test") & split["group_id"].str.startswith("target:")]
    return {group.split(":", 1)[1].lower() for group in held["group_id"].unique()}


def split_query_ids(benchmark: list[dict] | None = None, split_path: Path = SPLIT_PATH):
    """(calibration_ids, heldout_ids). Calibrate thresholds on the first, report on the second."""
    benchmark = benchmark or load_benchmark()
    held = heldout_query_ids(split_path)
    everything = {row["query_id"].lower() for row in benchmark}
    return everything - held, held


# ── provenance tiers ──────────────────────────────────────────────────────────

def claimed_cve(text: str) -> str | None:
    """The first CVE id in the chunk's own text ('CVE-2002-0241: ...'), or None."""
    match = _CVE_RE.search(text or "")
    return match.group(1) if match else None


def apply_provenance_tier(chunk_id: str, text: str, metadata: dict, tier: str) -> dict:
    """
    Metadata a defense will see for one chunk under a provenance tier. Reading
    is_poisoned here is the EXPERIMENT acting as the attacker (choosing what the
    poison claims about itself); the label is removed before any defense runs.
    """
    if tier not in TIERS:
        raise ValueError(f"tier must be one of {TIERS}, got '{tier}'")
    meta = dict(metadata or {})
    if tier == "thin":
        return meta
    out = enrich_metadata(chunk_id, text, meta)
    if tier == "as_stored" or not meta.get("is_poisoned"):
        return out
    out.update(
        doc_id=claimed_cve(text) or out["doc_id"],
        published_date=ATTACKER_DATE,
        modified_date=ATTACKER_DATE,
        chunk_hash=chunk_hash(text),
        source_type="nvd_official",
    )
    if tier == "spoofed":
        out.update(source_id="attacker_mirror", verified=False)
    else:   # forged
        out.update(source_id="nvd_official", verified=True)
    return out


# ── source ────────────────────────────────────────────────────────────────────

class Ieee5000Source(ChromaScenarioSource):
    """Retrieval from the ieee_5000 collections, one query per benchmark question."""

    def __init__(self, client, trigger_queries, embed_query, n_retrieve: int = N_RETRIEVE,
                 tier: str = "as_stored", collections: dict | None = None,
                 clean_collection: str | None = None, _raw_cache: dict | None = None):
        if tier not in TIERS:
            raise ValueError(f"tier must be one of {TIERS}, got '{tier}'")
        super().__init__(None, trigger_queries, embed_query=embed_query, n_retrieve=n_retrieve,
                         provenance="thin" if tier == "thin" else "stored")
        self.client = client
        self.tier = tier
        self.collections = dict(collections or COLLECTIONS)
        self.clean_collection = clean_collection or CLEAN_COLLECTION
        self.resolved_provenance = self.provenance
        self._raw_cache = _raw_cache if _raw_cache is not None else {}
        self._expected = {t["query_id"]: t.get("expected_doc_id") for t in trigger_queries}
        self.embeddings_reembedded = False

    @property
    def name(self) -> str:
        return f"ieee5000[{self.tier}]"

    def expected_doc(self, query_id: str):
        return self._expected.get(query_id)

    def with_queries(self, query_ids: set[str]) -> "Ieee5000Source":
        """Same source restricted to some queries; shares the retrieval cache."""
        wanted = {q.lower() for q in query_ids}
        triggers = [t for t in self.trigger_queries if t["query_id"].lower() in wanted]
        return Ieee5000Source(self.client, triggers, self.embed_query, self.n_retrieve, self.tier,
                              self.collections, self.clean_collection, self._raw_cache)

    def with_tier(self, tier: str) -> "Ieee5000Source":
        return Ieee5000Source(self.client, self.trigger_queries, self.embed_query, self.n_retrieve,
                              tier, self.collections, self.clean_collection, self._raw_cache)

    def collection_name(self, attack_type: str) -> str:
        if attack_type == CLEAN_ATTACK:
            return self.clean_collection
        code = ATTACK_CODES.get(attack_type)
        if code is None:
            raise ValueError(f"unknown attack '{attack_type}'; expected one of {sorted(ATTACK_CODES)}")
        return self.collections[code]

    def _raw_candidates(self, collection_name: str, trigger: dict) -> list[dict]:
        key = (collection_name, trigger["query_id"], self.n_retrieve)
        if key not in self._raw_cache:
            try:
                collection = self.client.get_collection(name=collection_name)
            except Exception as exc:
                raise RuntimeError(f"Chroma collection '{collection_name}' not found ({exc}). Build it with "
                                   "the scripts/ingest_*_ieee.py scripts first.") from exc
            query_vector = [list(self.embed_query(trigger["query_text"]))]
            try:
                raw = collection.query(query_embeddings=query_vector, n_results=self.n_retrieve,
                                       include=["embeddings", "documents", "metadatas", "distances"])
                raw = _embeddings_as_lists(raw)
            except Exception:   # noqa: BLE001  some Chroma builds fail to return stored embeddings
                raw = collection.query(query_embeddings=query_vector, n_results=self.n_retrieve,
                                       include=["documents", "metadatas", "distances"])
                raw["embeddings"] = [[list(self.embed_query(text)) for text in raw["documents"][0]]]
                self.embeddings_reembedded = True
            self._raw_cache[key] = from_chroma_query_result(raw)
        return self._raw_cache[key]

    def iter_queries(self, attack_type: str, ratio: float, seed: int) -> list[ScenarioQuery]:
        name = self.collection_name(attack_type)
        attacked = attack_type != CLEAN_ATTACK
        marker = RPI_MARKER if ATTACK_CODES.get(attack_type) == "rpi" else ""
        queries = []
        for trigger in self.trigger_queries:
            candidates = []
            for cached in self._raw_candidates(name, trigger):
                candidates.append({
                    **cached,
                    "metadata": apply_provenance_tier(cached["chunk_id"], cached["text"],
                                                      cached["metadata"], self.tier),
                })
            queries.append(ScenarioQuery(
                query_id=trigger["query_id"], query_text=trigger["query_text"],
                candidates=candidates, poison_marker=marker, attack_type=attack_type,
                poisoning_ratio=IEEE_RATIO if attacked else 0.0, seed=seed,
            ))
        return queries


def build_source(client=None, embed_query=None, tier: str = "as_stored", query_ids: set[str] | None = None,
                 n_retrieve: int = N_RETRIEVE, collections: dict | None = None,
                 clean_collection: str | None = None) -> Ieee5000Source:
    """Source on the live Chroma server and the API's own MiniLM embedder (the deployed pipeline)."""
    if client is None:
        from api.pipeline.vectorstore import get_client
        client = get_client()
    if embed_query is None:
        from api.pipeline.embedding import embed_texts
        embed_query = lambda text: embed_texts([text])[0]   # noqa: E731
    triggers = load_benchmark()
    if query_ids is not None:
        wanted = {q.lower() for q in query_ids}
        triggers = [t for t in triggers if t["query_id"].lower() in wanted]
    return Ieee5000Source(client, triggers, embed_query, n_retrieve, tier, collections, clean_collection)


def preflight(client, collections: dict | None = None, clean_collection: str | None = None) -> list[dict]:
    """Which ieee_5000 collections exist, and do their sizes match what the build scripts produced?"""
    collections = collections or COLLECTIONS
    wanted = [(clean_collection or CLEAN_COLLECTION, N_CLEAN_CHUNKS, 0)]
    wanted += [(name, N_CLEAN_CHUNKS + N_POISON_PER_ATTACK, N_POISON_PER_ATTACK) for name in collections.values()]
    rows = []
    for name, want_chunks, want_poison in wanted:
        try:
            col = client.get_collection(name=name)
            total = col.count()
            poisoned = len(col.get(where={"is_poisoned": True}, include=[])["ids"])
            rows.append({"collection": name, "exists": True, "chunks": total, "poisoned": poisoned,
                         "expected_chunks": want_chunks, "expected_poisoned": want_poison,
                         "ok": total == want_chunks and poisoned == want_poison})
        except Exception:
            rows.append({"collection": name, "exists": False, "chunks": 0, "poisoned": 0,
                         "expected_chunks": want_chunks, "expected_poisoned": want_poison, "ok": False})
    return rows
