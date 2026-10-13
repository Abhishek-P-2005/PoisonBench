"""
evaluation/real_source.py
-------------------------
Runs the harness on this repo's REAL benchmark instead of synthetic data.

  trigger queries   data/benchmark/real_trigger_queries_v1.jsonl   (20 queries)
  clean baseline    Chroma collection  poisonbench_rpi_clean_v1
  attacked          Chroma collections poisonbench_b_{sm|obea|rpi}_{1|5|10|20}pct_v1
                    (built by scripts/phase_b4_*; each holds the clean corpus plus
                     the poisoned chunks, isolated per attack and ratio)

Harness attack names map to the repo's attacks:
    naive -> sm (semantic mimicry)   optimized -> obea (optimized embedding)
    injection -> rpi (retrieved prompt injection)

Retrieval, embedding and the Chroma client are the API's own
(api.pipeline.vectorstore / api.pipeline.embedding), so the harness measures the
pipeline as deployed. Missing collections raise an error; there is no silent
fallback to synthetic data. Real collections are static, so use a single seed.

Generation-level success is defined only for the RPI attack, whose payload makes
the model emit POISONBENCH_RPI_SUCCESS (same marker as
scripts/evaluate_retrieved_prompt_injection_generation_real.py). For sm/obea no
marker exists, so gen_success is left empty rather than reported as 0.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from evaluation.config import CLEAN_ATTACK, N_RETRIEVE
from evaluation.scenarios import ChromaScenarioSource

ROOT = Path(__file__).resolve().parents[1]
TRIGGER_PATH = ROOT / "data" / "benchmark" / "real_trigger_queries_v1.jsonl"
SPLIT_MANIFEST = ROOT / "data" / "results" / "phase_c" / "split_manifest_v1.json"
CLEAN_COLLECTION = "poisonbench_rpi_clean_v1"
ATTACK_CODES = {"naive": "sm", "optimized": "obea", "injection": "rpi",
                "sm": "sm", "obea": "obea", "rpi": "rpi"}
RPI_MARKER = "POISONBENCH_RPI_SUCCESS"


def load_trigger_queries(path: Path = TRIGGER_PATH) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows.append({"query_id": row["query_id"], "query_text": row["query"],
                         "expected_doc_id": row.get("expected_doc_id")})
    return rows


def heldout_query_ids(manifest_path: Path = SPLIT_MANIFEST) -> set[str]:
    """Queries the Phase C/E classifiers never trained on ('target:q07' -> 'q07')."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    return {group.split(":", 1)[1].lower() for group in manifest["test_target_groups"]}


def collection_name(attack: str, ratio: float) -> str:
    if attack == CLEAN_ATTACK:
        return CLEAN_COLLECTION
    code = ATTACK_CODES.get(attack)
    if code is None:
        raise ValueError(f"unknown attack '{attack}'; expected one of {sorted(ATTACK_CODES)}")
    return f"poisonbench_b_{code}_{int(round(ratio * 100))}pct_v1"


class PhaseBChromaSource(ChromaScenarioSource):
    """ChromaScenarioSource that switches collection per (attack, ratio)."""

    def __init__(self, client, trigger_queries, embed_query, n_retrieve: int = N_RETRIEVE,
                 provenance: str = "auto"):
        super().__init__(None, trigger_queries, embed_query=embed_query,
                         n_retrieve=n_retrieve, provenance=provenance)
        self.client = client

    def iter_queries(self, attack_type: str, ratio: float, seed: int):
        name = collection_name(attack_type, ratio)
        try:
            self.collection = self.client.get_collection(name=name)
        except Exception as exc:
            raise RuntimeError(f"Chroma collection '{name}' not found ({exc}). Build it with the "
                               "phase_b4 ingestion scripts first.") from exc
        queries = super().iter_queries(attack_type, ratio, seed)
        marker = RPI_MARKER if ATTACK_CODES.get(attack_type) == "rpi" else ""
        return [dataclasses.replace(q, poison_marker=marker) for q in queries]


def build_chroma_source(provenance: str = "auto", query_ids: set[str] | None = None,
                        n_retrieve: int = N_RETRIEVE) -> PhaseBChromaSource:
    from api.pipeline.embedding import embed_texts
    from api.pipeline.vectorstore import get_client
    triggers = load_trigger_queries()
    if query_ids is not None:
        triggers = [t for t in triggers if t["query_id"].lower() in query_ids]
    return PhaseBChromaSource(get_client(), triggers, lambda text: embed_texts([text])[0],
                              n_retrieve=n_retrieve, provenance=provenance)


def preflight(client, attacks=("naive", "optimized", "injection"), ratios=(0.01, 0.05, 0.10, 0.20)) -> list[dict]:
    """Which Phase-B collections exist, how big they are, and how many chunks are labelled poisoned."""
    rows = []
    wanted = [CLEAN_COLLECTION] + [collection_name(a, r) for a in attacks for r in ratios]
    for name in wanted:
        try:
            col = client.get_collection(name=name)
            total = col.count()
            poisoned = len(col.get(where={"is_poisoned": True}, include=[])["ids"])
            rows.append({"collection": name, "exists": True, "chunks": total, "poisoned": poisoned})
        except Exception:
            rows.append({"collection": name, "exists": False, "chunks": 0, "poisoned": 0})
    return rows
