
import json
import re
from pathlib import Path
from collections import defaultdict

import chromadb
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from api.pipeline.embedding import embed_texts
from defenses.defense4_xgboost.feature_engineering import (
    FEATURE_COLUMNS, text_stat_features, retrieval_score_features
)

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "ieee_5000"
OUTPUT = BASE / "processed" / "defense_features_5300.csv"
MANIFEST = BASE / "processed" / "defense_features_manifest.json"

COLLECTIONS = {
    "clean": "poisonbench_ieee_5000_clean",
    "sm": "poisonbench_ieee_5000_sm",
    "obea": "poisonbench_ieee_5000_obea",
    "rpi": "poisonbench_ieee_5000_rpi",
}

FEATURES = [
    f for f in FEATURE_COLUMNS
    if f not in ("source_is_official", "source_age_days")
]

def fetch(collection):
    rows = []
    for offset in range(0, collection.count(), 500):
        batch = collection.get(
            limit=500,
            offset=offset,
            include=["documents", "metadatas", "embeddings"]
        )
        for i, cid in enumerate(batch["ids"]):
            rows.append({
                "chunk_id": cid,
                "text": batch["documents"][i],
                "metadata": batch["metadatas"][i],
                "embedding": np.asarray(batch["embeddings"][i], dtype=np.float32),
            })
    return rows

def main():
    if OUTPUT.exists() or MANIFEST.exists():
        raise FileExistsError("IEEE defense output already exists; refusing overwrite")

    client = chromadb.HttpClient(host="localhost", port=8001)
    collections = {
        key: client.get_collection(name)
        for key, name in COLLECTIONS.items()
    }

    clean = fetch(collections["clean"])
    assert len(clean) == 5000

    rows = list(clean)

    for attack in ("sm", "obea", "rpi"):
        attack_rows = fetch(collections[attack])
        poisons = [
            r for r in attack_rows
            if r["metadata"].get("is_poisoned") is True
        ]
        assert len(poisons) == 100, f"{attack}: expected 100 poison chunks"
        for r in poisons:
            r["attack_type"] = attack
        rows.extend(poisons)

    for r in clean:
        r["attack_type"] = "clean"

    assert len(rows) == 5300
    assert len({r["chunk_id"] for r in rows}) == 5300

    print("Loaded 5,000 clean + 300 poison chunks", flush=True)

    embeddings = np.stack([r["embedding"] for r in rows])
    centroid = embeddings.mean(axis=0)

    nn = NearestNeighbors(n_neighbors=2, metric="euclidean", algorithm="brute")
    nn.fit(embeddings)
    distances, _ = nn.kneighbors(embeddings)
    nearest = distances[:, 1]

    print("Embedding features calculated", flush=True)

    benchmark = [
        json.loads(line)
        for line in (BASE / "processed" / "benchmark_100.jsonl")
            .read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    query_vectors = embed_texts([q["query"] for q in benchmark])
    scores = defaultdict(list)

    for attack, collection in collections.items():
        for q, vector in zip(benchmark, query_vectors):
            result = collection.query(
                query_embeddings=[vector],
                n_results=5,
                include=["distances"]
            )
            for cid, distance in zip(result["ids"][0], result["distances"][0]):
                scores[cid].append(1.0 - float(distance))

    all_scores = [s for values in scores.values() for s in values]
    print("Retrieval features calculated", flush=True)

    output_rows = []

    for i, r in enumerate(rows):
        cid = r["chunk_id"]
        meta = r["metadata"]

        features = {
            "embedding_dist_from_centroid":
                float(np.linalg.norm(embeddings[i] - centroid)),
            "embedding_dist_from_nearest_neighbor": float(nearest[i]),
        }

        features.update(text_stat_features(r["text"]))
        features.update(retrieval_score_features(
            scores.get(cid, []), all_scores
        ))

        features["chunk_id"] = cid
        features["doc_id"] = meta["doc_id"]
        features["attack_type"] = r["attack_type"]
        features["is_poisoned"] = int(r["attack_type"] != "clean")

        output_rows.append(features)

    df = pd.DataFrame(output_rows)

    assert len(df) == 5300
    assert df["is_poisoned"].sum() == 300
    assert df[FEATURES].notna().all().all()
    assert np.isfinite(df[FEATURES].to_numpy(dtype=float)).all()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)

    MANIFEST.write_text(json.dumps({
        "dataset_rows": 5300,
        "clean_rows": 5000,
        "poison_rows": 300,
        "features": FEATURES,
        "collections": COLLECTIONS,
        "retrieval_top_k": 5,
        "transductive_features": True,
        "notes": "Controlled in-corpus classification; not independent-corpus generalization."
    }, indent=2), encoding="utf-8")

    print("PASS: Defense dataset created", flush=True)
    print("Rows:", len(df), flush=True)
    print("Features:", len(FEATURES), flush=True)
    print("Saved:", OUTPUT, flush=True)
    print("Manifest:", MANIFEST, flush=True)

if __name__ == "__main__":
    main()
