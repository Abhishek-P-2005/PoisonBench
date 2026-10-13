"""
evaluation/plots.py
-------------------
Deliverable 1: embedding map of clean vs poisoned chunks (t-SNE) with the
chunks Defense 1 flags at the calibrated threshold circled.
"""

from __future__ import annotations

import numpy as np


def plot_embedding_map(source, attack: str, ratio: float, seed: int, soft_z: float, path: str, max_points: int = 1500) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    from evaluation.pipeline import order_by_similarity, strip_labels
    from evaluation.stages import D1Stage

    scenario_queries = source.iter_queries(attack, ratio, seed)
    points, poisoned, flagged = [], [], []
    seen = set()
    for scenario_query in scenario_queries:
        batch = order_by_similarity([strip_labels(c) for c in scenario_query.candidates])
        output = D1Stage(soft_z=soft_z, hard_z=max(soft_z, 1e9)).run(batch)
        truth = {c["chunk_id"]: bool(c["metadata"].get("is_poisoned")) for c in scenario_query.candidates}
        for chunk in scenario_query.candidates:
            if chunk["chunk_id"] in seen:
                continue
            seen.add(chunk["chunk_id"])
            points.append(chunk["embedding"])
            poisoned.append(truth[chunk["chunk_id"]])
            flagged.append(chunk["chunk_id"] in output.flagged_ids)

    points = np.array(points[:max_points])
    poisoned = np.array(poisoned[:max_points])
    flagged = np.array(flagged[:max_points])
    perplexity = max(5, min(30, len(points) // 4))
    coords = TSNE(n_components=2, perplexity=perplexity, random_state=seed, init="pca").fit_transform(points)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(coords[~poisoned, 0], coords[~poisoned, 1], s=14, c="#1a73e8", label="clean")
    ax.scatter(coords[poisoned, 0], coords[poisoned, 1], s=22, c="#d93025", label="poisoned")
    ax.scatter(coords[flagged, 0], coords[flagged, 1], s=90, facecolors="none", edgecolors="black", label=f"Defense 1 flag (z >= {soft_z:.2f})")
    ax.set_title(f"Retrieved chunks, attack={attack}, ratio={ratio:.0%}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
