"""
Phase 4 - Optimization-Based Embedding Attack
Automatic candidate mutation + iterative optimization

Initial controlled experiment:
    q05 only

The optimizer:
    seed poison
        -> generate candidate mutations
        -> batch embed candidates
        -> score against trigger query
        -> retain best candidate
        -> repeat
"""

import random
import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api.pipeline.embedding import embed_texts


QUERY_ID = "q05"

TRIGGER_QUERY = (
    "Which UltraVNC vulnerability uses a weak pseudo-random generator "
    "for authentication challenge bytes, making the challenge predictable?"
)

SEED_TEXT = (
    "CVE-2026-44040 UltraVNC authentication vulnerability involving weak "
    "pseudo-random challenge generation. Predictable VNC authentication "
    "challenge bytes can enable challenge prediction and offline attacks."
)

RANDOM_SEED = 42

ROUNDS = 8
CANDIDATES_PER_ROUND = 12


@dataclass
class Candidate:
    text: str
    similarity: float


MUTATION_PHRASES = [
    "UltraVNC authentication challenge",
    "weak pseudo-random generator",
    "predictable authentication challenge bytes",
    "VNC authentication",
    "challenge prediction",
    "predictable challenge generation",
    "pseudo-random challenge bytes",
    "authentication challenge prediction",
    "UltraVNC vulnerability",
    "weak random challenge generation",
]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Embeddings are normalized by embedding.py.
    Therefore cosine similarity equals dot product.
    """
    return float(np.dot(a, b))


def generate_mutations(
    base_text: str,
    count: int,
    rng: random.Random,
) -> list[str]:
    """
    Generate controlled text mutations from the current best candidate.

    Mutation operations:
    - append query-relevant phrases
    - prepend query-relevant phrases
    - repeat a selected semantic phrase once
    - combine two relevant phrases

    The original base text is always retained as one candidate so that
    optimization cannot accidentally accept a worse candidate.
    """

    candidates = [base_text]

    while len(candidates) < count:

        operation = rng.choice(
            [
                "append",
                "prepend",
                "repeat",
                "combine",
            ]
        )

        phrase1 = rng.choice(MUTATION_PHRASES)

        if operation == "append":

            candidate = (
                base_text.rstrip(".")
                + ". "
                + phrase1
                + "."
            )

        elif operation == "prepend":

            candidate = (
                phrase1
                + ". "
                + base_text
            )

        elif operation == "repeat":

            candidate = (
                base_text.rstrip(".")
                + ". "
                + phrase1
                + " is central to this vulnerability."
            )

        else:

            phrase2 = rng.choice(MUTATION_PHRASES)

            candidate = (
                base_text.rstrip(".")
                + ". "
                + phrase1
                + " involving "
                + phrase2
                + "."
            )

        if candidate not in candidates:
            candidates.append(candidate)

    return candidates


def score_candidates(
    query_embedding: np.ndarray,
    texts: list[str],
) -> list[Candidate]:

    candidate_embeddings = np.array(
        embed_texts(texts),
        dtype=np.float32,
    )

    scored = []

    for text, embedding in zip(
        texts,
        candidate_embeddings,
    ):

        score = cosine_similarity(
            query_embedding,
            embedding,
        )

        scored.append(
            Candidate(
                text=text,
                similarity=score,
            )
        )

    scored.sort(
        key=lambda item: item.similarity,
        reverse=True,
    )

    return scored


def main():

    rng = random.Random(RANDOM_SEED)

    print("=" * 72)
    print("PHASE 4 - AUTOMATIC EMBEDDING OPTIMIZATION")
    print("=" * 72)

    print(f"Query ID              : {QUERY_ID}")
    print(f"Rounds                : {ROUNDS}")
    print(
        f"Candidates per round  : "
        f"{CANDIDATES_PER_ROUND}"
    )
    print(f"Random seed           : {RANDOM_SEED}")
    print()

    query_embedding = np.array(
        embed_texts([TRIGGER_QUERY])[0],
        dtype=np.float32,
    )

    seed_embedding = np.array(
        embed_texts([SEED_TEXT])[0],
        dtype=np.float32,
    )

    seed_similarity = cosine_similarity(
        query_embedding,
        seed_embedding,
    )

    current_text = SEED_TEXT
    current_similarity = seed_similarity

    print(
        f"Initial seed similarity: "
        f"{seed_similarity:.6f}"
    )
    print()

    optimization_history = []

    for round_number in range(
        1,
        ROUNDS + 1,
    ):

        candidates = generate_mutations(
            base_text=current_text,
            count=CANDIDATES_PER_ROUND,
            rng=rng,
        )

        scored = score_candidates(
            query_embedding,
            candidates,
        )

        best = scored[0]

        improved = (
            best.similarity
            > current_similarity
        )

        if improved:
            current_text = best.text
            current_similarity = best.similarity

        optimization_history.append(
            {
                "round": round_number,
                "best_similarity":
                    current_similarity,
                "improved": improved,
            }
        )

        print(
            f"Round {round_number:02d} | "
            f"best={best.similarity:.6f} | "
            f"accepted={improved}"
        )

    print()
    print("=" * 72)
    print("OPTIMIZATION RESULT")
    print("=" * 72)

    print(
        f"Seed similarity      : "
        f"{seed_similarity:.6f}"
    )

    print(
        f"Optimized similarity : "
        f"{current_similarity:.6f}"
    )

    improvement = (
        current_similarity
        - seed_similarity
    )

    print(
        f"Absolute improvement : "
        f"{improvement:+.6f}"
    )

    print()
    print("Optimized poison text:")
    print()
    print(current_text)

    print()
    print("=" * 72)


if __name__ == "__main__":
    main()