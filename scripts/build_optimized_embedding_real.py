"""
PoisonBench
Attack 2 - Optimization-Based Embedding Attack
Main Roadmap Phase 6

Scales the optimization-based embedding attack to the frozen
20-query real benchmark.

IMPORTANT METHODOLOGY:

Attack 1 semantic-mimicry poison
        ->
use as Attack 2 seed
        ->
generate controlled semantic mutations
        ->
batch embed candidates
        ->
measure cosine similarity against trigger query
        ->
retain only improved candidate
        ->
repeat for 8 rounds

This avoids creating the seed directly from the trigger query.

Phase 6 ONLY builds optimized poison documents.

It DOES NOT ingest them into ChromaDB.
Ingestion is reserved for Phase 7.
"""

import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Use the SAME embedding implementation as the RAG pipeline.
from api.pipeline.embedding import embed_texts


# ============================================================
# CONFIGURATION
# ============================================================

BENCHMARK_FILE = (
    PROJECT_ROOT
    / "data"
    / "benchmark"
    / "real_trigger_queries_v1.jsonl"
)

SEMANTIC_MIMICRY_FILE = (
    PROJECT_ROOT
    / "data"
    / "poisoned"
    / "semantic_mimicry_real.jsonl"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "poisoned"
    / "optimized_embedding_attack_real.jsonl"
)

STATS_FILE = (
    PROJECT_ROOT
    / "data"
    / "poisoned"
    / "optimized_embedding_attack_stats.jsonl"
)


ROUNDS = 8

CANDIDATES_PER_ROUND = 12

BASE_RANDOM_SEED = 42

TOP_QUERY_TERMS = 10


# ============================================================
# RESULT STRUCTURE
# ============================================================

@dataclass
class Candidate:
    text: str
    similarity: float


# ============================================================
# TEXT UTILITIES
# ============================================================

def normalize_text(text: str) -> str:
    """
    Collapse repeated whitespace and remove leading/trailing
    whitespace.
    """

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def tokenize(text: str) -> list[str]:
    """
    Simple deterministic tokenizer used only for generating
    candidate semantic phrases.
    """

    return re.findall(
        r"[A-Za-z0-9_\-]+",
        text.lower(),
    )


def important_query_terms(query: str) -> list[str]:
    """
    Extract important vocabulary from the trigger query.

    We deliberately do NOT copy the complete query into the
    poison document.

    Instead, these terms provide controlled semantic material
    for candidate mutations.
    """

    stopwords = {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "what",
        "which",
        "who",
        "why",
        "how",
        "of",
        "for",
        "to",
        "in",
        "on",
        "with",
        "and",
        "or",
        "from",
        "by",
        "as",
        "at",
        "it",
        "this",
        "that",
        "these",
        "those",
        "does",
        "do",
        "can",
        "could",
        "would",
        "should",
        "be",
        "been",
        "about",
        "against",
        "using",
        "used",
        "when",
        "through",
        "into",
        "than",
        "then",
        "its",
    }

    terms = []

    for token in tokenize(query):

        if len(token) < 3:
            continue

        if token in stopwords:
            continue

        if token not in terms:
            terms.append(token)

    return terms[:TOP_QUERY_TERMS]


# ============================================================
# PHRASE GENERATION
# ============================================================

def generate_query_phrases(
    query: str,
) -> list[str]:
    """
    Build short query-relevant semantic phrases.

    This provides a generic equivalent of the manually defined
    Q05 mutation phrases from the controlled Phase-4 experiment.

    Whole-query repetition is intentionally avoided.
    """

    terms = important_query_terms(query)

    phrases = []

    # Individual important terms
    for term in terms:
        phrases.append(term)

    # Adjacent 2-term phrases
    for index in range(len(terms) - 1):

        phrase = (
            f"{terms[index]} "
            f"{terms[index + 1]}"
        )

        phrases.append(phrase)

    # Adjacent 3-term phrases
    for index in range(len(terms) - 2):

        phrase = (
            f"{terms[index]} "
            f"{terms[index + 1]} "
            f"{terms[index + 2]}"
        )

        phrases.append(phrase)

    # Remove duplicates while preserving order.
    unique_phrases = []

    for phrase in phrases:

        if phrase not in unique_phrases:
            unique_phrases.append(phrase)

    return unique_phrases


# ============================================================
# EMBEDDING SIMILARITY
# ============================================================

def cosine_similarity(
    vector_a: np.ndarray,
    vector_b: np.ndarray,
) -> float:
    """
    Embeddings produced by api.pipeline.embedding are already
    normalized.

    Therefore cosine similarity is equivalent to the dot product.
    """

    return float(
        np.dot(
            vector_a,
            vector_b,
        )
    )


def score_candidates(
    query_embedding: np.ndarray,
    candidate_texts: list[str],
) -> list[Candidate]:
    """
    Batch embed all candidates for a round.

    This uses the shared embedding pipeline so Phase 6 uses the
    same MiniLM model/settings as the RAG retrieval system.
    """

    candidate_embeddings = np.array(
        embed_texts(candidate_texts),
        dtype=np.float32,
    )

    scored = []

    for text, embedding in zip(
        candidate_texts,
        candidate_embeddings,
    ):

        similarity = cosine_similarity(
            query_embedding,
            embedding,
        )

        scored.append(
            Candidate(
                text=text,
                similarity=similarity,
            )
        )

    scored.sort(
        key=lambda item: item.similarity,
        reverse=True,
    )

    return scored


# ============================================================
# CONTROLLED MUTATION OPERATORS
# ============================================================

def mutation_append_phrase(
    text: str,
    phrases: list[str],
    rng: random.Random,
) -> str:
    """
    Append one relevant semantic phrase.
    """

    if not phrases:
        return text

    phrase = rng.choice(phrases)

    candidate = (
        text.rstrip(".")
        + ". "
        + phrase
        + "."
    )

    return normalize_text(candidate)


def mutation_prepend_phrase(
    text: str,
    phrases: list[str],
    rng: random.Random,
) -> str:
    """
    Prepend one relevant semantic phrase.
    """

    if not phrases:
        return text

    phrase = rng.choice(phrases)

    candidate = (
        phrase
        + ". "
        + text
    )

    return normalize_text(candidate)


def mutation_repeat_phrase(
    text: str,
    phrases: list[str],
    rng: random.Random,
) -> str:
    """
    Reinforce one semantic phrase without copying the complete
    trigger query.
    """

    if not phrases:
        return text

    phrase = rng.choice(phrases)

    candidate = (
        text.rstrip(".")
        + ". "
        + phrase
        + " is relevant to this vulnerability."
    )

    return normalize_text(candidate)


def mutation_combine_phrases(
    text: str,
    phrases: list[str],
    rng: random.Random,
) -> str:
    """
    Combine two short query-relevant phrases.

    Similar in spirit to the controlled Q05 optimizer.
    """

    if len(phrases) < 2:
        return mutation_append_phrase(
            text,
            phrases,
            rng,
        )

    phrase_1 = rng.choice(phrases)

    phrase_2 = rng.choice(phrases)

    attempts = 0

    while (
        phrase_2 == phrase_1
        and attempts < 10
    ):
        phrase_2 = rng.choice(phrases)
        attempts += 1

    candidate = (
        text.rstrip(".")
        + ". "
        + phrase_1
        + " involving "
        + phrase_2
        + "."
    )

    return normalize_text(candidate)


MUTATION_OPERATORS = [
    mutation_append_phrase,
    mutation_prepend_phrase,
    mutation_repeat_phrase,
    mutation_combine_phrases,
]


# ============================================================
# CANDIDATE GENERATION
# ============================================================

def generate_mutations(
    base_text: str,
    query: str,
    count: int,
    rng: random.Random,
) -> list[str]:
    """
    Generate controlled mutations from the current best document.

    IMPORTANT:

    The unchanged base text is always retained.

    Therefore optimization cannot accidentally accept a worse
    candidate.
    """

    phrases = generate_query_phrases(
        query
    )

    candidates = [
        base_text
    ]

    attempts = 0

    maximum_attempts = count * 50

    while (
        len(candidates) < count
        and attempts < maximum_attempts
    ):

        attempts += 1

        operator = rng.choice(
            MUTATION_OPERATORS
        )

        candidate = operator(
            base_text,
            phrases,
            rng,
        )

        if candidate not in candidates:
            candidates.append(candidate)

    # Extremely unlikely fallback if candidate generation produced
    # too many duplicates.
    while len(candidates) < count:

        candidates.append(
            base_text
        )

    return candidates


# ============================================================
# QUERY OPTIMIZATION
# ============================================================

def optimize_query(
    query_id: str,
    query: str,
    seed_poison: str,
    rounds: int = ROUNDS,
    candidates_per_round: int = CANDIDATES_PER_ROUND,
    seed: int = BASE_RANDOM_SEED,
) -> dict:
    """
    Optimize one Attack-1 semantic-mimicry document against the
    corresponding frozen trigger query.
    """

    rng = random.Random(
        seed
    )

    # Embed the trigger query only once.
    query_embedding = np.array(
        embed_texts(
            [query]
        )[0],
        dtype=np.float32,
    )

    seed_embedding = np.array(
        embed_texts(
            [seed_poison]
        )[0],
        dtype=np.float32,
    )

    seed_similarity = cosine_similarity(
        query_embedding,
        seed_embedding,
    )

    current_text = seed_poison

    current_similarity = seed_similarity

    accepted_rounds = 0

    round_history = []

    print()
    print("=" * 72)
    print(
        f"Optimizing {query_id}"
    )
    print("=" * 72)

    print(
        f"Query              : "
        f"{query}"
    )

    print(
        f"Seed similarity    : "
        f"{seed_similarity:.6f}"
    )

    for round_number in range(
        1,
        rounds + 1,
    ):

        candidates = generate_mutations(
            base_text=current_text,
            query=query,
            count=candidates_per_round,
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

            current_similarity = (
                best.similarity
            )

            accepted_rounds += 1

        round_history.append(
            {
                "round": round_number,
                "best_similarity": (
                    best.similarity
                ),
                "accepted": improved,
                "current_similarity": (
                    current_similarity
                ),
            }
        )

        print(
            f"Round {round_number:02d} | "
            f"best={best.similarity:.6f} | "
            f"accepted={improved}"
        )

    improvement = (
        current_similarity
        - seed_similarity
    )

    print("-" * 72)

    print(
        f"Final similarity   : "
        f"{current_similarity:.6f}"
    )

    print(
        f"Improvement        : "
        f"{improvement:+.6f}"
    )

    print(
        f"Accepted rounds    : "
        f"{accepted_rounds}/{rounds}"
    )

    return {
        "query_id": query_id,
        "query": query,
        "seed_poison_text": (
            seed_poison
        ),
        "seed_similarity": (
            seed_similarity
        ),
        "optimized_similarity": (
            current_similarity
        ),
        "improvement": (
            improvement
        ),
        "accepted_rounds": (
            accepted_rounds
        ),
        "total_rounds": (
            rounds
        ),
        "candidates_per_round": (
            candidates_per_round
        ),
        "random_seed": (
            seed
        ),
        "optimized_poison_text": (
            current_text
        ),
        "round_history": (
            round_history
        ),
    }


# ============================================================
# JSONL LOADING
# ============================================================

def load_jsonl(
    path: Path,
) -> list[dict]:
    """
    Load JSONL file.
    """

    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:

                rows.append(
                    json.loads(
                        line
                    )
                )

            except json.JSONDecodeError as exc:

                raise ValueError(
                    f"Invalid JSON on line "
                    f"{line_number} of {path}"
                ) from exc

    return rows


# ============================================================
# QUERY FIELD EXTRACTION
# ============================================================

def extract_query_fields(
    row: dict,
) -> tuple[str, str]:
    """
    Supports likely benchmark field names without modifying the
    frozen benchmark.
    """

    query_id = (
        row.get("query_id")
        or row.get("id")
        or row.get("qid")
    )

    query = (
        row.get("query")
        or row.get("query_text")
        or row.get("question")
        or row.get("text")
        or row.get("trigger_query")
    )

    if not query_id:

        raise KeyError(
            f"Could not locate query ID "
            f"in benchmark row: {row}"
        )

    if not query:

        raise KeyError(
            f"Could not locate query text "
            f"in benchmark row: {row}"
        )

    return (
        str(query_id),
        str(query),
    )


# ============================================================
# OUTPUT
# ============================================================

def save_jsonl(
    path: Path,
    rows: list[dict],
):
    """
    Save list of dictionaries as JSONL.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        for row in rows:

            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


# ============================================================
# VALIDATION
# ============================================================

def validate_semantic_records(
    records: list[dict],
):
    """
    Confirm the Attack-1 corpus contains the fields required for
    Phase 6.
    """

    required_fields = {
        "query_id",
        "legitimate_doc_id",
        "poison_doc_id",
        "poison_text",
    }

    for record in records:

        missing = (
            required_fields
            - record.keys()
        )

        if missing:

            raise KeyError(
                f"Semantic-mimicry record "
                f"{record.get('query_id')} "
                f"is missing fields: "
                f"{sorted(missing)}"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("POISONBENCH")
    print(
        "ATTACK 2 - OPTIMIZATION-BASED "
        "EMBEDDING ATTACK"
    )
    print(
        "MAIN ROADMAP PHASE 6"
    )
    print(
        "SCALE OPTIMIZER TO FROZEN "
        "20-QUERY BENCHMARK"
    )
    print("=" * 72)

    print()

    print(
        f"Benchmark       : "
        f"{BENCHMARK_FILE}"
    )

    print(
        f"Attack-1 seeds  : "
        f"{SEMANTIC_MIMICRY_FILE}"
    )

    print(
        f"Output          : "
        f"{OUTPUT_FILE}"
    )

    print(
        f"Stats           : "
        f"{STATS_FILE}"
    )

    print(
        f"Rounds          : "
        f"{ROUNDS}"
    )

    print(
        f"Candidates      : "
        f"{CANDIDATES_PER_ROUND} "
        f"per round"
    )

    # --------------------------------------------------------
    # Check required files
    # --------------------------------------------------------

    if not BENCHMARK_FILE.exists():

        raise FileNotFoundError(
            f"Benchmark file not found:\n"
            f"{BENCHMARK_FILE}"
        )

    if not SEMANTIC_MIMICRY_FILE.exists():

        raise FileNotFoundError(
            f"Attack-1 semantic-mimicry "
            f"file not found:\n"
            f"{SEMANTIC_MIMICRY_FILE}"
        )

    # --------------------------------------------------------
    # Load benchmark
    # --------------------------------------------------------

    benchmark = load_jsonl(
        BENCHMARK_FILE
    )

    print()
    print(
        f"Frozen benchmark queries loaded: "
        f"{len(benchmark)}"
    )

    if len(benchmark) != 20:

        print(
            "WARNING: Expected 20 frozen "
            "queries, but found "
            f"{len(benchmark)}."
        )

    # --------------------------------------------------------
    # Load Attack-1 semantic-mimicry poisons
    # --------------------------------------------------------

    semantic_records = load_jsonl(
        SEMANTIC_MIMICRY_FILE
    )

    validate_semantic_records(
        semantic_records
    )

    print(
        f"Attack-1 semantic seeds loaded: "
        f"{len(semantic_records)}"
    )

    semantic_map = {
        str(
            record["query_id"]
        ).strip().lower():
        record

        for record
        in semantic_records
    }

    if len(semantic_map) != 20:

        print(
            "WARNING: Expected 20 unique "
            "semantic-mimicry seeds, but "
            f"found {len(semantic_map)}."
        )

    # --------------------------------------------------------
    # Optimization
    # --------------------------------------------------------

    results = []

    for index, benchmark_row in enumerate(
        benchmark
    ):

        query_id, query = (
            extract_query_fields(
                benchmark_row
            )
        )

        lookup_id = (
            query_id
            .strip()
            .lower()
        )

        if lookup_id not in semantic_map:

            raise KeyError(
                f"No Attack-1 semantic seed "
                f"found for query "
                f"{query_id}"
            )

        semantic_record = (
            semantic_map[
                lookup_id
            ]
        )

        seed_poison = str(
            semantic_record[
                "poison_text"
            ]
        ).strip()

        if not seed_poison:

            raise ValueError(
                f"Empty Attack-1 poison text "
                f"for {query_id}"
            )

        # Deterministic but query-specific
        # random seed.
        query_seed = (
            BASE_RANDOM_SEED
            + index
        )

        result = optimize_query(
            query_id=query_id,
            query=query,
            seed_poison=seed_poison,
            rounds=ROUNDS,
            candidates_per_round=(
                CANDIDATES_PER_ROUND
            ),
            seed=query_seed,
        )

        # Preserve mapping back to the legitimate
        # CVE and the Attack-1 seed document.
        result[
            "legitimate_doc_id"
        ] = str(
            semantic_record[
                "legitimate_doc_id"
            ]
        )

        result[
            "semantic_seed_doc_id"
        ] = str(
            semantic_record[
                "poison_doc_id"
            ]
        )

        result[
            "semantic_seed_source"
        ] = str(
            semantic_record.get(
                "source",
                "controlled_semantic_mimicry",
            )
        )

        results.append(
            result
        )

    # --------------------------------------------------------
    # Build clean Attack-2 poison corpus
    # --------------------------------------------------------

    poison_rows = []

    for result in results:

        normalized_query_id = (
            result["query_id"]
            .strip()
            .lower()
        )

        poison_doc_id = (
            f"POISON-OBEA-"
            f"{normalized_query_id}"
        )

        poison_rows.append(
            {
                "query_id": (
                    normalized_query_id
                ),

                "legitimate_doc_id": (
                    result[
                        "legitimate_doc_id"
                    ]
                ),

                "semantic_seed_doc_id": (
                    result[
                        "semantic_seed_doc_id"
                    ]
                ),

                "poison_doc_id": (
                    poison_doc_id
                ),

                "source": (
                    "controlled_"
                    "optimized_embedding"
                ),

                "attack_type": (
                    "optimization_embedding"
                ),

                "is_poisoned": True,

                "poison_text": (
                    result[
                        "optimized_poison_text"
                    ]
                ),

                "seed_similarity": (
                    result[
                        "seed_similarity"
                    ]
                ),

                "optimized_similarity": (
                    result[
                        "optimized_similarity"
                    ]
                ),

                "improvement": (
                    result[
                        "improvement"
                    ]
                ),
            }
        )

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    save_jsonl(
        OUTPUT_FILE,
        poison_rows,
    )

    save_jsonl(
        STATS_FILE,
        results,
    )

    # --------------------------------------------------------
    # Summary metrics
    # --------------------------------------------------------

    improved_count = sum(
        1
        for result in results
        if result["improvement"] > 0
    )

    unchanged_count = (
        len(results)
        - improved_count
    )

    mean_seed_similarity = float(
        np.mean(
            [
                result[
                    "seed_similarity"
                ]
                for result in results
            ]
        )
    )

    mean_optimized_similarity = float(
        np.mean(
            [
                result[
                    "optimized_similarity"
                ]
                for result in results
            ]
        )
    )

    mean_improvement = float(
        np.mean(
            [
                result[
                    "improvement"
                ]
                for result in results
            ]
        )
    )

    max_improvement_result = max(
        results,
        key=lambda row:
            row["improvement"],
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("PHASE 6 SUMMARY")
    print("=" * 72)

    print(
        f"Queries processed             : "
        f"{len(results)}"
    )

    print(
        f"Queries improved              : "
        f"{improved_count}/"
        f"{len(results)}"
    )

    print(
        f"Queries unchanged             : "
        f"{unchanged_count}/"
        f"{len(results)}"
    )

    print(
        f"Mean seed similarity          : "
        f"{mean_seed_similarity:.6f}"
    )

    print(
        f"Mean optimized similarity     : "
        f"{mean_optimized_similarity:.6f}"
    )

    print(
        f"Mean absolute improvement     : "
        f"{mean_improvement:+.6f}"
    )

    print(
        f"Maximum improvement           : "
        f"{max_improvement_result['query_id']} "
        f"{max_improvement_result['improvement']:+.6f}"
    )

    print()
    print(
        "Optimized poison corpus saved:"
    )
    print(
        OUTPUT_FILE
    )

    print()
    print(
        "Optimization statistics saved:"
    )
    print(
        STATS_FILE
    )

    print()
    print("=" * 72)

    print(
        "IMPORTANT:"
    )

    print(
        "Phase 6 only BUILDS the "
        "Attack-2 optimized poison corpus."
    )

    print(
        "No Attack-2 documents have been "
        "ingested into ChromaDB."
    )

    print(
        "ChromaDB ingestion is reserved "
        "for Main Roadmap Phase 7."
    )

    print("=" * 72)


if __name__ == "__main__":
    main()