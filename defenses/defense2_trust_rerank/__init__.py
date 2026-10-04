"""
defense2_trust_rerank
----------------------
Defense 2: Trust-Scored Re-ranking.

Assigns a trust score to every retrieved document based on source metadata,
document age, and corroboration by other documents in the corpus, then
re-weights the retrieval ranking so high-trust documents are favored over
raw cosine similarity alone.

Public API:
    from defenses.defense2_trust_rerank import rerank_results, TrustScorer
"""

from defenses.defense2_trust_rerank.reranker import rerank_results
from defenses.defense2_trust_rerank.trust_scorer import TrustScorer

__all__ = ["rerank_results", "TrustScorer"]
