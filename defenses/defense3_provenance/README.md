# Defense 3 — Provenance Verification

**Owner:** Akilan V S (Module 3) · **Status:** Implemented, tested, demo-ready

## What it does

An access-control style validation layer (PRD §6.5): checks whether a
document's claimed source identity is internally consistent, well-formed,
temporally sane, and untampered. Conceptually the same idea as role-based
access control, applied to document sources instead of user accounts.

| Check | Type | What it catches |
|---|---|---|
| Identity | **Hard block** | Claims official status but `source_id` isn't on the allow-list (spoofing) |
| Integrity | **Hard block** | Content hash doesn't match what was recorded at ingestion (tampering, e.g. appended prompt-injection payloads) |
| Format | Soft flag | Malformed CVE ID on an official-claiming document |
| Temporal | Soft flag | Modified date before published date, or dates in the future |

Hard-block failures are removed from context entirely. Soft-flag failures
stay in context but are marked — useful as a feature for Defense 4's
classifier later.

## Why this is not redundant with Defense 2

Defense 2 asks: *"how much should I trust this, relatively speaking?"*
(continuous score, can be gamed by borrowing corroboration — see the
combined demo).

Defense 3 asks: *"is this document's claimed identity even internally
consistent and untampered?"* (discrete pass/fail, independent of what
other documents exist in the corpus). It catches exactly the case Defense 2
alone cannot.

## Files

- `config.py` — allow-list (shared with Defense 2 via `corpus.metadata`), CVE
  ID format pattern, hard-block vs soft-flag policy
- `provenance_verifier.py` — the four checks
- `filter.py` — applies verification across a full retrieval result set
- `test_smoke.py` — 5 synthetic cases covering every check path

## Run it

```bash
export PYTHONPATH=.
python -m defenses.defense3_provenance.test_smoke
```

## What to show the panel

1. Run the smoke test live — 5 cases: legitimate, spoofed identity,
   malformed format, tampered content, temporal anomaly.
2. Point out that spoofing and tampering are **blocked**, while format and
   date irregularities are **flagged but kept** — this is a deliberate
   policy choice (real dirty data can have format quirks; there's no
   legitimate reason for spoofing or tampering).
3. For the strongest demo, run `defenses/demo_review3.py` instead — it
   chains Defense 2 → Defense 3 on one realistic attack scenario and shows
   why both layers are needed.
