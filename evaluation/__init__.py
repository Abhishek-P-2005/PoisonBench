"""
evaluation — PoisonBench evaluation harness (Akilan, PRD v2.0 Sections 3.3 / 4).

Runs attack type x poisoning ratio x defense configuration sweeps, logs
every result to PostgreSQL (or SQLite for local runs), and feeds the
analysis modules:

    analysis/calibration.py   Module 1  ROC / PR threshold calibration
    analysis/ordering.py      Module 2  defense ordering permutations
    analysis/interaction.py   Module 3  isolation, pairs, marginal uplift
    analysis/pareto.py        Module 4  3D Pareto trade-off profiling
    router_benchmark.py       Module 5  adaptive router vs static pipeline
"""
