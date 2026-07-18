"""Two-tower movie recommendation retrieval system on MovieLens 100K.

Package layout:
    config    — dataclass configuration + JSON persistence (split rule, hparams, paths)
    data      — dataset download (HTTPS + SHA-256 verification), loading, data contract
    splits    — temporal leave-last-k-out split per user + leakage audit
    baseline  — global popularity Top-K baseline (train-only statistics)
    model     — two-tower user/movie embedding model (in-batch sampled softmax)
    evaluate  — Recall@K, NDCG@K, catalogue coverage, popularity bias, user slices
    index     — brute-force serving index (SavedModel) + unknown-user fallback
    cli       — command line entry points: prepare / train / evaluate / recommend
"""

__version__ = "0.1.0"
