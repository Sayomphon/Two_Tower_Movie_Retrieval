# Model Card — Two-Tower Movie Retrieval (ml100k-retrieval-v1)

## Model details

| | |
|---|---|
| Architecture | Two-tower: `user_id → StringLookup → Embedding(32)`, `movie_id → StringLookup → Embedding(32)`, dot-product affinity |
| Training objective | In-batch sampled softmax (SUM reduction) with accidental-hit masking |
| Optimizer | Adagrad, lr 0.1, 15 epochs, batch 256, seed 42 |
| Framework | TensorFlow 2.20 / Keras 3 (pure — no TFRS dependency) |
| Selection | R1 (dim 32) chosen over R2 (dim 64 + L2) by validation Recall@10 |
| Version | `ml100k-retrieval-v1` — stamped into every serving response |

## Intended use

- **Primary**: portfolio/education — demonstrating retrieval-stage methodology
  (temporal evaluation, baseline discipline, coverage/bias guardrails, serving contract).
- **Out of scope**: any commercial or production use (prohibited by MovieLens data terms);
  ranking; contexts where recommendations could cause harm without human curation.

## Training data

MovieLens 100K (GroupLens): 100,000 explicit ratings (1–5), 943 users, 1,682 movies,
collected 1997–1998. Downloaded at runtime over HTTPS, verified against a pinned SHA-256,
never redistributed with this repository. Positive-interaction rule: every rating counts
as an interaction (sensitivity to `rating ≥ 4` reported below).

**Split**: per-user temporal leave-last-1-out (last interaction → test, second-last → val),
audited programmatically for future leakage, minimum train history, and train/test overlap.
Vocabularies and popularity statistics are computed from train only.

## Evaluation results (test = one held-out future interaction per user, full-catalogue scoring)

| metric | popularity baseline | two-tower v1 |
|---|---|---|
| Recall@10 | 0.0657 | 0.0583 |
| NDCG@10 | 0.0331 | 0.0266 |
| Recall@50 | 0.1835 | 0.2312 |
| Coverage@10 | 0.054 | 0.893 |
| Top-10%-popular slot share | 1.000 | 0.063 |
| Gini exposure | 0.986 | 0.477 |

Slices (Recall@10): low-activity users 0.084 / medium 0.062 / high 0.029;
head test items 0.027 / tail 0.073. OOV test items: 0.2%.
Sensitivity (`rating ≥ 4` positives): Recall@10 0.061, Recall@50 0.219 — conclusions stable.

**Interpretation**: popularity wins at K=10; two-tower wins at K=50 and dominates
coverage/long-tail exposure. Reported as-is per the model-selection rule — ID-only signal
on 100K interactions is limited, and hiding that would misrepresent the system.

## Cold start & fallback policy

- **Unknown user** → global popularity Top-K, `fallback_used: true` in the response.
- **Unknown movie** → not in candidate catalogue (cannot be retrieved); item cold-start
  requires content features (stretch goal).

## Ethical considerations & risks

- **Feedback loops**: retrieval trained on logged interactions amplifies exposure bias;
  coverage/Gini guardrails are monitored for this reason.
- **No safety filtering**: production deployment would require content-safety and
  editorial-control layers after retrieval.
- **Privacy**: behavioural data is personal data; ML-100K is de-identified research data,
  but a production analogue needs consent, retention, and access controls.

## Caveats

Offline metrics on a 1998 dataset with one test event per user: high variance at K=10,
no impression data (unobserved ≠ disliked), and no claim of online lift. Any production
adoption requires online experimentation.
