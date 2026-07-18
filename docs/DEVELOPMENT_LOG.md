# Development Log — Two-Tower Movie Retrieval

> บันทึกการขึ้นโปรเจคตาม blueprint ใน `05_movie_recommender_ai_engineering_plan.docx`
> วันที่: 18 กรกฎาคม 2026 · Environment: macOS arm64, Python 3.11.15, TensorFlow 2.20.0

---

## 1. สิ่งที่ทำไปทั้งหมด (ตามลำดับ)

### 1.1 Scaffold + Environment
- `git init` + `.gitignore` (กัน `data/`, `artifacts/`, `.venv/`, secrets ทุกรูปแบบ)
- `pyproject.toml` — package `movie-retrieval` พร้อม CLI entry point, pinned dependency ranges,
  ruff config (รวม `S` security rules), pytest config
- สร้าง venv ด้วย Python 3.11 (Homebrew) → ติดตั้ง dependencies → freeze เป็น
  `requirements-lock.txt` เพื่อ reproducibility

### 1.2 Source package (`src/movie_retrieval/`)
| module | หน้าที่ |
|---|---|
| `config.py` | dataclass ทุก design decision (split rule, hparams) + serialize ลง `split_config.json` |
| `data.py` | download HTTPS + **pinned SHA-256 verify**, zip-slip-safe extract, **data contract** (schema/range/duplicate/known stats) |
| `splits.py` | temporal leave-last-1-out ต่อ user + `audit_no_leakage()` ที่ raise `LeakageError` ถ้าพบ future leakage |
| `baseline.py` | popularity Top-K จาก train เท่านั้น + seen filtering + deterministic tie-break |
| `model.py` | two-tower (pure TF/Keras 3): StringLookup → Embedding(32) → dot product, in-batch sampled softmax + accidental-hit masking, custom GradientTape loop |
| `evaluate.py` | Recall@K / NDCG@K / HitRate, catalogue coverage, popularity bias (percentile / top-10% share / Gini), slices ตาม user activity + item popularity |
| `index.py` | BruteForce index เป็น SavedModel (pure TF ops — ไม่ผูก Keras version), unknown-user fallback → popularity, `RetrievalService` พร้อม input validation |
| `pipeline.py` | orchestration: prepare → train (เลือก model จาก val) → evaluate (test ครั้งเดียว) → export artifacts |
| `cli.py` | `movie-retrieval prepare/train/evaluate/all/recommend` |

### 1.3 Tests (56 ตัว — ผ่านทั้งหมด)
- **Leakage tests**: temporal ordering ต่อ user, test user ไร้ train history ต้อง fail,
  poisoned split ต้อง raise `LeakageError`
- **Metric correctness**: เทียบค่า Recall/NDCG/Gini ที่คำนวณมือ
- **Security tests**: zip-slip archive ถูกปฏิเสธ, user_id/k validation ปฏิเสธ input แปลก
  (SQL injection string, path traversal, k เกิน bound)
- **Index tests**: save → load → Top-K ต้องเหมือนเดิมทุกตัว, unknown user → fallback flag
- **Integration test**: รัน prepare บน ml-100k จริงใน tmp dir (skip อัตโนมัติถ้าไม่มีไฟล์)

### 1.4 รัน pipeline จริง + ผลลัพธ์
- `movie-retrieval all --sensitivity` สำเร็จ (exit 0)
- Experiment matrix: R1 dim32 (val recall@10 = 0.0541) ชนะ R2 dim64 (0.0424) → เลือก R1
- ลองจูน lr 0.5/0.2, epochs 30 เพิ่มเติม → ไม่ดีขึ้น ยืนยัน config เดิม
- Final test: ดูตารางใน `README.md` / `artifacts/metrics.json`
- Serving: reload consistency ผ่าน, latency p50 = 0.17ms / p95 = 0.23ms
- CLI demo: user 42 → personalized recs, user 99999 → popularity fallback ✅

### 1.5 Notebook + Docs
- `notebooks/movielens_two_tower_retrieval.ipynb` — 19 sections ตาม blueprint บทที่ 5,
  **execute จริงครบทุก cell** (42 cells) พร้อม outputs/plots ฝังในไฟล์
- `README.md` — Problem → Approach → Result → Key decisions → Limitations (recruiter-first)
- `docs/model_card.md` — intended use, data terms, metrics, cold-start policy, ethics
- ไฟล์นี้ — development log

---

## 2. Design decisions สำคัญ (และเหตุผล)

| ตัดสินใจ | เหตุผล |
|---|---|
| **ไม่ใช้ TFRS** — เขียน retrieval loss เอง (~60 บรรทัด) | TFRS อยู่ใน maintenance mode + ไม่ compatible กับ Keras 3 (ต้องใช้ `TF_USE_LEGACY_KERAS` env hack ซึ่ง contaminate ทั้ง process) การ own โค้ดโปร่งใสกว่าและเป็น interview material ที่ดีกว่า |
| **Full-catalogue evaluation** แทน sampled | catalogue แค่ 1,682 items — matmul ตรงๆ ถูกกว่าและไม่มี sampling bias |
| **ดาวน์โหลด dataset ตอน runtime, ห้าม commit** | MovieLens terms ห้าม redistribute — บังคับผ่าน `.gitignore` + README |
| **Serving index เป็น pure TF ops** (StaticHashTable ไม่ใช่ Keras layer) | artifact ไม่ผูกกับ Keras version ใดๆ โหลดได้ทุก env ที่มี TF |
| **Export fail-hard ถ้า reload ไม่ตรง** | artifact ที่ reproduce ตัวเองไม่ได้ อันตรายกว่าไม่มี artifact |
| **แยก selection (val) ออกจาก final test** | test แตะครั้งเดียวหลังตัดสินใจครบ — วินัยตาม blueprint บทที่ 5/7 |

## 3. Bug ที่เจอระหว่างพัฒนา (บันทึกไว้เป็นบทเรียน)

1. **Flat loss จาก MEAN reduction** — ตอนแรกใช้ `reduce_mean` กับ in-batch softmax
   ทำให้ gradient เล็กกว่าที่ควร ~batch_size เท่า → Adagrad แทบไม่ขยับ, loss แบนสนิท
   แต่โครงสร้าง embedding ขยับเล็กน้อยจนดูเหมือน "เรียนรู้ได้" → เปลี่ยนเป็น **SUM
   reduction** ตาม semantics ของ TFRS Retrieval task แล้ว loss ลดปกติ
   (การ debug: ตรวจ gradient norm + apply หนึ่ง step แล้ววัด max abs change)
2. **`build_vocabs` assume ID เป็นตัวเลข** (`key=int`) — พังกับ synthetic test IDs
   → เปลี่ยนเป็น sort ด้วย `(len, str)` ซึ่ง deterministic เสมอและเทียบเท่า numeric
   order สำหรับ id ตัวเลข
3. **sed rename side effect** — rename `_val_recall` → `val_recall` ไปโดน key string
   `selected_val_recall@10` ใน report ด้วย → แก้กลับ (บทเรียน: ระวัง sed กับ substring)

## 4. Security measures

- ✅ HTTPS download + **pinned SHA-256** — ไฟล์ checksum ไม่ตรงถูกลบทิ้งทันที
- ✅ Zip-slip protection ตอน extract (ตรวจ path ทุก member + มี test ยืนยัน)
- ✅ Input validation ที่ serving API: `user_id` regex `[A-Za-z0-9_-]{1,64}`, `k ∈ [1,100]`
- ✅ `.gitignore` ครอบคลุม data/artifacts/secrets (`.env`, `*.pem`, `*.key`, `credentials*`)
- ✅ ruff `S` (bandit-style security rules) เปิดใน lint config
- ✅ ไม่มี secret/token/PII ใดๆ ใน repo — dataset เป็น de-identified research data

## 5. ผลลัพธ์สรุป (ตัวเลขจริงจาก final run)

- **Popularity ชนะที่ K=10** (recall 0.066 vs 0.058) — คาดการณ์ได้สำหรับ ID-only model
  บน temporal split ขนาดเล็ก และ blueprint สั่งให้รายงานตรงไปตรงมา
- **Two-tower ชนะที่ K=50** (0.231 vs 0.184) — regime ที่ retrieval stage ใช้จริง
- **Coverage 89.3% vs 5.4%** และ top-10%-popular share 6.3% vs 100% — จุดแข็งที่แท้จริง
- Slices: model แข็งสุดที่ long-tail items (2.6× head) และ low-activity users
- Sensitivity rating≥4: ข้อสรุปไม่เปลี่ยน (0.061 / 0.219)

## 6. งานต่อ (ถ้าจะพัฒนาต่อ)

1. Metadata-enhanced movie tower (title/genres) + item cold-start evaluation
2. Ranking MLP หลัง retrieval
3. ScaNN/Faiss index + recall-latency benchmark
4. Log-Q correction สำหรับ in-batch sampling bias
5. CI (GitHub Actions): pytest + ruff ทุก push
6. MLflow tracking แทน experiments.json เมื่อ experiment เยอะขึ้น

## 7. วิธีรันซ้ำจากศูนย์

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
movie-retrieval all --sensitivity   # ~2 นาทีบน CPU
movie-retrieval recommend --user-id 42 --k 10
pytest                              # 56 tests
```
