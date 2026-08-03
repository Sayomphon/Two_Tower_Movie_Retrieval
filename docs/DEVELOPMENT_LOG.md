# Development Log — Two-Tower Movie Retrieval

> **In English** — The build log for this project, kept as it was written rather than
> tidied afterwards. It covers, in order: what was built (§1), the design decisions and why
> each was taken — including choosing *not* to use TFRS (§2), **the bugs hit along the way
> and what they taught** (§3 — the most useful section: a silently-flat training loss caused
> by MEAN loss reduction, an overclaim about temporal splits that the data itself refuted,
> and a Colab bootstrap that failed because editable installs are invisible to an already
> running interpreter), the security measures (§4), headline results (§5), what would come
> next (§6), and how to reproduce everything from scratch (§7). *Body text is Thai; code,
> metric names, and commands are language-neutral.*

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
| `demo.py` | Gradio UI บน `RetrievalService` เดิม (optional dependency — `import gradio` อยู่ข้างใน `build_demo()` เท่านั้น) |

### 1.3 Tests (101 ตัว — ผ่านทั้งหมด)
- **Leakage tests**: temporal ordering ต่อ user, test user ไร้ train history ต้อง fail,
  poisoned split ต้อง raise `LeakageError`
- **Metric correctness**: เทียบค่า Recall/NDCG/Gini ที่คำนวณมือ
- **Security tests**: zip-slip archive ถูกปฏิเสธ, user_id/k validation ปฏิเสธ input แปลก
  (SQL injection string, path traversal, k เกิน bound)
- **Index tests**: save → load → Top-K ต้องเหมือนเดิมทุกตัว, unknown user → fallback flag
- **No-duplicate tests**: Top-K ต้องไม่มีหนังซ้ำทั้ง 3 ชั้น (evaluate / baseline / serving
  ซึ่งชั้น serving เสี่ยงสุดเพราะ over-fetch `k + len(seen)` ก่อน filter)
- **Experiment-log tests**: เขียน run ซ้ำลง `experiments.json` ต้องแทนที่ ไม่ใช่สะสมแถวซ้ำ
- **CLI contract tests**: argv → เรียก pipeline stage ไหนด้วย argument อะไร (monkeypatch
  pipeline ทั้งหมด ไม่แตะ network/artifacts) — จุดที่พลาดง่ายสุดคือ `--include-seen`
  ที่ถูกกลับด้านเป็น `exclude_seen` ก่อนส่งต่อ
- **Integration test**: รัน prepare บน ml-100k จริงใน tmp dir (skip อัตโนมัติถ้าไม่มีไฟล์)
- **Demo UI tests**: สิ่งที่ผู้ใช้เห็นบนหน้าจอ (ผลลัพธ์ / status badge / ประวัติ / stat chips)
  ทดสอบได้โดยไม่ต้องติดตั้ง gradio เพราะ logic แยกจาก UI wiring — คุมสองเรื่องหลักคือ
  input แปลก ๆ ต้องกลายเป็นข้อความบอกผู้ใช้ไม่ใช่ exception และ **ทุกค่าที่มาจากข้อมูล
  ต้องถูก escape** (ชื่อหนังที่เป็น `<script>` ต้องออกมาเป็น text ไม่ใช่ tag)

### 1.4 รัน pipeline จริง + ผลลัพธ์
- `movie-retrieval all --sensitivity` สำเร็จ (exit 0)
- Experiment matrix: R1 dim32 (val recall@10 = 0.0541) ชนะ R2 dim64 (0.0424) → เลือก R1
- ลองจูน lr 0.5/0.2, epochs 30 เพิ่มเติม → ไม่ดีขึ้น ยืนยัน config เดิม
- Final test: ดูตารางใน `README.md` / `artifacts/metrics.json`
- Serving: reload consistency ผ่าน, index build 0.19s, latency p50 = 0.56ms / p95 = 1.10ms
  (laptop CPU — ค่านี้แกว่งตามโหลดเครื่องราว 0.3–1.1ms ระหว่างรอบรัน ต่างจาก metric อื่นที่ deterministic)
- CLI demo: user 42 → personalized recs, user 99999 → popularity fallback ✅

### 1.5 Notebook + Docs
- `notebooks/movielens_two_tower_retrieval.ipynb` — 19 sections ตาม blueprint บทที่ 5,
  **execute จริงครบทุก cell** (46 cells) พร้อม outputs/plots ฝังในไฟล์
- **Visual evidence ครบ 6/6** ตาม blueprint บทที่ 10: long-tail EDA (§05), temporal split
  timeline + gap audit ของทั้ง 943 users (§07), comparison table (§11), final metrics table
  คู่กับ coverage/Lorenz chart (§12), embedding neighbours (§15) และ production architecture
  เป็น Mermaid (§18 + หน้าแรกของ README ให้ recruiter อ่านได้โดยไม่ต้องเปิด notebook)
- `README.md` — Problem → Approach → Result → Key decisions → Limitations (recruiter-first)
- `docs/model_card.md` — intended use, data terms, metrics, cold-start policy, ethics
- `docs/interview_prep.md` — pitch 90 วินาที + 7 คำถามออกแบบ + คำถามยากจากผลจริง + deep-dive
- ไฟล์นี้ — development log

### 1.6 Colab bootstrap + CI
- notebook §02 มี bootstrap cell: ถ้า `import movie_retrieval` ไม่เจอ → clone repo + pip install
  แล้ว `importlib.invalidate_caches()` ให้ import ได้ในรอบเดียวโดยไม่ต้อง restart runtime
  (ดูบทเรียนข้อ 3.6 — ทำไมต้อง install แบบปกติ ไม่ใช่ `-e`)
- `requires-python` เปิดเป็น `>=3.11,<3.14` ให้ Colab runtime ใหม่ติดตั้งได้ (TF 2.20 มี cp313 wheels)
- `.github/workflows/ci.yml` — ruff + pytest บน Python 3.11/3.12 ทุก push/PR,
  `permissions: contents: read`, actions pin ด้วย commit SHA, `persist-credentials: false`
- **Coverage gate `--cov-fail-under=65`** — ตั้งจากค่าที่ CI เห็นจริง (68%) ลบ buffer
  ไม่ใช่เลขที่ตั้งลอย ๆ · เป็น regression gate ไม่ใช่เป้าหมาย · ค่าบน CI ต่ำกว่าในเครื่อง
  เพราะ integration test skip เมื่อไม่มี dataset ทำให้ `pipeline.py` ไม่ถูกไล่
  (ตอนตั้ง gate เจอว่า `cli.py` coverage 0% → เขียน CLI contract tests เพิ่ม 9 ตัว
  ดัน cli.py เป็น 95% และ total 62% → 68%)
- **ยืนยันบน Colab จริงแล้ว** (2 ส.ค. 2026, Python 3.12.13 / TF 2.20.0 / GPU T4): Run all
  จาก runtime เปล่าผ่าน 21/21 cells, 0 errors, ~5 นาที end-to-end, ไม่แก้ cell ใด ๆ (diff source ทั้ง 47 cells
  กับไฟล์ใน repo แล้วเหมือนกันทุกตัวอักษร) และได้ metric ตรงกับที่รันบน macOS arm64/Python 3.11
  ทุกหลัก — หลักฐานอยู่ที่ `notebooks/movielens_two_tower_retrieval_Colab_Ran.ipynb`
  ส่วน latency ต่างกันตามเครื่อง (p95 2.94 ms vs 1.10 ms) ซึ่งเอกสารระบุไว้แต่แรกว่าไม่ deterministic

### 1.7 Interactive demo (Gradio)
- `app.py` + `movie_retrieval/demo.py` — พิมพ์ user id แล้วเห็น Top-K จาก **serving index จริง**
  (`RetrievalService` ตัวเดียวกับ CLI, artifacts ชุดเดียวกัน) พร้อมปุ่มตัวอย่างที่คลิกเดียวเห็นผล
  และเปิดหน้ามาก็มีผลของ user 42 แสดงอยู่แล้ว
- แยกชั้นชัดเจน: ฟังก์ชันที่แปลง response → ตาราง/สถานะ/ประวัติ เป็น pure function ทดสอบได้
  ส่วน `import gradio` อยู่ข้างใน `build_demo()`/`main()` — คนที่ใช้แค่ library/CLI ไม่ต้องโหลด UI stack
- Security: ใช้ validation ของ service เดิม (regex user_id, `k ∈ [1,100]`) แล้วแปลง `ValueError`
  เป็นข้อความบนหน้าจอ · `analytics_enabled=False` (ไม่ส่ง telemetry) · `api_open=False` ·
  จำกัดคิว 32 · **ไม่มี `share=True`** และไม่มีช่องอัปโหลดไฟล์ใด ๆ
- ผลลัพธ์ render เป็น HTML ที่เขียนเอง ไม่ใช่ `gr.Dataframe` — Dataframe ใช้ฟอนต์ monospace
  อ่านชื่อหนังยาว ๆ ลำบาก และลาก affordance ที่ demo อ่านอย่างเดียวไม่ควรมีมาด้วย (import CSV,
  edit cell) · แต่ละแถวมี **score bar** ที่ normalize ด้วย min/max ภายใน response เดียว
  (คะแนน dot product กับ popularity count เทียบกันตรง ๆ ไม่ได้) — บอก "ห่างกันแค่ไหนในลิสต์นี้"
- hero มี **stat chip อ่านจาก `metrics.json` จริง** (Recall@50 / Coverage@10 / p95 / ขนาด catalogue)
  อ่านครั้งเดียวตอน start ไม่แตะ disk ต่อ request · ถ้าไฟล์หายก็แค่ไม่มี chip ไม่ใช่หน้าพัง
- สีทุกค่าอ้าง CSS variable ของ Gradio (`--body-text-color`, `--border-color-primary`, …)
  → dark mode ใช้ได้ทันทีโดยไม่ต้องเขียน CSS สองชุด · ตรวจแล้วทั้ง light และ dark
- **แลกความปลอดภัยกลับมาด้วย escape:** `gr.HTML` ไม่ sanitize ให้ ดังนั้นชื่อหนัง / version /
  ข้อความ error ผ่าน `html.escape()` ทุกจุด, ความกว้าง bar ถูก format เป็นตัวเลขก่อนเข้า `style`
  และ `tone` ของ badge มาจากชุดค่าคงที่ในโค้ดเท่านั้น — มี test คุมทั้งสามข้อ

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
4. **เกือบ over-claim เรื่อง temporal split** — ร่างแรกของ split diagram (§07) จะเขียนว่า
   "val/test อยู่หลัง train เสมอ" แต่ตรวจข้อมูลจริงก่อนวาดพบว่า **422/943 users มี val
   timestamp เท่ากับ interaction สุดท้ายของ train เป๊ะ** (ML-100K ละเอียดระดับวินาที คนเดียว
   เรตหลายเรื่องรวดเดียว) ซึ่งตรงกับที่ `audit_no_leakage` ยืนยันแค่ `train ≤ val ≤ test`
   และตัด tie ด้วย `movie_id` → เปลี่ยนภาพเป็นสอง panel (timeline + สัดส่วน gap ทั้ง
   population) และเขียนคำอธิบายตามความจริง (บทเรียน: ตรวจข้อมูลก่อนเขียน caption)
5. **ตัวเลขในเอกสารปัดผิด** — README/DEVELOPMENT_LOG เขียน recall@50 ของ popularity เป็น
   0.184 ทั้งที่ค่าจริงคือ 0.18346 (ปัด 3 ตำแหน่ง = 0.183) → เจอเพราะเขียน gate
   เทียบทุกตัวเลขในเอกสารกับ `artifacts/metrics.json` แบบอัตโนมัติ ไม่ใช่ไล่อ่านเอง
6. **`pip install -e` ทำให้ Colab bootstrap พังแบบเงียบ ๆ** — ร่างแรกของ bootstrap cell ใช้
   editable install แล้วเรียก `importlib.invalidate_caches()` ตามที่คิดว่าพอ พอทดสอบด้วยการ
   จำลอง runtime เปล่า (venv ใหม่ + working dir ที่ไม่มี `pyproject.toml`) พบว่า cell ถัดไปยัง
   `ModuleNotFoundError` เพราะ editable install ชี้ path ผ่านไฟล์ `__editable__*.pth` ซึ่ง Python
   ประมวลผลเฉพาะตอน start interpreter (`site.addsitedir`) → `invalidate_caches()` ที่ล้างแค่
   FileFinder cache จึงช่วยไม่ได้ เปลี่ยนเป็น install แบบปกติ (ลงใน `site-packages` ที่อยู่บน
   `sys.path` อยู่แล้ว) จึงผ่าน (บทเรียน: gate ที่เขียนว่า "Run all ต้องผ่านโดยไม่แก้ cell" ต้อง
   ทดสอบใน runtime เปล่าจริง ๆ — รันในเครื่องที่ติดตั้ง package ไว้แล้วจะเข้า branch no-op เสมอ
   และไม่พิสูจน์อะไรเลย)

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
- **Two-tower ชนะที่ K=50** (0.231 vs 0.183) — regime ที่ retrieval stage ใช้จริง
- **Coverage 89.3% vs 5.4%** และ top-10%-popular share 6.3% vs 100% — จุดแข็งที่แท้จริง
- Slices: model แข็งสุดที่ long-tail items (2.6× head) และ low-activity users
- Sensitivity rating≥4: ข้อสรุปไม่เปลี่ยน (0.061 / 0.219)

## 6. งานต่อ (ถ้าจะพัฒนาต่อ)

1. Metadata-enhanced movie tower (title/genres) + item cold-start evaluation
2. Ranking MLP หลัง retrieval
3. ScaNN/Faiss index + recall-latency benchmark
4. Log-Q correction สำหรับ in-batch sampling bias
5. MLflow tracking แทน experiments.json เมื่อ experiment เยอะขึ้น

## 7. วิธีรันซ้ำจากศูนย์

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
movie-retrieval all --sensitivity   # ~2 นาทีบน CPU
movie-retrieval recommend --user-id 42 --k 10
pytest                              # 101 tests
```
