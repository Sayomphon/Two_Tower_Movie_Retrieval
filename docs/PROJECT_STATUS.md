# Project Status — เทียบกับ Blueprint (docx)

> อัปเดต: 2 สิงหาคม 2026 · เอกสารอ้างอิง: `05_movie_recommender_ai_engineering_plan.docx`
> สรุปสั้น: **โค้ด + 62 tests + notebook (executed) + docs + CI ครบ และ push ขึ้น GitHub แล้ว**
> ([Sayomphon/Two_Tower_Movie_Retrieval](https://github.com/Sayomphon/Two_Tower_Movie_Retrieval))
> เหลือค้างข้อเดียวคือ **กด Run all บน Colab จริง** เพื่อปิด Definition of Done ข้อสุดท้าย
> — bootstrap cell พร้อมแล้วและผ่านการจำลอง bare runtime บนเครื่อง แต่ยังไม่ได้รันบน Colab เอง

---

## 1. เทียบความคืบหน้าตาม docx (14 บท)

| บท | หัวข้อ | สถานะ | หลักฐาน / หมายเหตุ |
|---|---|---|---|
| 1 | Executive Summary | ✅ เสร็จ | `README.md` + notebook §01 |
| 2 | Business Problem & Framing | ✅ เสร็จ | notebook §01, README "Problem" |
| 3 | Dataset, Data Contract, Leakage | ✅ เสร็จ | `data.py` (SHA-256 + contract), `splits.py` (audit) |
| 4 | Colab Environment & Reproducibility | 🟡 เกือบครบ | seed/version/lock ครบ · bootstrap cell (notebook §02) ติดตั้ง package เองบน runtime เปล่า · `requires-python` เปิดถึง `<3.14` ให้ Colab runtime ใหม่ติดตั้งได้ · **ยังไม่ได้กด Run all บน Colab จริง** |
| 5 | Detailed Notebook Blueprint (19 sections) | ✅ เสร็จ | notebook 47 cells execute ครบ พร้อม outputs/plots ฝังในไฟล์ |
| 6 | Implementation Plan: Data to Model | ✅ เสร็จ | `model.py`, `baseline.py`, `experiments.json` มีครบ R0–R4 |
| 7 | Evaluation, Testing, Error Analysis | ✅ เสร็จ | `evaluate.py` — Recall/NDCG/coverage/bias + slices · 62 tests |
| 8 | Execution Plan (5 days) | ✅ เทียบเท่า | ทำครบเนื้อหาทั้ง 5 วัน (ดู mapping ข้อ 3) |
| 9 | Packaging, Inference, Deployment | ✅ เสร็จ | `index.py` (SavedModel + fallback), inference contract, monitoring plan (notebook §18) |
| 10 | GitHub Portfolio Packaging | ✅ เสร็จ | push ขึ้น GitHub แล้ว · `requirements.txt` ครบตาม structure · visual evidence 6/6 · CI badge |
| 11 | Technical Interview Preparation | ✅ เสร็จ | `docs/interview_prep.md` — pitch 90 วินาที + 7 คำถามออกแบบ + คำถามยากจากผลจริง + deep-dive |
| 12 | Trade-offs, Limitations, Stretch Goals | ✅ เสร็จ (core) | ตาราง trade-off ใน README/model_card — **stretch goals ยังไม่ทำ (optional ตาม docx)** |
| 13 | Definition of Done | 🟡 8/9 | ค้างข้อ Colab ข้อเดียว (ดู checklist ข้อ 2) |
| 14 | Web References | ✅ เสร็จ | notebook "References" |

**สรุป: 12/14 บทเสร็จสมบูรณ์ · บท 4 กับ 13 ค้างเรื่องเดียวกันคือการยืนยันบน Colab จริง**

---

## 2. Definition of Done checklist (docx บทที่ 13)

- [x] Dataset terms + policy ไม่ commit raw files — `.gitignore` + README + LICENSE
- [x] positive interaction rule + temporal leave-last-k split ชัดเจน — `config.py`, `splits.py`
- [x] popularity baseline พร้อม seen-item filtering — `baseline.py`
- [x] two-tower model + candidate dataset — `model.py`
- [x] Recall@K / NDCG / coverage / popularity bias + slices — `evaluate.py`, `metrics.json`
- [x] index/vocab/model reload + unknown-user fallback ผ่าน — `index.py` + tests
- [x] recommendation examples + qualitative audit — notebook §15
- [x] README แยก retrieval/ranking/online validation ชัดเจน — `README.md`
- [ ] **Colab Run all ได้ภายใน timebox** — เหลือข้อเดียว (ดูข้อ 5)

**Final run checklist:**
- [x] Restart แล้ว Run all สำเร็จ (local — clean state rerun ได้เลขเป๊ะ)
- [x] split/schema เหมือนเดิมตาม seed/version
- [x] load model artifact แล้ว prediction สอดคล้อง (`reload_consistent = true`)
- [x] ไม่มี secret/token/PII/dataset ใน repo
- [x] README / notebook / model card / interview prep ใช้ตัวเลขจาก final run เดียวกัน

---

## 3. Mapping แผน 5 วัน (docx บทที่ 8) → สิ่งที่ทำ

| วัน | เป้าหมาย docx | ทำแล้วที่ |
|---|---|---|
| Day 1 | retrieval objective + temporal split | `splits.py` + notebook §06–07 + tests |
| Day 2 | popularity baseline + retrieval pipeline | `baseline.py` + `model.py` + notebook §09–10 |
| Day 3 | train embeddings + เทียบ config | experiment matrix R1/R2 + notebook §11 |
| Day 4 | bias/slices + serving index | `evaluate.py` slices + `index.py` + notebook §13,16 |
| Day 5 | package + interview story | README + model_card + interview_prep + notebook §16–19 |

---

## 4. งานที่ปิดไปหลังรอบแรก (3 รอบทำงาน)

| งาน | สถานะ | หลักฐาน |
|---|---|---|
| `requirements.txt` ตาม structure ของ blueprint | ✅ | ไฟล์ที่ root + ตารางเทียบ 3 ไฟล์ใน README |
| Experiment tracking ครบ field + run R0–R4 | ✅ | `experiments.json`: recall/ndcg/coverage/train_seconds/n_params + `candidate_configuration` + `R4-serving` |
| Comparison table มี metric/latency/size/explainability | ✅ | notebook §11 |
| Test: Top-K ห้ามมีหนังซ้ำ (3 ชั้น) | ✅ | `test_evaluate.py`, `test_baseline.py`, `test_index.py` — รวมเป็น 62 tests |
| Visual evidence 6/6 | ✅ | long-tail §05 · split timeline §07 · comparison §11 · coverage+Lorenz §12 · embedding neighbours §15 · architecture (Mermaid) §18 + README |
| Colab bootstrap cell | ✅ | notebook §02 cell แรก — clone + install เมื่อ import ไม่เจอ, no-op เมื่อรัน local |
| `docs/interview_prep.md` | ✅ | pitch + 7 คำถาม + คำถามยากจากผลจริง + deep-dive |
| CI (GitHub Actions) | ✅ | `.github/workflows/ci.yml` — ruff + pytest บน Python 3.11/3.12, actions pin ด้วย commit SHA |

**บันทึกการยืนยัน bootstrap cell — จำลอง bare runtime บนเครื่อง (ไม่ใช่ Colab จริง):**

| สิ่งที่ทดสอบ | วิธี | ผล |
|---|---|---|
| Run all จาก runtime เปล่า โดยไม่แก้ cell | venv ใหม่ (Python 3.11.15) ที่มี TF/pandas/numpy แต่ **ไม่มี** `movie_retrieval` · working dir ว่างที่ไม่มี `pyproject.toml` (จำลอง `/content`) · `jupyter execute` ทั้งไฟล์ | ✅ 21/21 code cells, 0 errors, 26.8 วินาที |
| ตัวเลขที่ได้จาก runtime เปล่า | เทียบ output §12 กับ `artifacts/metrics.json` | ✅ recall@10 = 0.0657 / 0.0583 ตรงเป๊ะ |
| bootstrap เป็น no-op เมื่อรัน local | รันในเครื่องที่ติดตั้ง package แล้ว | ✅ พิมพ์ "already importable — bootstrap skipped" |
| `Paths.default()` นอก checkout | ตรวจค่า `project root` ที่พิมพ์ออกมา | ✅ ชี้ไป working directory ปัจจุบันตามที่ออกแบบ |

> **สิ่งที่การทดสอบนี้ยังไม่ครอบคลุม** (จึงยังไม่ติ๊ก DoD): Python/TensorFlow version จริงของ Colab,
> ความเร็วเน็ตของ Colab runtime และพฤติกรรมเมื่อ TF ที่ติดมากับ Colab อยู่นอกช่วง `>=2.16,<2.21`
>
> **บั๊กที่การทดสอบนี้จับได้:** ร่างแรกใช้ `pip install -e` แล้ว cell ถัดไป `ModuleNotFoundError`
> เพราะ editable install ชี้ path ผ่าน `__editable__*.pth` ซึ่ง Python อ่านเฉพาะตอนเริ่ม interpreter
> → `importlib.invalidate_caches()` ช่วยไม่ได้ · เปลี่ยนเป็น install แบบปกติจึงผ่าน
> (ดู `DEVELOPMENT_LOG.md` §3 ข้อ 6) — ถ้าไม่ทดสอบใน runtime เปล่าจริง จะไม่มีทางเจอ
> เพราะเครื่องที่มี package อยู่แล้วเข้า branch no-op เสมอ

---

## 5. เหลืออะไร

### 🔴 ปิด Definition of Done ข้อสุดท้าย (ต้องทำโดยเจ้าของ repo — ใช้บัญชี Google)

1. เปิด notebook บน Colab: [Open in Colab](https://colab.research.google.com/github/Sayomphon/Two_Tower_Movie_Retrieval/blob/main/notebooks/movielens_two_tower_retrieval.ipynb)
2. Runtime → Restart runtime → **Run all** (ห้ามแก้ cell ด้วยมือ)
3. คาดว่าใช้เวลา ~5–10 นาที (install ~2–3 น. + train 2 runs + eval + export)
4. กรอกผลกลับมาที่ตารางนี้แล้วติ๊ก checklist ข้อ 9 ในข้อ 2

| ช่องที่ต้องกรอกหลังรันจริง | ค่า |
|---|---|
| วันที่ทดสอบ | _(รอกรอก)_ |
| Colab Python / TensorFlow version | _(รอกรอก)_ |
| เวลา end-to-end | _(รอกรอก)_ |
| ผล Run all | _(รอกรอก)_ |

> ถ้าติดปัญหา: อาการที่เป็นไปได้คือ Colab มี TensorFlow เวอร์ชันนอกช่วง `>=2.16,<2.21`
> ทำให้ pip ถอน/ลงใหม่และใช้เวลานาน — แก้โดยขยาย upper bound ใน `pyproject.toml`
> แล้วรัน `pytest` ยืนยันซ้ำก่อน commit **ห้าม hardcode token ใด ๆ ใน notebook เด็ดขาด**

### 🟢 Optional (Stretch goals — docx บทที่ 12, ทำเมื่อมีเวลา)

เรียงตาม impact ต่อ portfolio · docx ระบุ scope guardrail ว่าให้เลือก **ข้อเดียวทำให้จบ**
พร้อม test + เอกสาร ดีกว่าทำครึ่ง ๆ หลายข้อ

1. Metadata-enhanced movie tower (title/genres) + item cold-start evaluation — แก้ข้อจำกัดใหญ่สุดที่ประกาศไว้เอง
2. ScaNN/Faiss index + benchmark recall-vs-latency เทียบ BruteForce
3. Log-Q correction สำหรับ in-batch sampling bias
4. Ranking MLP หลัง retrieval (rating + context features)
5. Diversity/novelty reranking + business constraints

---

## 6. ส่วนที่ทำต่างจาก docx อย่างตั้งใจ (deliberate deviations)

| docx เขียน | ที่ทำจริง | เหตุผล |
|---|---|---|
| ใช้ TensorFlow Recommenders (TFRS) | เขียน in-batch sampled softmax เอง (pure TF/Keras 3) | TFRS อยู่ใน maintenance mode + ไม่ compat Keras 3 (ต้องใช้ legacy-Keras hack) — ดู `DEVELOPMENT_LOG.md` §2 |
| `tests/test_recommendation_index.py` | `tests/test_index.py` (+ อีก 8 ไฟล์) | ครอบคลุมกว่า — 62 tests รวม leakage/security/metric/no-duplicate |
| `requirements.txt` | มีครบทั้ง 3 ไฟล์: `pyproject.toml` (ทางหลัก) + `requirements.txt` (ranges ตาม structure) + `requirements-lock.txt` (pin เป๊ะ) | packaging มาตรฐานสมัยใหม่ แต่ไม่ทิ้งไฟล์ที่ผู้รีวิว/Colab คาดหวัง — README อธิบายว่าแต่ละไฟล์ใช้เมื่อไร |
| Colab notebook อย่างเดียว | notebook + installable package (`src/`) | โค้ดหลัก testable/reusable, notebook เป็น narrative — ดีต่อ portfolio |
| ไม่ได้ระบุเรื่อง CI | เพิ่ม GitHub Actions (ruff + pytest, matrix 3.11/3.12) | กัน commit ที่ทำ test พังหลุดเข้า main + สื่อ engineering maturity ตั้งแต่หน้าแรก |
| `requires-python` (ไม่ได้ระบุ) | เปิดจาก `<3.13` เป็น `<3.14` | ให้ Colab runtime ที่ใช้ Python 3.13 ติดตั้งได้ (TF 2.20 มี cp313 wheels) · CI ทดสอบจริงที่ 3.11 และ 3.12 |

> ทุก deviation ทำให้โปรเจค **ดีกว่าหรือเทียบเท่า** blueprint และบันทึกเหตุผลไว้ครบ

---

## 7. ผลลัพธ์ล่าสุด (จาก final run — reproduce ได้)

| metric @ K | Popularity (R0) | Two-tower (R1, dim32) |
|---|---|---|
| Recall@10 | **0.0657** | 0.0583 |
| Recall@50 | 0.1835 | **0.2312** |
| NDCG@10 | **0.0331** | 0.0266 |
| Coverage@10 | 0.054 | **0.893** |
| Gini exposure | 0.986 | **0.477** |

Serving latency p95 ≈ 1.10 ms (laptop CPU, แกว่งตามโหลดเครื่อง) · index build 0.19 s ·
reload consistent ✅ · OOV test items 0.21% · sensitivity (rating ≥ 4): ข้อสรุปไม่เปลี่ยน

> `index_version` (`catalog-<วันที่รัน>`) และ `generated_at` เปลี่ยนตามวันที่รันเป็นเรื่องปกติ
> ตัวเลข metric ที่เหลือต้องเท่าเดิมเป๊ะเพราะ seed คงที่ (42) และ split เป็น deterministic
