# Project Status — เทียบกับ Blueprint (docx)

> อัปเดต: 18 กรกฎาคม 2026 · เอกสารอ้างอิง: `05_movie_recommender_ai_engineering_plan.docx`
> สรุปสั้น: **โครงหลักเสร็จครบ** (โค้ด + tests + notebook + docs + ผลลัพธ์จริง) —
> เหลืองาน "เผยแพร่" (git commit → GitHub) และ "ยืนยันบน Colab จริง"

---

## 1. เทียบความคืบหน้าตาม docx (14 บท)

| บท | หัวข้อ | สถานะ | หลักฐาน / หมายเหตุ |
|---|---|---|---|
| 1 | Executive Summary | ✅ เสร็จ | สรุปใน `README.md` + notebook §01 |
| 2 | Business Problem & Framing | ✅ เสร็จ | notebook §01, README "Problem" |
| 3 | Dataset, Data Contract, Leakage | ✅ เสร็จ | `data.py` (SHA-256 + contract), `splits.py` (audit) |
| 4 | Colab Environment & Reproducibility | 🟡 บางส่วน | seed/version/lock file ครบ, reproduce ซ้ำได้ — **แต่ยังไม่ได้รันบน Colab จริง** |
| 5 | Detailed Notebook Blueprint (19 sections) | ✅ เสร็จ | `notebooks/movielens_two_tower_retrieval.ipynb` execute ครบทุก cell |
| 6 | Implementation Plan: Data to Model | ✅ เสร็จ | `model.py`, `baseline.py`, experiment matrix R0/R1/R2/R3 |
| 7 | Evaluation, Testing, Error Analysis | ✅ เสร็จ | `evaluate.py` — Recall/NDCG/coverage/bias + slices, notebook §12–14 |
| 8 | Execution Plan (5 days) | ✅ เทียบเท่า | ทำครบเนื้อหาทั้ง 5 วันในรอบเดียว (ดู mapping ด้านล่าง) |
| 9 | Packaging, Inference, Deployment | ✅ เสร็จ | `index.py` (SavedModel + fallback), inference contract, monitoring plan |
| 10 | GitHub Portfolio Packaging | 🟡 บางส่วน | ไฟล์ครบตาม structure — **แต่ยังไม่ commit / push ขึ้น GitHub** |
| 11 | Technical Interview Preparation | 🟡 บางส่วน | เนื้อหา Q&A อยู่ใน notebook/README — **การซ้อมตอบเป็นงานของผู้ใช้เอง** |
| 12 | Trade-offs, Limitations, Stretch Goals | ✅ เสร็จ (core) | ตาราง trade-off ใน README/model_card — **stretch goals ยังไม่ทำ (optional)** |
| 13 | Definition of Done | 🟡 เกือบครบ | ดู checklist ด้านล่าง — ติดแค่ข้อ Colab |
| 14 | Web References | ✅ เสร็จ | notebook "References" section |

**สรุป: 10/14 บทเสร็จสมบูรณ์, 4 บทเหลือส่วนที่เป็น "เผยแพร่/ยืนยัน/optional"**

---

## 2. Definition of Done checklist (docx บทที่ 13)

- [x] Dataset terms + policy ไม่ commit raw files — ระบุใน `.gitignore` + README + LICENSE
- [x] positive interaction rule + temporal leave-last-k split ชัดเจน — `config.py`, `splits.py`
- [x] popularity baseline พร้อม seen-item filtering — `baseline.py`
- [x] two-tower model + candidate dataset — `model.py`
- [x] Recall@K / NDCG / coverage / popularity bias + slices — `evaluate.py`, `metrics.json`
- [x] index/vocab/model reload + unknown-user fallback ผ่าน — `index.py` + tests
- [x] recommendation examples + qualitative audit — notebook §15
- [x] README แยก retrieval/ranking/online validation ชัดเจน — `README.md`
- [ ] **Colab Run all ได้ภายใน timebox** — ⚠️ execute ผ่านบน local เท่านั้น ยังไม่ยืนยันบน Colab

**Final run checklist:**
- [x] Restart แล้ว Run all สำเร็จ (local — clean state rerun ได้เลขเป๊ะ)
- [x] split/schema เหมือนเดิมตาม seed/version
- [x] load model artifact แล้ว prediction สอดคล้อง (reload_consistent = True)
- [x] ไม่มี secret/token/PII/dataset ใน repo
- [x] README/notebook/model card ใช้ตัวเลขจาก final run เดียวกัน

---

## 3. Mapping แผน 5 วัน (docx บทที่ 8) → สิ่งที่ทำ

| วัน | เป้าหมาย docx | ทำแล้วที่ |
|---|---|---|
| Day 1 | retrieval objective + temporal split | `splits.py` + notebook §06–07 + tests |
| Day 2 | popularity baseline + retrieval pipeline | `baseline.py` + `model.py` + notebook §09–10 |
| Day 3 | train embeddings + เทียบ config | experiment matrix R1/R2 + notebook §11 |
| Day 4 | bias/slices + serving index | `evaluate.py` slices + `index.py` + notebook §13,16 |
| Day 5 | package + interview story | README + model_card + notebook §16–19 |

---

## 4. ขั้นตอนต่อไป (เรียงตามลำดับที่ควรทำ)

### 🔴 ต้องทำ (ปิด gap ตาม Definition of Done)
1. **Git commit ครั้งแรก** — repo `git init` แล้วแต่ยังไม่มี commit เลย
   → initial commit + สร้าง GitHub repo + push (นี่คือบทที่ 10 โดยตรง)
2. **ยืนยันบน Google Colab** — docx ให้ Colab เป็น primary environment
   → เปิด notebook บน Colab, `%pip install -e .` (หรือ clone repo), Restart → Run all
   ให้ผ่านจริง แล้วแคป cell outputs (ข้อสุดท้ายของ Definition of Done)

### 🟡 ควรทำ (ยกระดับความเป็นมืออาชีพ — เกิน docx)
3. **CI (GitHub Actions)** — รัน `pytest` + `ruff` อัตโนมัติทุก push
4. **requirements.txt** — docx บทที่ 10 ระบุชื่อไฟล์นี้ (ตอนนี้ใช้ `pyproject.toml` +
   `requirements-lock.txt` ซึ่งครอบคลุมกว่า) — เพิ่มให้ตรง structure ถ้าต้องการ

### 🟢 Optional (Stretch goals — docx บทที่ 12, ทำเมื่อมีเวลา)
5. Metadata-enhanced movie tower (title/genres) + item cold-start evaluation
6. Ranking MLP หลัง retrieval (rating + context features)
7. ScaNN/Faiss index + benchmark recall-vs-latency เทียบ BruteForce
8. Diversity/novelty reranking + business constraints
9. Log-Q correction สำหรับ in-batch sampling bias

---

## 5. ส่วนที่ทำต่างจาก docx อย่างตั้งใจ (deliberate deviations)

| docx เขียน | ที่ทำจริง | เหตุผล |
|---|---|---|
| ใช้ TensorFlow Recommenders (TFRS) | เขียน in-batch sampled softmax เอง (pure TF/Keras 3) | TFRS อยู่ใน maintenance mode + ไม่ compat Keras 3 (ต้องใช้ legacy-Keras hack) — ดู `DEVELOPMENT_LOG.md` §2 |
| `tests/test_recommendation_index.py` | `tests/test_index.py` (+ อีก 7 ไฟล์) | ครอบคลุมกว่า — 56 tests รวม leakage/security/metric |
| `requirements.txt` | `pyproject.toml` + `requirements-lock.txt` | packaging มาตรฐานสมัยใหม่ + lock ที่ reproduce ได้แน่นอน |
| Colab notebook อย่างเดียว | notebook + installable package (`src/`) | โค้ดหลัก testable/reusable, notebook เป็น narrative — ดีต่อ portfolio |

> ทุก deviation ทำให้โปรเจค **ดีกว่าหรือเทียบเท่า** blueprint และบันทึกเหตุผลไว้ครบ

---

## 6. ผลลัพธ์ล่าสุด (จาก final run — reproduce ได้)

| metric @ K | Popularity (R0) | Two-tower (R1, dim32) |
|---|---|---|
| Recall@10 | **0.0657** | 0.0583 |
| Recall@50 | 0.1835 | **0.2312** |
| NDCG@10 | **0.0331** | 0.0266 |
| Coverage@10 | 0.054 | **0.893** |
| Gini exposure | 0.986 | **0.477** |

Serving latency p95 ≈ 0.2ms · reload consistent ✅ · OOV test items 0.2% ·
sensitivity (rating≥4): ข้อสรุปไม่เปลี่ยน
