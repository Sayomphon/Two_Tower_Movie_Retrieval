# Interview Prep — Two-Tower Movie Retrieval

> **In English** — Interview preparation notes for this project. §0 is a figures cheat
> sheet, §1 a 90-second pitch (Thai and English), §2 the seven standard retrieval design
> questions — retrieval vs ranking, why not RMSE, why temporal splits, where negatives come
> from, cold start, brute force vs ANN, and the limits of offline scoring — and §3 the
> harder questions *this* project's own results invite, chiefly: the popularity baseline
> beats the model at K=10, so why ship the model at all. §4 goes deeper on loss reduction,
> accidental-hit masking, full-catalogue evaluation, and index refresh. Every figure quoted
> is cross-checked against `artifacts/metrics.json`. *Body text is Thai.*

> ซ้อมตอบสำหรับโปรเจคนี้โดยเฉพาะ · ตัวเลขทุกตัวมาจาก **final run เดียวกัน**
> (`artifacts/metrics.json` + `artifacts/experiments.json`) — ถ้ารัน pipeline ใหม่แล้วตัวเลขขยับ
> ต้องกลับมาแก้ไฟล์นี้ด้วย
> อ่านคู่กับ `docs/model_card.md` (ผลลัพธ์ทางการ) และ `docs/DEVELOPMENT_LOG.md` (บทเรียนระหว่างทำ)

---

## 0. ตัวเลขที่ต้องจำให้ได้ (cheat sheet)

| | popularity (R0) | two-tower (R1) |
|---|---|---|
| Recall@10 | **0.0657** | 0.0583 |
| Recall@50 | 0.1835 | **0.2312** |
| NDCG@10 | **0.0331** | 0.0266 |
| Catalogue coverage@10 | 0.054 | **0.893** |
| Top-10%-popular share ของ slot | 1.000 | **0.063** |
| Gini exposure | 0.986 | **0.477** |

| อย่างอื่นที่ถูกถามบ่อย | ค่า |
|---|---|
| Dataset | MovieLens 100K — 100,000 ratings · 943 users · 1,682 movies (train vocab 1,679) |
| Split | temporal leave-last-1-out ต่อ user → test 943 / val 943 interactions |
| Model selection (val Recall@10) | R1 dim32 = 0.0541 · R2 dim64+L2 = 0.0424 → เลือก R1 |
| ขนาดโมเดล | R1 = 83,968 params (~328 KB float32) · R2 = 167,936 (2 เท่า) |
| Train time | ~16 วินาที/run บน laptop CPU (15 epochs, batch 256) |
| Serving | index build 0.19 s · query p50 0.56 ms / **p95 1.10 ms** · reload consistent ✅ |
| Slices Recall@10 | tail items 0.0723 vs head 0.0273 (**2.6×**) · low-activity users 0.0841 vs high 0.0287 |
| OOV test items | 0.21% (รายงาน ไม่ silently drop) |
| Sensitivity (rating ≥ 4) | Recall@10 0.0608 · Recall@50 0.2186 → ข้อสรุปไม่เปลี่ยน |
| Tests | 71 tests (leakage / metric / security / index reload / no-duplicate) |

---

## 1. 90-second pitch

**ภาษาไทย**

> ผมสร้าง *retrieval stage* ของ recommender ให้ครบวงจรบน MovieLens 100K — รับ `user_id`
> คืน Top-K หนังที่เป็น candidate ให้ ranking stage ต่อ โจทย์คือ catalogue ใหญ่เกินกว่าจะเอา
> ranking model ไปคิดคะแนนทุกชิ้นทุก request
> โมเดลเป็น two-tower: user tower กับ movie tower สร้าง embedding มิติ 32 แล้ววัดความเข้ากันด้วย
> dot product เทรนด้วย in-batch sampled softmax พร้อม accidental-hit masking ที่ผมเขียนเองด้วย
> TensorFlow ล้วนประมาณ 60 บรรทัด เพราะ TFRS อยู่ใน maintenance mode และไม่ compat กับ Keras 3
> จุดที่ผมให้น้ำหนักที่สุดคือ **วินัยการวัดผล**: split ตามเวลาแบบ leave-last-1-out ต่อ user
> พร้อม leakage audit ที่ raise error หยุด pipeline ทันทีถ้าพบอนาคตรั่ว, สถิติทุกอย่างคำนวณจาก
> train เท่านั้น, และ test set ถูกแตะครั้งเดียวหลังตัดสินใจครบแล้ว
> ผลลัพธ์ตรงไปตรงมา: popularity baseline ชนะที่ K=10 (0.066 vs 0.058) แต่ two-tower ชนะที่
> K=50 (0.231 vs 0.183) ซึ่งเป็น regime จริงของ retrieval และให้ coverage 89% เทียบกับ 5%
> ของ baseline — ผมรายงานผลตามที่มันเป็น พร้อม slice ที่ชี้ว่าโมเดลแข็งตรงไหน (long-tail 2.6×)
> ปิดท้ายด้วย serving index เป็น SavedModel ที่ตรวจ reload consistency ก่อน export, มี seen filter,
> cold-start fallback, input validation และ 71 tests

**English**

> I built the *retrieval stage* of a recommender end-to-end on MovieLens 100K: given a
> `user_id`, return Top-K candidates for a downstream ranker. Two-tower model — user and
> movie embeddings of dim 32, dot-product affinity, trained with in-batch sampled softmax
> and accidental-hit masking, written in ~60 lines of pure TensorFlow because TFRS is in
> maintenance mode and isn't Keras-3 compatible.
> The emphasis is evaluation discipline: per-user temporal leave-last-1-out split with an
> automated leakage audit that fails the pipeline, train-only vocabularies and popularity
> statistics, and a test set touched exactly once after all decisions were made.
> The honest headline is that the popularity baseline beats the ID-only model at K=10
> (0.066 vs 0.058), while the two-tower model wins at K=50 (0.231 vs 0.183) — the regime a
> retrieval stage actually operates in — with 89% catalogue coverage against 5%, and 2.6×
> higher recall on long-tail items. It ships as a reload-verified SavedModel index with
> seen-item filtering, a cold-start fallback, input validation, and 71 tests.

**กติกาเวลาพูด:** อย่าเลี่ยงผลที่แพ้ — พูดออกมาเองตั้งแต่ต้น แล้วอธิบายว่าทำไมยังเป็นผลที่ใช้ได้
คนสัมภาษณ์ให้คะแนน "รู้ตัวว่าผลแปลว่าอะไร" มากกว่า "ตัวเลขสวย"

---

## 2. คำถามออกแบบ 7 ข้อ (blueprint บทที่ 11)

### 2.1 retrieval ต่างจาก ranking อย่างไร

Retrieval ตัด catalogue หลักหมื่น–หลักล้านให้เหลือหลักร้อย ภายใน latency ไม่กี่ ms — ต้องใช้
โครงสร้างที่ precompute ได้ (two-tower + ANN) ตัวชี้วัดคือ **Recall@K ที่ K ใหญ่** เพราะหน้าที่คือ
"อย่าทำของดีหลุด" ส่วน ranking รับ candidate หลักร้อยมาเรียงด้วย feature หนัก ๆ (cross features,
context, sequence) วัดด้วย NDCG/AUC/calibration ที่ K เล็ก
ข้อจำกัดที่ทำให้แยกกัน: two-tower **ห้าม** ให้ user กับ item interact กันก่อนขั้น dot product
ไม่งั้น precompute item vector ไม่ได้ — นั่นคือราคาที่จ่ายเพื่อความเร็ว และเป็นเหตุผลว่าทำไมยังต้องมี ranker

### 2.2 ทำไมไม่ใช้ RMSE

RMSE วัดว่า "ทายคะแนนได้แม่นแค่ไหน" แต่ระบบจริงตัดสินใจว่า **จะโชว์อะไร** ซึ่งเป็นปัญหา ranking
ไม่ใช่ regression · RMSE ให้ค่าเท่ากันกับ error ที่อันดับ 1 และอันดับ 500, ประเมินบนเฉพาะคู่ที่มี rating
(missing-not-at-random) และไม่บอกอะไรเลยเรื่อง coverage/diversity
เราจึงใช้ Recall@K / NDCG@K บน implicit positives + วัด coverage และ popularity bias คู่กันเสมอ

### 2.3 ทำไมต้อง temporal split

เพราะ production ทำนายอนาคตจากอดีต ถ้า random split โมเดลจะได้เห็น interaction ที่เกิด
*หลัง* สิ่งที่มันต้องทาย = leakage และ metric จะพองทุกตัว
เราใช้ leave-last-1-out ต่อ user (ล่าสุด → test, ก่อนหน้า → val) แล้วบังคับด้วย `audit_no_leakage()`
ที่ raise `LeakageError` หยุด pipeline ถ้าเจอ future leakage / train history ไม่พอ / train∩test ทับกัน
**จุดที่ควรพูดเองเพราะสะท้อนความละเอียด:** invariant จริงคือ `max(train) ≤ val ≤ test` ไม่ใช่
"<" เพราะ ML-100K ละเอียดระดับวินาที และ **422 จาก 943 users** มี val timestamp เท่ากับ
interaction สุดท้ายของ train เป๊ะ (เรตหลายเรื่องรวดเดียว) — เราจึงตัด tie ด้วย `movie_id` ให้ deterministic
และวาดกราฟ §07 ตามความจริงข้อนี้ ไม่ over-claim

### 2.4 negative samples มาจากไหน

ใช้ **in-batch negatives**: ในหนึ่ง batch item ของ user คนอื่นคือ negative ของเรา → คิด softmax
บน logit matrix `[B, B]` ครั้งเดียว ได้ negative ฟรี B−1 ตัวโดยไม่ต้อง sample เพิ่ม
สองรายละเอียดที่ต้องพูดถึง:
1. **Accidental-hit masking** — ถ้า user คนอื่นใน batch ดูหนังเรื่องเดียวกัน มันคือ positive ไม่ใช่
   negative ต้อง mask logit นั้นด้วย `-inf` ก่อน softmax ไม่งั้นเรากำลังสอนโมเดลให้ผลักของที่ถูก
2. **Popularity bias ของ in-batch** — หนังดังโผล่เป็น negative บ่อยตามสัดส่วนความดัง โมเดลจะถูก
   ลงโทษเกินจริงเวลาแนะนำหนังดัง วิธีแก้มาตรฐานคือ **log-Q correction** (Yi et al. 2019)
   หัก `log P(item)` ออกจาก logit — อยู่ใน stretch goal และเป็นข้อที่ผมคิดว่าน่าจะช่วยที่ K=10 มากที่สุด

### 2.5 cold start ทำอย่างไร

- **User ใหม่ (แก้แล้ว):** `RetrievalService` ตรวจว่า user อยู่ใน vocab ไหม ถ้าไม่ → คืน popularity Top-K
  พร้อม `fallback_used: true` ใน response (มี test ยืนยัน) — ตอบตรงกว่าปล่อย OOV embedding ที่ไม่มีความหมาย
- **Item ใหม่ (ยังไม่แก้ — ประกาศเป็นข้อจำกัด):** ID-only tower ไม่มีทางสร้าง embedding ให้หนังที่ไม่เคยเห็น
  ทางแก้คือ **metadata tower** ใช้ genre 19 flags + title text ที่ `u.item` มีอยู่แล้ว → หนังใหม่ได้
  embedding จาก content ทันที เป็น stretch goal อันดับ 1 เพราะแก้ข้อจำกัดใหญ่สุดที่เราประกาศเอง
- ตอนนี้ test items ที่ไม่อยู่ใน train vocab มี 0.21% — เรารายงานเป็น metric ไม่ใช่ตัดทิ้งเงียบ ๆ

### 2.6 BruteForce ใช้ production ได้ไหม

**ได้ที่ scale นี้ ไม่ได้ที่ scale จริง** — 1,679 items × dim 32 คือ matmul จิ๋ว ๆ วัดได้ p95 = 1.10 ms
และ exact 100% ไม่มี recall loss จาก approximation
เส้นแบ่งคือขนาด catalogue: brute force เป็น O(N·d) ต่อ query พอ N ขึ้นหลักล้าน latency กับ memory
bandwidth จะพัง ต้องเปลี่ยนเป็น ANN (ScaNN / Faiss / vector DB) ซึ่งแลก recall เล็กน้อยกับ latency
สิ่งที่ดีคือ **contract ไม่เปลี่ยน** — ยังเป็น "user vector → nearest item vectors" เปลี่ยนแค่ตัว index
ถ้าจะเปลี่ยนจริงต้องมี benchmark recall@K เทียบ brute force ที่ latency budget เดียวกันก่อน
ไม่ใช่เปลี่ยนเพราะชื่อเท่ (เป็น stretch goal ข้อ 2)

### 2.7 offline score พอไหม

ไม่พอ และเป็นข้อจำกัดเชิงโครงสร้าง ไม่ใช่แค่ข้อมูลน้อย:
- ไม่มี **impression log** → "ไม่มี interaction" อาจแปลว่าไม่ชอบ หรือไม่เคยเห็นเลยก็ได้
  offline recall จึงลงโทษ candidate ที่ดีแต่ผู้ใช้ไม่เคยมีโอกาสเจอ
- ข้อมูลที่เทรนมาจาก policy เดิม (**feedback loop**) — โมเดลใหม่ที่แนะนำต่างออกไปจะถูกวัดต่ำเสมอ
- test มี 1 interaction/user → variance สูงมาก ต่างกัน ±0.01 ที่ K=10 คือ noise
สรุปคือ offline ใช้ **คัดออก** ได้ (อะไรที่แย่ชัด ๆ ไม่ต้องเอาขึ้น) แต่ยืนยันว่า "ดีขึ้นจริง" ต้อง
online A/B test ที่วัด business metric (CTR / watch rate / retention) พร้อม guardrail เรื่อง coverage
และ latency

---

## 3. คำถามยากที่มาจากผลลัพธ์ของเราเอง ⭐

> ส่วนสำคัญที่สุดของเอกสารนี้ — ผลจริงคือ **popularity ชนะ two-tower ที่ K=10** ต้องซ้อมให้ตอบได้นิ่ง

### 3.1 "โมเดลแพ้ baseline แล้วจะเอาไปใช้ทำไม?"

แพ้เฉพาะที่ K=10 (0.0583 vs 0.0657) แต่ **ชนะที่ K=50 (0.2312 vs 0.1835)** ซึ่งเป็น K ที่
retrieval stage ทำงานจริง — หน้าที่ของมันคือส่ง candidate ให้ ranker ไม่ใช่ตัดสินหน้าจอสุดท้าย
ที่ K=50 ความต่างคือ +26% relative และไม่ใช่ noise แบบที่ K=10
มิติที่สองคือของที่ baseline ให้ไม่ได้เลย: coverage 89.3% vs 5.4% (16×), Gini 0.477 vs 0.986,
top-10%-popular share 6.3% vs 100% — popularity แนะนำหนังชุดเดียวกันให้ทุกคน
ซึ่งแปลว่า "ไม่มี personalization" และทำให้ feedback loop แคบลงเรื่อย ๆ
ในระบบจริง retrieval ผลิตหลาย candidate source พร้อมกัน (popularity + two-tower + trending)
แล้วให้ ranker เลือก — คำถามที่ถูกจึงไม่ใช่ "อันไหนชนะ" แต่คือ "แต่ละ source เติมอะไรที่อีกอันไม่มี"

### 3.2 "ตัวเลขต่างกันแค่นี้มีนัยสำคัญไหม?"

ที่ K=10 — **ไม่**: test มี 1 interaction ต่อ user, n = 943, hit rate ระดับ 6% แปลว่าต่างกัน 0.007
คือหลักสิบ user ± ไม่กี่คน ผมจึงเขียนไว้ในทั้ง README และ model card ว่า ±0.01 ที่ K=10 คือ noise-level
ถ้าจะเคลมจริงต้องมี bootstrap CI หรือ paired test ต่อ user (งานที่ควรทำเพิ่ม)
ที่ K=50 ช่องว่างกว้างกว่ามาก (+0.048, +26% relative) และไปทางเดียวกับ coverage/slice — สามหลักฐาน
ที่ independent กันชี้ทางเดียวกัน น่าเชื่อกว่าตัวเดียวโดด ๆ
**ห้ามตอบว่า "significant" ถ้ายังไม่ได้คำนวณ** — จุดนี้คนสัมภาษณ์จับได้ทันที

### 3.3 "แล้วจะทำให้ดีขึ้นยังไง?" (เรียงตามที่คิดว่าคุ้มจริง)

1. **Log-Q correction** — แก้ popularity bias ของ in-batch negatives ตรงจุด เป็นข้อที่น่าจะขยับ K=10
   ได้มากที่สุดด้วยแรงน้อยที่สุด (~4 ชม.)
2. **Metadata tower** (genre 19 flags + title) — แก้ item cold-start และเพิ่ม signal ให้หนังที่มี
   interaction น้อย ซึ่งคือหางยาวส่วนใหญ่ของ catalogue
3. **Hard negatives / mixed negatives** — in-batch อย่างเดียวได้ negative ที่ง่ายเกินไป
4. **Hybrid retrieval** — ยิงหลาย source แล้ว dedupe ก่อนส่ง ranker (ตรงกับข้อ 3.1)
5. **Sequence model ฝั่ง user** — ML-100K มี timestamp ครบ แต่ user tower ตอนนี้ใช้แค่ ID เดียว
   ทั้งที่ประวัติล่าสุดคือ signal ที่แรงที่สุดในระบบจริง

### 3.4 "ทำไมไม่ใช้ TFRS ตามที่ blueprint บอก?"

TFRS อยู่ใน maintenance mode และไม่ compatible กับ Keras 3 — ต้องตั้ง `TF_USE_LEGACY_KERAS=1`
ซึ่ง contaminate ทั้ง process (ทุก Keras import ในโปรเจคเปลี่ยนพฤติกรรมตาม) ผมเลยเขียน retrieval
loss เอง ~60 บรรทัด: dot product → mask accidental hits → `softmax_cross_entropy` แบบ **SUM reduction**
ได้ 3 อย่าง: dependency น้อยลง, โค้ดที่ audit ได้จริง, และความเข้าใจว่า loss ทำอะไรอยู่
(deviation นี้บันทึกไว้ใน `DEVELOPMENT_LOG.md` §2 พร้อมเหตุผล — ไม่ใช่แอบเปลี่ยน)

### 3.5 "coverage สูงแปลว่าดีจริงหรือ?"

ไม่จำเป็น — **random recommender ก็ได้ coverage 100%** coverage เดี่ยว ๆ วัดแค่ว่า
"กระจายไหม" ไม่ได้วัดว่า "ตรงไหม" ต้องอ่านคู่กับ relevance เสมอ
ในเคสนี้สิ่งที่ทำให้พูดได้เต็มปากคือ coverage 89.3% มาพร้อม **Recall@50 ที่สูงกว่า baseline** และ
slice ที่ชี้ว่าโมเดลเก่งตรง long-tail จริง (tail 0.0723 vs head 0.0273 = 2.6×) — ถ้าเป็นการสุ่ม
สองอย่างนี้จะไม่เกิดพร้อมกัน
เหตุผลที่ต้องวัด coverage/Gini ตั้งแต่แรกคือ feedback loop: สิ่งที่โชว์วันนี้กลายเป็น training label
พรุ่งนี้ ถ้า exposure กระจุก (Gini 0.986 แบบ popularity) ระบบจะแคบลงเรื่อย ๆ โดยที่ offline metric
ไม่ฟ้องอะไรเลย

---

## 4. Deep-dive ที่อาจโดนถามต่อ

### 4.1 ทำไม SUM reduction ถึงจำเป็น (บทเรียนจริงจาก `DEVELOPMENT_LOG.md` §3)

ตอนแรกใช้ `reduce_mean` กับ in-batch softmax → loss แบนสนิท แต่ embedding ขยับนิดหน่อยจนดู
เหมือนเรียนรู้ได้ สาเหตุ: MEAN หาร loss ด้วย batch size (256) → gradient ต่อ parameter เล็กลง ~256 เท่า
และเนื่องจาก Adagrad สะสม squared gradient ใน denominator, step ที่เล็กมากตั้งแต่ต้นทำให้แทบไม่ขยับ
เปลี่ยนเป็น SUM (semantics เดียวกับ TFRS `Retrieval` task) แล้ว loss ลดตามปกติ
**วิธี debug ที่ควรเล่า:** ไม่ได้เดา — วัด gradient norm แล้ว apply 1 step วัด max abs change
ของ embedding matrix เทียบสองเวอร์ชัน

### 4.2 accidental-hit masking คืออะไร ถ้าไม่ทำจะเกิดอะไร

ใน batch เดียวกัน ถ้า user A และ user B ต่างก็ดู *Star Wars* logit ของ (A, item ของ B) คือ
positive ที่ถูก label ว่า negative → cross-entropy จะดัน score ของหนังดังลง ทั้งที่มันถูก
ยิ่ง item ดัง ยิ่งชนกันบ่อย = ลงโทษหนังดังเป็นสัดส่วนกับความดัง
วิธีทำ: สร้าง mask จากการเทียบ item id ใน batch แล้วเซ็ต logit ที่ชนเป็น `-inf` **ยกเว้น diagonal**
(ตัวมันเองต้องเหลือไว้เป็น positive)

### 4.3 ทำไม full-catalogue eval ดีกว่า sampled

Sampled evaluation (เทียบ positive กับ negative สุ่ม 100 ตัว) ให้ตัวเลขสวยกว่าและ **บิดอันดับโมเดล
สลับกันได้** (Krichene & Rendle, KDD 2020) เพราะ negative ที่สุ่มมามักง่าย
catalogue เรามีแค่ 1,679 items → scoring ทั้งหมดเป็น matmul ครั้งเดียว ถูกกว่าการมาเถียงเรื่อง bias
ทีหลังมาก ที่ scale ใหญ่ค่อยใช้ sampled ตอน monitor รายวัน แต่ผลที่ report ควรเป็น full

### 4.4 scale ไป ANN อย่างไร และ index refresh บ่อยแค่ไหน

- **Index:** brute-force SavedModel → ScaNN/Faiss เมื่อ N โต; ต้อง benchmark recall@K vs latency
  เทียบของเดิมก่อนตัดสินใจ ไม่งั้นก็แค่แลก recall ทิ้งไปเปล่า ๆ
- **Refresh cadence แยกกัน 3 ชั้น:** item embedding + index rebuild (รายวัน/รายชั่วโมงตามอัตราหนังใหม่),
  user embedding (ถี่กว่า หรือคำนวณสดจาก history), full retrain (ตามรอบหรือตาม drift trigger)
- **Triggers:** OOV rate, coverage/Gini หลุด threshold, score distribution drift, index age
- **สิ่งที่คนมักลืม:** ตอน swap index ต้อง atomic + มี version stamp ในทุก response
  (เราทำแล้ว: `model_version` / `index_version` ติดไปกับทุก response) ไม่งั้นเวลา metric เพี้ยน
  จะสืบไม่ได้ว่ามาจาก index ตัวไหน

### 4.5 security/robustness ที่ใส่ไว้ (ถ้าเจอสาย platform/infra)

download ผ่าน HTTPS + **pinned SHA-256** (checksum ไม่ตรง = ลบทิ้งทันที), extract แบบกัน zip-slip,
input validation ที่ serving (`user_id` regex `[A-Za-z0-9_-]{1,64}`, `k ∈ [1,100]`),
export แบบ fail-hard ถ้า reload แล้ว Top-K ไม่ตรงกับ in-memory, ruff `S` rules ใน lint,
และ CI รัน ruff + pytest ทุก push — artifact ที่ reproduce ตัวเองไม่ได้ อันตรายกว่าไม่มี artifact
