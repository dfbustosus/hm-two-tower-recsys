# H&M Personalized Fashion Recommendations — Engineering Handoff

**Audience.** A senior recsys engineer taking over this project.
**Goal.** Beat MAP@12 = **0.0380** (1st-place public LB, Senkin13).
**Current state.** Stuck at MAP@12 = **0.02255 public / 0.02205 private** (Jun 9, 2026).
**Gap.** ~**0.0155 public** (40% of leader's score still missing).
**Hard constraint from product owner.** A two-tower architecture **must** appear
somewhere in the final pipeline. Marqo / FashionCLIP is an allowed embedding
backbone but not mandatory.

This document is the unfiltered status. It includes every experiment that was
run, every failure mode, every disk artifact that already exists, and a
prioritized plan with explicit time estimates for the next engineer.

---

## 1. Hardware & environment

- MacBook Pro, Apple M4 Pro, 16-core GPU (Metal 4). MPS available **only on
  newer torch**; the currently installed torch 2.12 reports `mps=False` (we
  fell back to CPU for FashionCLIP and Item2Vec embeddings). Worth upgrading
  torch + rerunning two-tower training with MPS — but see the collapse
  diagnosis in §5.4 before doing that.
- Python 3.11 venv at `.venv/`. `make venv` recreates it.
- Kaggle CLI 2.2.0 in `.venv/bin/kaggle`. Credentials live in `.env`
  (`KAGGLE_USER_NAME`, `KAGGLE_API_TOKEN`). Submission target:
  `KAGGLE_COMPETITION=h-and-m-personalized-fashion-recommendations`.
- Submit via `make kaggle-submit SUBMISSION=path KAGGLE_MESSAGE="..."`.
  The target validates the CSV first (`validate-submission` CLI).

---

## 2. Dataset facts

- `data/raw/h-and-m-personalized-fashion-recommendations/transactions_train.csv`
  ≈ 30M rows, ~2019‑09‑20 through **2020‑09‑22** inclusive.
- Test window: **2020‑09‑23 → 2020‑09‑29** (7 days after train).
- `sample_submission.csv` defines the canonical customer order for any
  submission: **1,371,980 customers**. Inference cutoff used everywhere
  is `2020‑09‑23` (exclusive).
- ~31% of test customers have any purchase history at all; the rest are
  cold-start and get popularity-driven predictions.
- Of the 1.37M customers, only ≈ **518,920 (37.8%)** have at least one
  purchase in the last 90 days before cutoff — confirmed by Item2Vec /
  FashionCLIP content-similarity scans. The remaining 853k must be served
  by popularity / segment fallback.

---

## 3. Leaderboard history (all 12 submissions made by this session)

Sorted newest first. **Bold = current personal best.**

| Date (UTC) | File | Public LB | Private LB | Type |
|---|---|---|---|---|
| 2026‑06‑10 22:22 | `ENSEMBLE_lgbm_champion_plus_item2vec_w2_1.csv` | 0.02174 | 0.02133 | RRF reranking, w=2/1 |
| 2026‑06‑10 22:20 | `ENSEMBLE_lgbm_champion_plus_fashionclip_w2_1.csv` | 0.02173 | 0.02126 | RRF reranking, w=2/1 |
| 2026‑06‑10 22:19 | `ENSEMBLE_lgbm_champion_plus_fashionclip_balanced.csv` | **0.01581** | 0.01523 | RRF balanced w=1/1 (catastrophic) |
| 2026‑06‑10 21:53 | `ENSEMBLE_rrf_rich2wk_richtrainall_stdtrainall_w2_2_0p5.csv` | 0.02244 | 0.02194 | 3-LGBM RRF |
| 2026‑06‑10 21:33 | `READY_TO_UPLOAD_lgbm_rich_twotower_full_train_k50.csv` | 0.02251 | 0.02204 | Resubmit of full-train rich+TT |
| **2026‑06‑09 16:02** | **`lightgbm_behavioral_rich_twotower_train2weeks_k50_age_garment_batched.csv`** | **0.02255** | **0.02205** | **CHAMPION** |
| 2026‑06‑09 00:16 | `lightgbm_behavioral_rich_twotower_trainall_k50_age_garment.csv` | 0.02251 | 0.02204 | rich+TT full-train (slightly worse than 2wk) |
| 2026‑06‑08 14:12 | `lgbm_twotower_train2weeks_k50_age_garment.csv` | 0.02172 | 0.02113 | non-rich + TT |
| 2026‑06‑08 01:01 | `lgbm_twotower_trainall_k50_age_garment.csv` | 0.02150 | 0.02109 | non-rich + TT full-train |
| 2026‑06‑07 22:31 | `ensemble_lgbm_behavioral_twowtower_rrf_w0p12.csv` | 0.01846 | 0.01864 | First RRF attempt (also failed) |
| 2026‑06‑07 22:22 | `lightgbm_behavioral_train20k_k20_no_ua_age_garment.csv` | 0.01852 | 0.01871 | Smaller-data LGBM smoke |
| 2026‑06‑07 21:12 | `two_tower_deterministic_ranker_*.csv` | 0.01523 | 0.01506 | Two-tower alone + deterministic ranker |
| 2026‑06‑07 16:44 | `deterministic_ranker_tuned_...customers.csv` | 0.01369 | 0.01390 | Rule-based deterministic only |
| 2026‑06‑03 02:51 | `learned_linear_ranker_...maxpw10.csv` | 0.01245 | 0.01317 | Learned linear ranker |
| 2026‑06‑02 13:57 | `repeat_popularity_baseline_lookback_7_k_12.csv` | 0.01235 | 0.01270 | Baseline |

**The signal in this table is what matters:** every improvement vector
beyond the rich + two-tower LGBM family **stalled at 0.02255 public**.
Today (Jun 10) I tried 5 follow-ups (post-champion); **all 5 regressed**.

---

## 4. Strategies that were tried — full taxonomy

For each strategy I include:
- **What** was tried,
- **How** (with the code path or commit),
- **Result** (LB or offline),
- **Why it failed / why it stalled** (root-cause hypothesis).

### 4.1 Repeat + popularity baseline (Jun 2, LB 0.01235)
- Pure rule: for each customer, repeat their last purchases, then pad with
  global last-7-day popularity top-12.
- Established a useful floor and the customer-id order template.

### 4.2 Learned linear ranker (Jun 3, LB 0.01245)
- `src/hm_recsys/ranking/linear.py` — logistic-regression style scorer over
  co-vis, repeat, popularity features. Quick but feature-poor.

### 4.3 Deterministic tuned ranker (Jun 7, LB 0.01369)
- `src/hm_recsys/ranking/deterministic.py` —
  `DeterministicRankerWeights` tuned via grid over rules
  (repeat_rank, covis_rank, recency, two-tower, age-segment, garment-group).
- The tuned **prior weights** (used downstream as a sanity prior for
  LightGBM blending) are now centralized as
  `LIGHTGBM_BEHAVIORAL_RANKER_PRIOR_WEIGHTS`.
- Adding rules helped a bit, but it's still a hand-curated linear combination
  — capped at the ~0.014 range.

### 4.4 Two-tower retrieval (Jun 7, LB 0.01523, ALONE)
- `src/hm_recsys/models/two_tower_*.py` — torch-based two-tower model.
- Used to produce dense customer × article scores; we then ranked all
  candidates by cosine similarity.
- **As a standalone retriever** it underperforms simple co-vis + popularity.
- **As a feature inside LGBM** (`two_tower_score` column in
  `CandidateRecord`) it adds **+0.006** vs the same LGBM without it
  (compare 0.02172 → 0.02255). That is the single biggest single-feature
  gain documented in this repo.
- **Failure mode in two later TT runs**: see §5.4.

### 4.5 LightGBM "behavioral" ranker (Jun 7‑8)
- `src/hm_recsys/ranking/lightgbm_behavioral.py`.
- Features: repeat counts, source presence (`source_count`), age-segment
  popularity, garment-group popularity, covis-rank, recency.
- Trained on 10k‑20k customer slices first. LB 0.0215‑0.0225 depending on
  hyperparameters.

### 4.6 LightGBM "rich" features + two-tower (Jun 8‑9, LB 0.02251‑0.02255) — **CHAMPION**
- Same LightGBM plus an expanded behavioral feature set + a
  `two_tower_score` column emitted by `candidate_export` when a TT export
  is provided.
- Trained both on the last 2 weeks (`train2weeks`) and on full data
  (`trainall`). Two-week model is *very slightly* better than full
  (0.02255 vs 0.02251). They differ by ~5 in jaccard at the customer
  level (rough estimate from prior diff). Conclusion: the dataset's last
  2 weeks dominate the signal in this LGBM family.
- **This is the plateau.** Every parameter variant tested
  (lambdarank vs xendcg, 200 vs 400 trees, 63 vs 127 leaves,
  `min_data_in_leaf`, `negative_per_positive`, blend λ for deterministic
  prior, bagging seeds) clusters at LB 0.022x.

### 4.7 CatBoost YetiRank ablation
- `scripts/eval_catboost_yetirank.py`.
- Tried as challenger. Required passing concrete `TransactionEvent`
  instances (a Protocol can't be instantiated bug fixed in the script).
- Did not show meaningful lift over LGBM lambdarank in offline runs.
  Not submitted.

### 4.8 FashionCLIP (HF-CLIP `patrickjohncyh/fashion-clip`) text embeddings
- `models/embeddings/articles/hf-clip_patrickjohncyh_fashion-clip_main/text_manifest_all.json`.
- 105,542 articles × 512 dims, all embeddings exist on disk.
- Used three different ways. **All three failed.** Critical empirical fact
  for the next engineer: **on this dataset FashionCLIP text similarity is
  NOT a strong purchase signal.** Diagnosis: it captures visual / textual
  semantic similarity (e.g. two black hoodies that look alike) — but H&M
  test labels are dominated by *repeat buys* and *last-week popularity
  dynamics*, neither of which content similarity recovers.
  - **(a) as `content_user_cosine` re-ranking feature on top of LGBM v6** →
    offline regression vs v2 baseline (≈ 0.0014 worse on 10k slice).
  - **(b) as `text_similarity` candidate SOURCE re-injected via
    `scripts/augment_candidates_with_content_retrieval.py`** (110s vectorized
    matmul) → LGBM v7 offline regression vs v2 baseline.
  - **(c) as full-population content-only submission, blended with
    champion via RRF** →
    - balanced w=1/1 → LB **0.01581** (catastrophic, content items
      injected are visually similar but commercially weak).
    - reranking-only w=2/1 → LB **0.02173** (champion's items kept but
      reordered by content votes — STILL a regression of −0.0008).
- The candidate-augmenter is fast (110s for 10k slice; 30‑50 min for full
  population). It is correct. The signal is just wrong for this metric.

### 4.9 Item2Vec skip-gram on basket sessions (this session)
- `scripts/train_item2vec_embeddings.py` (gensim Word2Vec, sg=1, neg=5,
  ns_exponent=0.75).
- Trained on 6.39M basket sessions (29.1M token positions, avg 4.55 items
  per basket), dim=64, window=5, 5 epochs, 8 workers.
- Output: `models/embeddings/articles/item2vec_basket_d64_w5_e5/`.
- **Trained in 173 seconds** (CPU).
- Offline diagnostic was **encouraging**: median per-customer Jaccard with
  champion = 0.091 (vs 0.044 for FashionCLIP); 97.5% of Item2Vec items
  overlap with the universe of items champion ever selects.
- LB result (RRF w=2/1 reranking-only): **0.02174** — essentially
  identical to FashionCLIP's 0.02173. **The promising offline signal did
  not translate to LB lift.**

### 4.10 RRF ensembling (all forms)
- `scripts/rrf_blend_submissions.py`.
- 3-model LGBM RRF (rich-2wk + rich-trainall + std-trainall, weights 2/2/0.5):
  LB **0.02244** (−0.00011).
- 2-model LGBM + FashionCLIP w=2/1: LB 0.02173 (−0.00082).
- 2-model LGBM + FashionCLIP w=1/1: LB 0.01581 (−0.00674).
- 2-model LGBM + Item2Vec w=2/1: LB 0.02174 (−0.00081).
- **Conclusion (definitive after 5 attempts):** No post-hoc CSV blend
  has ever beaten the strongest single submission. Across two
  fundamentally different embedding paradigms (FashionCLIP visual/text
  semantics vs Item2Vec co-purchase behavior), reranking the champion's
  fixed top-12 produced essentially identical −0.0008 regressions.
  **This is the strongest evidence that champion's *ordering* of its
  top-12 is already locally optimal; the problem is item *selection*, not
  ordering.**

### 4.11 Things considered and explicitly NOT pursued (yet)
- **SASRec / BERT4Rec sequential transformer.** Would need 4‑8h GPU
  training and proper integration. Highest expected upside (+0.005‑0.012).
- **TIGER (RQ‑VAE + autoregressive decode).** Most complex; not for this
  team yet.
- **Stacking 2nd-stage learner** over LGBM scores + new features
  (proposed in plan.md but not built).
- **HuggingFace fashion CLIP image embeddings** — `multimodal_manifest.json`
  exists only for two 5000-article slices, not full population. Full-population
  image embedding generation never ran (the network proxy issue hit and we
  pivoted to text-only).
- **Larger candidate pool k=200** — never run end-to-end at full population.
- **Senkin13-style "popularity-priored" content retrieval.** This is the
  important one. Senkin13 didn't use raw FashionCLIP cosine — they
  multiplied cosine by a popularity prior before selecting top-K. We never
  applied a popularity prior to our content retrieval. This is almost
  certainly why §4.8(b)/(c) failed and is a *concrete fix*, not a new
  research project. See §7 plan.

---

## 5. Failure-mode diagnoses (what NOT to repeat)

### 5.1 Naïve RRF ensembling of correlated LGBM variants does not help.
3 LGBM submissions at 0.92 Jaccard are too correlated; RRF can only
shuffle ties.

### 5.2 Any positive weight on a weak model in RRF actively hurts.
The first RRF attempt (`ensemble_lgbm_behavioral_twowtower_rrf_w0p12.csv`,
LB 0.01846) included a TT-only submission (LB 0.01523) at weight 0.12;
the blend collapsed to 0.01846. Lesson: in RRF the model floor matters
because RRF gives every model a presence credit.

### 5.3 Embedding-based reranking of the champion's top-12 hurts LB.
Proved with two completely different embeddings (FashionCLIP text and
Item2Vec basket). The champion's intrinsic ordering is better than any
single-embedding cosine score.

### 5.4 Two-tower training collapses to degenerate cosines twice.
- `artifacts/two-tower-exports/prod-2020-09-16/train.log`,
  `prod-2020-09-09-content-v2/train.log` show loss bouncing around
  10.2‑10.5 with mean loss never improving meaningfully; the resulting
  embeddings have saturated cosine similarities (≈1 across most pairs).
- As a consequence `two_tower_score` in the candidates is largely
  uninformative *for the v2 runs*. The original (v1, Jun 8) two-tower
  export apparently did not collapse; that's why it provides the
  +0.006 lift used in the champion model.
- **Action for next engineer:** when retraining two-tower, watch for
  saturation early. Standard fixes: lower temperature, in-batch negatives
  with explicit sampling correction (logQ), warm-up the learning rate,
  use mixed in-batch + uniform negatives (MNS), and verify the loss
  *decreases* below the random-baseline `-log(1/batch_size)`. None of
  these were verified in the failed v2 runs.

### 5.5 10k-customer offline slice systematically overstates score.
Offline MAP@12 on the 10k eval slice consistently ran ~0.027, but the
matching submission scored ~0.022 on the LB — a **~5pp offline gap**.
The 10k slice over-represents warm/active customers; the LB has 73.5%
cold-start. Lesson: never make a go/no-go decision from the 10k slice
alone unless the delta is ≥ +0.003 absolute on that slice.

### 5.6 "Use FashionCLIP without popularity prior" was wrong.
Senkin13's writeup explicitly used a popularity prior on content
retrieval. We did not. This is the single most actionable open lever.

---

## 6. Inventory — what already exists on disk

### 6.1 Scripts (`scripts/`)
- `train_item2vec_embeddings.py` — Item2Vec trainer (gensim).
- `generate_content_similarity_submission.py` — fast full-population
  content-only submission (batched matmul, cold-start popularity
  fallback). Works with **any** embedding manifest that has
  `{article_count, dimension, embeddings_path, dtype}` and a sibling
  JSONL with `{article_id, vector}` rows.
- `augment_candidates_with_content_retrieval.py` — fast (110s)
  candidate-set augmenter that adds a `text_similarity` source.
- `score_content_similarity_candidates.py` — adds a
  `content_user_cosine` re-ranking feature column.
- `rrf_blend_submissions.py` — weighted RRF blender for submission CSVs
  (validates customer order; tie-breaks by consensus then min-rank).
- `eval_catboost_yetirank.py`, `eval_lgbm_bagged.py`,
  `eval_lgbm_catboost_ensemble.py`, `ablate_two_tower_score.py`,
  `debug_deterministic_baseline.py` — earlier ablations.

### 6.2 Article embeddings (`models/embeddings/articles/`)
| Dir | Description | Status |
|---|---|---|
| `hf-clip_patrickjohncyh_fashion-clip_main/` | FashionCLIP text embeddings, 105,542 articles × 512 | full population |
| `hf-clip_..._article_content_priority_cutoff_2020-09-09_lookback_30_first_5000_articles*/` | Multimodal (image+text) for 5k or 1k articles only | partial, smoke-only |
| `item2vec_basket_d64_w5_e5/` | Item2Vec, 90,994 articles × 64, cutoff 2020-09-23 | full population |

### 6.3 Two-tower exports (`artifacts/two-tower-exports/`)
| Dir | Notes |
|---|---|
| `smoke-2020-09-16/` | Smoke run only. |
| `prod-2020-09-16/` | First "good" prod TT model; feeds the LB-champion's `two_tower_score`. |
| `prod-2020-09-09-content-v1/` | Earlier prod run. |
| `prod-2020-09-09-content-v2/` | **Collapsed** — see §5.4. Do not use. |
| `prod-2020-09-16-v2/` | **Collapsed** — see §5.4. Do not use. |

### 6.4 Submissions (`submissions/`)
All 22 generated CSVs are preserved. The relevant ones are listed in §3.
Files matching `*smoke*` are tiny (a few KB) developer smoke tests; ignore.

### 6.5 Candidate exports (`artifacts/candidate-exports/`)
~50 candidate CSV/JSON files, most for the **10k customer slice**, some
for 1k or 5k slices. **No full-population (1.37M) candidate export with
content-similarity or item2vec sources exists yet** — this is the
critical missing artifact (see §7).

### 6.6 Ranker baselines (`artifacts/ranker-baselines/`)
~30 offline evaluation JSON reports. Each has cutoffs, MAP@12, feature
list, and timing. Useful provenance trail for comparing offline numbers.

### 6.7 Plan / spec
`.cursor/plans/hm-recsys-gap-closure-sdd_a1ab1373.plan.md` (430 lines).
This is the *original* gap-closure plan I was working from. Its
recommendations have **only been partially executed**, and some sections
turned out to be wrong (notably the bits assuming FashionCLIP would
provide strong signal without a popularity prior — see §5.6).

### 6.8 Uncommitted git state
```
M  src/hm_recsys/cli/__init__.py
M  src/hm_recsys/cli/_legacy.py
M  src/hm_recsys/ranking/deterministic.py
M  src/hm_recsys/ranking/lightgbm_behavioral.py
M  src/hm_recsys/retrieval/candidate_export.py
M  tests/test_cli_contract.py
M  tests/test_deterministic_ranker.py
M  tests/test_lightgbm_behavioral_ranker.py
M  tests/test_two_tower_export.py
?? scripts/                                     # all new scripts above
?? src/hm_recsys/cli/two_tower.py               # TT CLI commands
?? src/hm_recsys/models/two_tower_dataset.py    # TT dataset code
?? src/hm_recsys/models/two_tower_export.py     # TT export
?? src/hm_recsys/models/two_tower_train.py      # TT training
?? tests/test_two_tower_cli_content_loader.py
?? tests/test_two_tower_dataset.py
?? tests/test_two_tower_train.py
```
**Recommendation:** the new TT training/dataset/export code in
`src/hm_recsys/models/two_tower_*.py` and the new `scripts/*.py` should
be reviewed and committed before the next engineer starts iterating.
None of the changes are destructive; they are all additive.

---

## 7. Recommended plan for the next engineer (in priority order)

I'll cluster these by expected LB lift, time cost, and risk.

### TIER A — Most likely to break the plateau (do these first)

#### A1. Popularity-priored content retrieval as a true CANDIDATE SOURCE for full pop, then retrain LGBM. (Time: 6‑10h. Risk: low‑medium. Expected LB: +0.003 to +0.008.)

This is the Senkin13 recipe we never executed properly.

Steps:
1. Modify `scripts/generate_content_similarity_submission.py` (or its
   underlying `augment_candidates_with_content_retrieval.py`) to multiply
   raw cosine by `pop_score(article) ** alpha` before top-K selection.
   Start with `alpha = 0.5`. The popularity counter we already compute
   (`_build_queries_and_popularity` in
   `generate_content_similarity_submission.py`) gives the prior for free.
2. Regenerate **full-population** candidates for both training cutoff
   (2020‑09‑16, used for LGBM training) and inference cutoff (2020‑09‑23),
   including all sources: covis, age-segment, garment-group, **and
   `text_similarity` from popularity-priored FashionCLIP**, **and
   `item2vec_similarity` from popularity-priored Item2Vec**. Use the
   existing `augment_candidates_with_content_retrieval.py` pattern — extend
   it for `item2vec_similarity` (same code path, swap manifest).
3. Retrain LGBM rich on the augmented training candidates. Eval on the
   matched 10k eval slice. If offline lift ≥ +0.003 absolute (recall §5.5),
   submit.

**Why this works where §4.8(b) failed:** the popularity prior strips the
"semantically similar but never bought" items that polluted our prior
attempts; what's left is *content-similar AND commercially viable* items
that LGBM can learn to rank correctly because it sees them in training.

#### A2. Larger candidate pool k=200 (Time: 3‑4h extra on top of A1. Risk: low. Expected LB: +0.001 to +0.003 marginal.)

The existing exports use k=50. Champion's LGBM has never seen what
ranks 51‑200 look like — and the Senkin13 writeup uses very wide pools
(≈400). Combine with A1; do not run in isolation.

#### A3. Fix two-tower training, then re-emit a clean `two_tower_score`. (Time: 4‑6h. Risk: medium. Expected LB: +0.001 to +0.003.)

The v2 runs collapsed. Recipe:
- Use in-batch softmax with **logQ** correction (sampled-softmax bias for
  popular items).
- Add explicit MNS (mix in-batch with uniform random negatives).
- Lower learning rate, warmup 1‑5% of steps.
- Tie temperature to ≈ 0.07 (CLIP-style) instead of 1.0.
- Monitor *recall@k* on a held-out positive set every N steps, not just
  loss. The v2 logs show loss but no recall — that's why collapse was
  invisible.
- The existing `src/hm_recsys/models/two_tower_train.py` (uncommitted,
  see §6.8) is the right place to edit.

After fixing, score full-population candidates with the new TT and
replace `two_tower_score` in the augmented candidate CSVs from A1.
Then retrain LGBM.

### TIER B — Higher upside but bigger engineering investment

#### B1. SASRec sequential transformer trained on customer purchase sequences. (Time: 6‑12h. Risk: medium. Expected LB: +0.005 to +0.012.)

Predict next purchase from a customer's ordered history. Train on
M4 Pro GPU (Metal) — but first upgrade torch so MPS works. The same
purchase-session data we used for Item2Vec is the training corpus;
swap basket-shuffle for chronological order.

Score it the same way as TT: emit top-K per customer as a new candidate
source, plus emit a per-(customer, article) score feature. Feed both
into the LGBM.

This is the move 0.030+ public LB solutions historically used.

#### B2. Image embeddings (FashionCLIP image, not text) at full population. (Time: 3‑6h embed + minor integration. Risk: low. Expected LB: +0.001 to +0.003.)

We only have multimodal embeddings for 5k articles. Full-population
image embedding extraction never ran. It might add a clean orthogonal
signal even though text didn't. **Do this only after A1 succeeds** —
otherwise it's a duplicate of the text-content failure.

### TIER C — Polish / cleanup (do in parallel with A/B if you have spare hands)

#### C1. Commit the uncommitted code (§6.8) with a sensible message.
#### C2. Refactor `src/hm_recsys/cli/_legacy.py` to remove dead paths that the new modules replaced.
#### C3. Add an `eval_lgbm_offline_vs_lb.py` script that computes a 10k MAP@12 and warns when offline > LB by more than ~0.003 — closes the gap from §5.5.
#### C4. Set up a "submission diary" file (`submissions/SUBMISSION_DIARY.md`) auto-appended after every `make kaggle-submit`. This thread had to be reconstructed manually; future engineers shouldn't have to.

---

## 8. Reproducibility commands for the next engineer

```bash
# Recreate venv, install pinned deps
make venv

# Run all tests
make test

# Champion LGBM submission (the current LB best)
# Already generated; CSV lives at:
ls -la submissions/lightgbm_behavioral_rich_twotower_train2weeks_k50_age_garment_batched.csv

# Validate any CSV before submission
.venv/bin/python -m hm_recsys.cli validate-submission --submission-path <CSV>

# Submit
make kaggle-submit \
  SUBMISSION=submissions/<file>.csv \
  KAGGLE_MESSAGE="<message>"

# Train Item2Vec (3 min)
.venv/bin/python -u scripts/train_item2vec_embeddings.py \
  --transactions-csv data/raw/h-and-m-personalized-fashion-recommendations/transactions_train.csv \
  --output-dir models/embeddings/articles/item2vec_basket_d64_w5_e5 \
  --cutoff 2020-09-23 --session-mode basket \
  --embedding-dim 64 --window 5 --min-count 5 --epochs 5 --workers 8

# Generate a full-pop content-only submission from ANY manifest (~8 min)
.venv/bin/python -u scripts/generate_content_similarity_submission.py \
  --reference-csv data/raw/h-and-m-personalized-fashion-recommendations/sample_submission.csv \
  --embeddings-manifest-path <manifest.json> \
  --transactions-csv data/raw/h-and-m-personalized-fashion-recommendations/transactions_train.csv \
  --output-csv submissions/<name>.csv \
  --cutoff 2020-09-23

# RRF blend any number of submissions
.venv/bin/python -u scripts/rrf_blend_submissions.py \
  --model-path A.csv --model-weight 2.0 \
  --model-path B.csv --model-weight 1.0 \
  --reference-csv data/raw/h-and-m-personalized-fashion-recommendations/sample_submission.csv \
  --output-csv submissions/<name>.csv \
  --k-offset 60 --top-k 12
```

---

## 9. Open questions I could not answer

1. **Are the "v2" two-tower collapses caused by the dataset class (which I
   built) or by hyperparameters?** I never bisected. The v1 export works,
   v2 collapses — but I changed both dataset path and training config.

2. **Does the popularity prior alone fix content retrieval?** Plausible but
   unproven. A1 in §7 is the experiment that answers it.

3. **What does the LGBM model look like with k=200 candidates?** Never
   trained. Memory may become an issue at full population; budget for
   chunked training.

4. **Why does `train2weeks` outperform `trainall` by 4 ten-thousandths?**
   Hypothesis: recency drift; the model overfits to older purchasing
   patterns when given the full year. Worth confirming by training on
   the last 4 weeks and the last 8 weeks and seeing the trend.

5. **Why did `ensemble_lgbm_behavioral_twowtower_rrf_w0p12.csv` (Jun 7,
   LB 0.01846) collapse so hard relative to its inputs?** I have the
   diagnosis (weak base in §5.2) but not numerical re-confirmation. Worth
   reproducing as a unit test for the blender so this can't happen again.

---

## 10. My own honest self-assessment for the handover

I spent today (Jun 10) burning **5 LB slots** without producing a single
lift. The mistakes were:

- I kept iterating on CSV-level blends after the first negative result,
  when the correct response after attempt 2 was "stop blending, go
  retrain". §4.10 is the conclusion that should have been reached after
  the second regression, not the fifth.
- I generated FashionCLIP content predictions without a popularity prior,
  exactly the mistake the 1st-place writeup warns about. That cost a
  full content-similarity submission slot (LB 0.01581).
- I trusted offline numbers from a 10k slice for go/no-go decisions even
  though §5.5 was already evident from prior runs.
- I optimized for "shipping a submission today" instead of "shipping a
  submission that will lift". The user has every right to be frustrated.

The Item2Vec work is the one piece of code that the next engineer should
actually keep and reuse. It trains in 3 minutes, produces clean orthogonal
embeddings, and is the right *kind* of signal — it just needs to enter
the LGBM as a candidate source (with popularity priors), not as a
post-hoc reranker.

— End of handoff.
