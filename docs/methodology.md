# Mathematical Methodology For The H&M Recommender

This document formalizes the methodology used in this repository for the H&M
Personalized Fashion Recommendations task.  The project objective is not to
build the most complex model first; it is to build a leakage-safe, measurable,
multi-stage recommender whose final output is exactly 12 article recommendations
per customer and whose model changes are justified by temporal validation.

The current system is a staged retrieval-and-ranking pipeline:

```text
raw H&M data
→ leakage-safe temporal split
→ candidate generators
→ candidate diagnostics
→ ranker-ready feature table
→ deterministic / learned ranker
→ submission validation
```

Multimodal and two-tower models are treated as candidate-generation challengers.
They must improve candidate recall or downstream MAP@12 before promotion.

## 1. Problem Definition

Let

- $\mathcal{U}$ be the set of customers in `sample_submission.csv`,
- $\mathcal{I}$ be the set of valid articles in `articles.csv`,
- $\mathcal{T}$ be the transaction table,
- each transaction be a tuple
  $$
  (t, u, i, p, c) \in \mathcal{T},
  $$
  where $t$ is the purchase date, $u \in \mathcal{U}$ is a customer,
  $i \in \mathcal{I}$ is an article, $p$ is price, and $c$ is sales
  channel.

For every customer $u \in \mathcal{U}$, the system must output an ordered list

$$
\hat{Y}_u = (\hat{i}_{u,1}, \hat{i}_{u,2}, \ldots, \hat{i}_{u,K}),
\qquad K \le 12,
$$

with no duplicate article IDs.  In practice we backfill to exactly 12 predictions
when enough valid articles are available.

The hidden Kaggle target is the set of articles purchased by customer $u$ in
the 7-day period immediately after the training data ends.  We approximate this
with temporal validation windows.

## 2. Temporal Validation Contract

For a validation cutoff date $\tau$, define:

$$
\mathcal{T}^{\text{train}}_{\tau}
= \{(t,u,i,p,c) \in \mathcal{T}: t < \tau\},
$$

$$
\mathcal{T}^{\text{val}}_{\tau}
= \{(t,u,i,p,c) \in \mathcal{T}: \tau \le t < \tau + 7\text{ days}\}.
$$

The validation labels for customer $u$ are the unique articles purchased in the
validation window:

$$
Y_u^{\tau}
= \{i : (t,u,i,p,c) \in \mathcal{T}^{\text{val}}_{\tau}\}.
$$

All candidate generation, feature computation, negative sampling, popularity
counts, co-visitation counts, customer histories, ranker training, and two-tower
training for cutoff $\tau$ must use only $\mathcal{T}^{\text{train}}_{\tau}$.
This is the main leakage-prevention rule.

For scoring, we evaluate only customers with non-empty validation labels:

$$
\mathcal{U}^{\text{eval}}_{\tau}
= \{u \in \mathcal{U}: |Y_u^{\tau}| > 0\}.
$$

This mirrors Kaggle behavior, where customers with no hidden-period purchases do
not contribute to MAP@12.  The pipeline still generates predictions for the full
submission customer universe.

## 3. MAP@12 Objective

For one customer $u$, let $Y_u$ be the unique relevant article set and let
$\hat{Y}_u = (\hat{i}_{u,1}, \ldots, \hat{i}_{u,K})$ be the ordered predictions.
Only ranks $1,\ldots,12$ are considered.

Define the hit indicator at rank $r$:

$$
h_u(r) = \mathbf{1}\{\hat{i}_{u,r} \in Y_u\}
\mathbf{1}\{\hat{i}_{u,r} \notin \{\hat{i}_{u,1},\ldots,\hat{i}_{u,r-1}\}\}.
$$

Duplicates receive no additional credit and still consume rank positions.

Precision at rank $r$ is

$$
P_u(r) = \frac{\sum_{s=1}^{r} h_u(s)}{r}.
$$

Average precision at 12 is

$$
AP@12(u)
= \frac{1}{\min(|Y_u|, 12)}
\sum_{r=1}^{12} P_u(r) h_u(r),
$$

with $AP@12(u)=0$ if $|Y_u|=0$.

Mean average precision is

$$
MAP@12
= \frac{1}{|\mathcal{U}^{\text{eval}}_{\tau}|}
\sum_{u \in \mathcal{U}^{\text{eval}}_{\tau}} AP@12(u).
$$

MAP@12 is the final ranking metric.  Candidate generators are evaluated primarily
with recall at larger cutoffs because a ranker can only recover relevant items
that are present in the candidate set.

## 4. Candidate Recall And Coverage

Let $C_u^{(K)}$ be the top-$K$ candidate set for customer $u$, ignoring
candidate order for recall.  Candidate recall is

$$
Recall@K(u) = \frac{|Y_u \cap C_u^{(K)}|}{|Y_u|},
$$

and

$$
Recall@K
= \frac{1}{|\mathcal{U}^{\text{eval}}_{\tau}|}
\sum_{u \in \mathcal{U}^{\text{eval}}_{\tau}} Recall@K(u).
$$

We track at least $K \in \{12,50,100\}$.  For each source and source blend we
also track:

- customer coverage:
  $$
  \frac{|\{u : |C_u| > 0\}|}{|\mathcal{U}_{\text{target}}|},
  $$
- full-count coverage:
  $$
  \frac{|\{u : |C_u| \ge K\}|}{|\mathcal{U}_{\text{target}}|},
  $$
- article coverage:
  $$
  |\bigcup_u C_u|,
  $$
- duplicate candidate rows,
- candidate count distribution.

## 5. Baseline Candidate Sources

### 5.1 Recent Global Popularity

For lookback window $L$, define the recent training window

$$
W_{\tau,L} = \{(t,u,i,p,c) \in \mathcal{T}^{\text{train}}_{\tau}: t \ge \tau-L\}.
$$

The popularity count for article $i$ is

$$
g_{\tau,L}(i) = \sum_{(t,u,j,p,c) \in W_{\tau,L}} \mathbf{1}\{j=i\}.
$$

Articles are ranked by descending $g_{\tau,L}(i)$, then by recent date, then by
deterministic article ID tie-breaks.  This source is strong for H&M because the
hidden target week is highly recency- and trend-dependent.

### 5.2 All-Time Popularity

All-time popularity uses all pre-cutoff transactions:

$$
g_{\tau,\infty}(i)
= \sum_{(t,u,j,p,c) \in \mathcal{T}^{\text{train}}_{\tau}} \mathbf{1}\{j=i\}.
$$

It provides stable backfill and coverage.

### 5.3 Repeat Purchases

For each customer $u$, define their pre-cutoff purchase history:

$$
H_u^{\tau} = \{(t,i): (t,u,i,p,c) \in \mathcal{T}^{\text{train}}_{\tau}\}.
$$

For article $i$, define customer-specific repeat frequency and last purchase:

$$
f_u(i) = \sum_{(t,j) \in H_u^{\tau}} \mathbf{1}\{j=i\},
$$

$$
r_u(i) = \max\{t : (t,i) \in H_u^{\tau}\}.
$$

Repeat candidates are ranked by recent purchase, frequency, and deterministic ID
tie-breaks.  This source captures H&M's repeat-buying behavior.

### 5.4 Source Blending

Given ranked source lists $S_{u,1}, S_{u,2}, \ldots, S_{u,m}$, deterministic
blending traverses sources in a fixed order and keeps the first occurrence of
each article:

$$
B_u = \operatorname{dedupe}(S_{u,1} \Vert S_{u,2} \Vert \cdots \Vert S_{u,m}).
$$

The current baseline order is repeat candidates followed by recent popularity
and all-time popularity backfill.

## 6. Co-Visitation Item-Item Retrieval

Co-visitation models local item-item structure from pre-cutoff customer histories.
Let $R_u^{\tau}$ be the most recent $M$ unique articles purchased by customer
$u$, ordered newest first.

For two distinct articles $i$ and $j$, define a co-visitation count:

$$
A_{ij}
= \sum_{u}\mathbf{1}\{i \in R_u^{\tau}\}\mathbf{1}\{j \in R_u^{\tau}\},
\qquad i \ne j.
$$

For each source article $i$, keep the top $N$ neighbors ranked by
$A_{ij}$.  For a target customer $u$, aggregate neighbor scores over recent
history:

$$
s_{\text{covisit}}(u,j)
= \sum_{r=1}^{|R_u^{\tau}|}
\frac{1}{r} A_{i_r j},
$$

where $i_r$ is the article at history rank $r$.  The $1/r$ weight gives
more influence to recent purchases.

Co-visitation is evaluated as its own source and as part of deterministic blends.

## 7. Ranker-Ready Candidate Table

All candidate sources emit rows of the form:

$$
(u, i, \text{source}, \text{source\_rank}, \text{source\_score}).
$$

Multiple sources may emit the same $(u,i)$ pair.  The ranker feature builder
aggregates source rows into a single feature vector:

$$
x_{ui} = \phi(u,i,\mathcal{S}_{ui}),
$$

where $\mathcal{S}_{ui}$ is the set of source rows for that pair.  Features
include source indicators, source ranks, source scores, and derived rank/score
statistics.

Labels for ranker training on cutoff $\tau$ are

$$
y_{ui}^{\tau} = \mathbf{1}\{i \in Y_u^{\tau}\}.
$$

## 8. Deterministic Ranker

The deterministic ranker is a transparent scoring function:

$$
S_{\text{det}}(u,i)
= \sum_{s \in \mathcal{S}} w_s \mathbf{1}\{(u,i) \text{ emitted by } s\}
+ \sum_{s \in \mathcal{S}} a_s \frac{1}{\operatorname{rank}_s(u,i)}
+ \sum_{s \in \mathcal{S}} b_s\operatorname{score}_s(u,i).
$$

It is not learned from labels; it encodes a deterministic preference for strong
sources and good source ranks.  It is useful because every learned ranker must
beat it on comparable leakage-safe splits.

## 9. Learned Linear Ranker

The learned ranker is a logistic linear model over candidate features:

$$
P(y_{ui}=1\mid x_{ui}) = \sigma(w^\top x_{ui} + b),
$$

where

$$
\sigma(z) = \frac{1}{1+e^{-z}}.
$$

The objective is weighted binary cross entropy with L2 regularization:

$$
\mathcal{L}(w,b)
= -\sum_{(u,i)}
\alpha y_{ui}\log \sigma(z_{ui})
+ (1-y_{ui})\log(1-\sigma(z_{ui}))
+ \lambda \|w\|_2^2,
$$

where $z_{ui}=w^\top x_{ui}+b$ and $\alpha$ is a positive-class weight used
to address severe label imbalance.

To prevent leakage, when evaluating cutoff $\tau$, the learned ranker is
trained on the previous non-overlapping label window:

$$
[\tau-7, \tau)
\quad \rightarrow \quad
[\tau, \tau+7).
$$

The training window's features use only data before $\tau-7$, and the
evaluation window's features use only data before $\tau$.

## 10. Multimodal Article Embeddings

Each article $i$ has structured metadata, text fields, and optionally an image.
Let

$$
q_i^{\text{text}}
$$

be the normalized text prompt built from fields such as product name,
description, department, garment group, color, graphical appearance, and product
type.  Let

$$
q_i^{\text{image}}
$$

be the local article image when available.

An open-source encoder $E_\theta$, such as FashionCLIP, OpenCLIP, or SigLIP,
maps article content to vectors:

$$
e_i^{\text{text}} = E_\theta^{\text{text}}(q_i^{\text{text}}),
$$

$$
e_i^{\text{image}} = E_\theta^{\text{image}}(q_i^{\text{image}}).
$$

Vectors are L2-normalized:

$$
\tilde{e}_i = \frac{e_i}{\|e_i\|_2}.
$$

For multimodal embeddings, the current provider averages available text and
image vectors before final normalization:

$$
e_i^{\text{multi}}
= \frac{1}{|M_i|}\sum_{m \in M_i} \tilde{e}_i^{(m)},
$$

where $M_i \subseteq \{\text{text},\text{image}\}$ is the set of available
modalities.

Embedding caches are written under ignored local storage with manifests recording
provider name, model ID, model revision, dimension, preprocessing, license notes,
and exact row-index-to-article-ID mappings.

## 11. Content-Similarity Retrieval

Cached article embeddings define a content candidate source.  For a customer
$u$, let $R_u^{\tau}$ be their recent unique pre-cutoff history restricted to
articles with cached embeddings.  The customer content query vector is

$$
z_u
= \operatorname{norm}\left(
\frac{1}{|R_u^{\tau}|}
\sum_{i \in R_u^{\tau}} \tilde{e}_i
\right).
$$

Similarity to candidate article $j$ is cosine similarity:

$$
s_{\text{content}}(u,j)
= z_u^\top \tilde{e}_j.
$$

Historical articles may be excluded from recommendations:

$$
j \notin R_u^{\tau}.
$$

The current implementation uses exact search first.  Approximate nearest-neighbor
indexes such as FAISS or HNSW should only be introduced after exact retrieval
validates embeddings, ID mappings, and candidate metrics.

Content retrieval emits source rows such as:

```text
image_similarity
text_similarity
multimodal_similarity
```

It is not promoted unless it improves candidate recall or downstream MAP@12 on
the same temporal split.

## 12. Two-Tower Retrieval Challenger

The planned two-tower model learns a customer tower $f_\theta(u)$ and item tower
$g_\psi(i)$ in a shared embedding space:

$$
a_u = f_\theta(u, H_u^{\tau}, \text{customer metadata}),
$$

$$
b_i = g_\psi(i, \text{article metadata}, e_i^{\text{text}}, e_i^{\text{image}}).
$$

Retrieval score is dot product or cosine similarity:

$$
s_{\text{2tower}}(u,i) = a_u^\top b_i.
$$

For a batch $B=\{(u_b,i_b^+)\}_{b=1}^n$, in-batch softmax loss is

$$
\mathcal{L}_{\text{IB}}
= -\sum_{b=1}^{n}
\log
\frac{\exp(a_{u_b}^\top b_{i_b^+}/T)}
{\sum_{k=1}^{n}\exp(a_{u_b}^\top b_{i_k^+}/T)},
$$

where $T$ is a temperature.  Other negative strategies can be added:

- deterministic random negatives from known pre-cutoff articles,
- popularity-weighted negatives,
- same-category hard negatives,
- content-neighbor hard negatives.

Negative sampling must exclude the customer's pre-cutoff positives:

$$
i^- \notin \{i : (t,u,i,p,c) \in \mathcal{T}^{\text{train}}_{\tau}\}.
$$

The two-tower is a retrieval challenger.  It must export candidate rows and prove
candidate recall or MAP@12 gains before it can influence a final submission.

### 12.1 Research-Grade Two-Tower Upgrade Ladder

The repository should evaluate modern two-tower ideas by their expected impact on
H&M MAP@12, not by novelty alone.  The practical research ladder is:

1. **Sampling-bias and selection-bias control.**  Random, popularity-weighted,
   and mixed negative sampling are leakage-safe when sampling probabilities are
   computed only from $\mathcal{T}^{\text{train}}_{\tau}$.  LogQ-style correction
   adjusts logits as
   $$
   s^c(u,i)=a_u^\top b_i-\alpha\log q_\tau(i),
   $$
   where $q_\tau(i)$ is estimated from pre-cutoff sampled/training item
   frequencies.  In this project, LogQ is a tunable challenger because strong
   correction can improve deeper recall while hurting MAP@12.
2. **In-batch and cross-batch negatives.**  These are the correct next training
   objective once a batched trainer exists.  The validation contract must mask a
   user's known pre-cutoff positives from that user's negative denominator to
   limit false negatives.  Cross-batch memory queues must reset per split and be
   reproducible for fixed seeds.
3. **Feature-rich towers.**  ID-only towers have sparse-customer coverage and
   cold-item limitations.  Higher-value tower inputs for H&M are recent purchase
   summaries, age buckets, customer flags, product type/group, department, color,
   garment group, price/channel aggregates, and cached text/image embeddings.
4. **Hard negatives.**  Same-category, co-visitation, content-neighbor, and
   high-score-miss negatives are useful only if mined before the training cutoff.
   They should be mixed with uniform/popularity negatives rather than replacing
   them.
5. **Multi-vector or late-interaction retrieval.**  IntTower/FIT-style sum-max or
   lightweight similarity scoring can address the late-interaction bottleneck, but
   they are expensive and should follow a proven candidate-recall gap.  For H&M,
   multi-interest history vectors or per-category user vectors are likely simpler
   first steps than a full FIT implementation.
6. **OneBP-style updates.**  One Backpropagation is attractive for false-negative
   robustness in one-class collaborative filtering, but it requires a batched
   training loop and stable item-tower updates.  It is a second-stage optimization,
   not the first missing piece.
7. **LLM/generative retrieval.**  LEARN/TIGER/TTDS/URM-style methods are research
   candidates for content extraction or semantic indexing.  They are not a near
   term submission path unless they produce candidate-recall gains under the same
   temporal split and can generate exact valid `article_id` values.

The evidence so far supports this ranking: current candidate sets have a high
perfect-ranker upper bound relative to achieved MAP@12, so the immediate
engineering bottleneck is **fast ranking experimentation** and stronger tabular
ranker features, while two-tower improvements should be measured as candidate
sources rather than assumed to solve the task alone.

### 12.2 Fast Iteration Protocol

The default workflow for research experiments is:

1. Build or reuse ranker-ready candidate CSVs for a bounded previous-window
   tuning set and an evaluation set.
2. Load candidate features once, then evaluate a compact grid in memory.
3. Select weights only on the previous non-overlapping label window.
4. Apply the selected configuration once to full validation.
5. Promote only after rolling windows confirm the result.

For deterministic ranker research, use the fast grid rather than ad-hoc scripts:

```bash
make deterministic-ranker-tuning \
  CUTOFF=2020-09-16 \
  RANKER_CANDIDATE_K=100 \
  RANKER_MAX_TARGET_CUSTOMERS=10000 \
  INCLUDE_AGE_SEGMENT_POPULARITY=1 \
  INCLUDE_GARMENT_GROUP_POPULARITY=1 \
  INCLUDE_TWO_TOWER_RETRIEVAL=1 \
  DETERMINISTIC_TUNING_RESEARCH_GRID=1
```

This is intentionally a research harness, not an automatic submission gate.

## 13. Submission Construction

For each customer $u \in \mathcal{U}$, the final ranked list is produced by a
ranker or deterministic source blend and then backfilled:

$$
\hat{Y}_u
= \operatorname{dedupe}(R_u^{\text{ranker}} \Vert B_u^{\text{popularity}})_{1:12}.
$$

Submission validation enforces:

- exact header `customer_id,prediction`,
- exact customer set from `sample_submission.csv`,
- valid article IDs,
- no duplicate predictions per row,
- no more than 12 predictions,
- preserved string IDs and leading zeroes.

## 14. Experiment Governance

Every meaningful experiment should log:

- command and configuration,
- source commit or working-tree state,
- cutoff and validation window,
- row/customer/article counts,
- seed,
- feature and embedding versions,
- candidate-source recall at 12/50/100,
- MAP@12,
- coverage metrics,
- runtime,
- artifact paths.

Promotion requires a champion/challenger comparison on the same split.  For major
changes, rolling-window validation is required.  A modern model is not promoted
because it is modern; it is promoted only when it improves validated retrieval or
ranking metrics without breaking leakage, reproducibility, or submission
contracts.

## 15. Current Evidence And Interpretation

The current champion is a learned linear ranker over repeat, popularity, and
co-visitation candidate rows.  It improves over deterministic ranking on rolling
temporal windows.

The current multimodal implementation has successfully produced a bounded
FashionCLIP cache and evaluated it as `multimodal_similarity`.  With only 1,000
cached articles, coverage and recall are expectedly weak; this is a technical
smoke test, not a promoted model.  The next methodological step is to scale the
embedding cache to a larger article universe, then compare content-similarity
Recall@12/50/100 against the existing candidate sources before blending it into
the ranker.
