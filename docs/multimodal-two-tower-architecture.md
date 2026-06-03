# Multimodal Two-Tower Retrieval Architecture Decision Record

## Status

Accepted as the next challenger-design direction. Implementation must proceed in
small, measured slices and must not replace the current learned-ranker champion
unless it improves leakage-safe candidate recall or downstream MAP@12.

## Context

The current repository champion is a leakage-safe learned linear ranker over
repeat, popularity, all-time popularity, and co-visitation candidates. Full
rolling validation showed learned ranking with co-visitation improved over the
deterministic ranker on all three checked windows. A validated Kaggle submission
has already been generated from this champion.

The next professional milestone is not “train a neural model because it is
modern.” The correct goal is to add **image-aware and two-tower retrieval
challengers** that increase candidate recall and/or downstream MAP@12 when
blended into the existing multi-source retrieval/ranking pipeline.

H&M is a fashion task with structured metadata, product descriptions, and article
images. Images are therefore a first-class signal, but they must be evaluated as
retrieval/ranking features rather than assumed beneficial.

## Evidence Reviewed

### H&M competition evidence

- The official Kaggle task asks for article recommendations using transactions,
  customer metadata, article metadata, text descriptions, and garment images.
- A public H&M silver-medal solution ranked 45/3006 reports:
  - two recall strategies,
  - three rankers per strategy: LGB ranker, LGB classifier, and DNN,
  - ensembling because recall strategies were complementary,
  - average roughly 50 candidates per user under RAM constraints,
  - pre-trained embeddings including DSSM, YouTube-style, word2vec, and
    `image_embd.npy`.
  Source: <https://github.com/Wp-Zhang/H-M-Fashion-RecSys>
- Other public H&M writeups emphasize a two-stage architecture: multi-source
  retrieval followed by LightGBM/LambdaRank-style ranking with behavioral and
  source features.
  Source: <https://github.com/Spyrunite/hm-fashion-recommendations>

Implication: for this competition, strong retrieval and ranking structure matter
more than the most fashionable model family. Image embeddings are plausible and
used publicly, but they should enter as measured candidate sources/features.

### Industry retrieval evidence

- YouTube’s industrial recommender architecture separates candidate generation
  from ranking and uses deep candidate generation to retrieve a small candidate
  set before ranking.
  Source: <https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/>
- TensorFlow Recommenders’ retrieval tutorial describes the standard two-tower
  setup: a query tower and candidate tower produce same-dimensional embeddings;
  dot products score affinity; retrieval evaluates top-K metrics and exports ANN
  indices for serving.
  Source:
  <https://raw.githubusercontent.com/tensorflow/recommenders/main/docs/examples/basic_retrieval.ipynb>
- Google’s recommendation-system guidance states that large-scale retrieval uses
  nearest-neighbor search over user/item embeddings and may use ANN systems such
  as ScaNN. It also highlights negative sampling and hard negatives to avoid
  embedding “folding.”
  Sources:
  <https://developers.google.com/machine-learning/recommendation/dnn/retrieval>,
  <https://developers.google.com/machine-learning/recommendation/dnn/training>
- Faiss is a mature ANN library for dense-vector similarity search with support
  for exact, approximate, inner-product, L2, GPU, and large-scale indexes.
  Source: <https://faiss.ai/>

Implication: retrieval evaluation must track recall@K independently from MAP@12,
and the two-tower model must export item embeddings plus exact article-ID
mappings before it can become a candidate source.

### Image and multimodal evidence

- CLIP learns image/text representations using contrastive image-text training
  and can transfer to new visual tasks, but it has limitations on fine-grained
  and domain-specific distinctions.
  Source: <https://openai.com/index/clip/>
- SigLIP replaces softmax contrastive normalization with pairwise sigmoid loss,
  improving scaling behavior and smaller-batch efficiency for language-image
  pretraining.
  Source: <https://arxiv.org/abs/2303.15343>
- FashionCLIP is a fashion-domain CLIP adaptation trained on fashion image-text
  pairs. The Hugging Face model card reports improved zero-shot fashion benchmark
  performance over OpenAI CLIP and includes important limitations: fashion-product
  image bias, gender/fashion-text bias, and model-selection caveats.
  Source: <https://huggingface.co/patrickjohncyh/fashion-clip>

Implication: use FashionCLIP or a strong open CLIP/SigLIP model as cached article
image/text embeddings, but evaluate them locally. Do not assume image embeddings
improve MAP@12 globally; they may help cold-ish items, sparse users, and visual
substitution/complementarity slices.

## Decision

Build a **multimodal retrieval ladder** rather than one monolithic two-tower
model:

1. Keep the current repeat/popularity/co-visitation/learned-ranker champion.
2. Add cached article image/text embeddings as local artifacts.
3. Add a standalone image/text ANN candidate source and evaluate candidate recall.
4. Add a two-tower retrieval challenger that can consume structured article
   metadata and frozen image/text embeddings.
5. Blend two-tower and image candidates into the existing candidate table with
   source names and rerank with the champion ranker or a stronger ranker.

The first image-aware source should be **standalone content retrieval**, not the
two-tower itself. This gives a fast, interpretable ablation: do visual/text
neighbors add candidate recall beyond repeat, popularity, and co-visitation?

## Architecture

### Artifact layout

All generated artifacts remain ignored by git.

```text
models/embeddings/articles/{provider_slug}/
  article_embeddings.npy
  article_ids.csv
  manifest.json

models/indexes/articles/{provider_slug}/
  faiss_or_exact_index.*
  article_ids.csv
  manifest.json

artifacts/multimodal-retrieval/
  image_text_retrieval_cutoff_*.json
  candidates_*.csv

artifacts/two-tower/
  examples_*.csv
  customer_mapping_*.csv
  article_mapping_*.csv
  report_*.json

models/two-tower/{run_id}/
  model checkpoint / weights
  item_embeddings.npy
  customer_embeddings.npy or query export metadata
  article_ids.csv
  customer_ids.csv
  manifest.json
```

### Image/text embedding cache

Article embedding provider interface must record:

- provider name and model ID, e.g. `fashion-clip`, `openai-clip-vit-b32`,
  `siglip-base-patch16`;
- model revision/hash when available;
- embedding dimension;
- preprocessing settings: image size, normalization, prompt template, text fields;
- article ID mapping order;
- number of articles, missing images, failed images, and skipped articles;
- runtime and hardware notes.

Image path resolution must follow the H&M layout: image subfolders are named by
the first three digits of the article ID. Missing images are normal and must not
break the pipeline.

Recommended article text prompt:

```text
{prod_name}. {product_type_name}. {product_group_name}. {graphical_appearance_name}.
{colour_group_name}. {perceived_colour_value_name}. {department_name}.
{index_group_name}. {garment_group_name}. {detail_desc}
```

### Standalone image/text retrieval source

For each target customer at cutoff:

1. Build a customer content query vector from recent pre-cutoff purchases:
   - weighted average of purchased article image/text embeddings;
   - weights by recency and repeat count;
   - optional normalization to unit length.
2. Query the article ANN/exact index for top `K_content` items.
3. Exclude or down-rank articles already in the customer’s recent history only as
   an ablation. H&M has repeat-purchase structure, so repeats should not be
   blindly removed.
4. Emit candidate rows:

```text
customer_id,article_id,source,source_rank,source_score
```

Candidate source names should distinguish modality:

- `image_similarity`
- `text_similarity`
- `multimodal_similarity`

Evaluate source-only and blended recall@12/50/100/200 before any ranker changes.

### Two-tower challenger

The two-tower model is a retrieval challenger, not a submission model by itself.

#### Query/customer tower inputs

Start simple, then ablate:

- customer ID embedding;
- customer metadata: age bucket, club status, fashion-news setting;
- recent purchase sequence summary:
  - last N article ID embeddings,
  - recency-weighted mean article embedding,
  - optional category/color/garment preference counts;
- price preference summary and channel counts;
- active-user recency features.

#### Item/article tower inputs

Start simple, then ablate:

- article ID embedding;
- article categorical metadata:
  product type/group, department, index group, section, garment group, color,
  perceived color, graphical appearance;
- item age/popularity features computed strictly before cutoff;
- frozen text embedding projection;
- frozen image embedding projection;
- optional multimodal embedding projection.

#### Representation and scoring

- Output shared dimension: start 64 or 128; tune later.
- Normalize final query/item vectors and use dot product/cosine with learnable or
  configured temperature.
- Export item embeddings with exact row-index-to-article-ID mapping.
- Retrieve at depths 50, 100, 200, and maybe 500 for recall diagnostics.

#### Loss and negatives

Professional default sequence:

1. In-batch softmax / sampled softmax on implicit positive pairs.
2. Add random negatives from pre-cutoff known articles for smoke tests.
3. Add popularity-weighted negatives with sampling-bias/logit correction if used
   for softmax training.
4. Add hard negatives from:
   - co-visitation neighbors not bought in target window,
   - image/text nearest neighbors,
   - high-scoring two-tower retrieval misses from previous epoch/run.

Negative sampling must exclude customer’s pre-cutoff positives. Validation-window
transactions must not affect negative pools, article eligibility, or features.

### Ranking integration

Two-tower and image candidates enter the same candidate table as existing sources.
The ranker should receive:

- source indicators;
- source rank/score;
- two-tower score;
- image/text similarity score;
- article/customer/interaction features;
- candidate-source ablation flags.

The final submission still needs full backfill to 12 valid, non-duplicate
article IDs per customer.

## Validation Protocol

Use the existing temporal contract:

```text
train.t_dat < cutoff
cutoff <= validation.t_dat < cutoff + 7 days
```

Required metrics:

- retrieval recall@12, @50, @100, @200 by source;
- blended candidate recall at the same depths;
- coverage for all target customers;
- recall slices:
  - no-history customers,
  - sparse-history customers,
  - dense-history customers,
  - customers with image-covered history,
  - customers with missing-image history,
  - cold-ish/low-history articles;
- downstream MAP@12 after ranking;
- runtime and artifact sizes.

Promotion criteria:

- Do not promote image or two-tower retrieval unless it improves combined
  candidate recall or downstream MAP@12 on comparable rolling windows.
- A two-tower source that improves recall but hurts MAP must remain a candidate
  source for ranker ablation, not a champion.
- Public leaderboard feedback cannot replace offline rolling validation.

## Implementation Plan

### Step 1: Finish two-tower export foundation

Keep the pure CSV export already started:

- cutoff-safe unique positive pairs with `positive_count`;
- stable string-ID mappings;
- deterministic random negatives;
- JSON summary.

This remains useful for smoke training, but is not enough for SOTA retrieval.

### Step 2: Add image inventory and embedding manifests

Implement local-only image inventory:

- map `article_id` to image path when available;
- report missing images and malformed paths;
- do not load all images in tests;
- write ignored JSON under `artifacts/multimodal/`.

### Step 3: Add embedding provider adapters

Add optional providers behind contracts:

- `fashion-clip` as the primary fashion-domain provider;
- `clip` or `siglip` as alternatives for ablation;
- optional local CPU/GPU execution;
- no hard dependency in the base package unless the provider is used.

### Step 4: Add image/text ANN retrieval source

- Cache article embeddings and mappings.
- Build exact cosine index first; Faiss/ScaNN optional after correctness.
- Export `image_similarity` / `text_similarity` / `multimodal_similarity`
  candidate rows.
- Run candidate diagnostics and rolling recall ablations.

### Step 5: Add minimal two-tower training smoke

- Use exported examples and frozen content embeddings.
- Train small retrieval model on one cutoff.
- Export item embeddings and retrieve top-K candidates.
- Compare recall to existing sources before any submission path.

### Step 6: Add hard negatives and ranker integration

- Mine hard negatives from co-visitation/content/two-tower misses.
- Add two-tower and image scores to ranker features.
- Evaluate rolling MAP@12 deltas and ablations.

## Non-Goals

- No direct LLM-generated top-12 recommendations.
- No committing images, embedding arrays, indexes, checkpoints, or submissions.
- No claim that two-tower is SOTA until it wins offline ablations.
- No copying public Kaggle solution code; use public writeups only as design
  evidence.

## References

- H&M public silver-medal solution: <https://github.com/Wp-Zhang/H-M-Fashion-RecSys>
- Public two-stage H&M writeup: <https://github.com/Spyrunite/hm-fashion-recommendations>
- YouTube DNN recommendations: <https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/>
- TensorFlow Recommenders retrieval tutorial:
  <https://raw.githubusercontent.com/tensorflow/recommenders/main/docs/examples/basic_retrieval.ipynb>
- TensorFlow Recommenders feature preprocessing:
  <https://raw.githubusercontent.com/tensorflow/recommenders/main/docs/examples/featurization.ipynb>
- Google recommendation retrieval guidance:
  <https://developers.google.com/machine-learning/recommendation/dnn/retrieval>
- Google negative sampling guidance:
  <https://developers.google.com/machine-learning/recommendation/dnn/training>
- CLIP: <https://openai.com/index/clip/>
- SigLIP: <https://arxiv.org/abs/2303.15343>
- FashionCLIP model card: <https://huggingface.co/patrickjohncyh/fashion-clip>
- Faiss documentation: <https://faiss.ai/>
