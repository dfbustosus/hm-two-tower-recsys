---
name: recommender-validation
description: Use for temporal validation, MAP@12 tests, candidate recall checks, and leakage prevention in recommender workflows.
license: MIT
compatibility: opencode
metadata:
  domain: recommender-systems
  workflow: evaluation
---

# Recommender Validation Skill

Use this skill when designing or reviewing splits, metrics, and offline evaluation.

## Temporal Split Pattern

For H&M, a strong default is:

- Train features and models on transactions before the validation target window.
- Use the next 7 days as validation labels.
- Exclude customers with no validation purchases from MAP@12 scoring, while still generating predictions for diagnostics.
- Repeat across rolling windows before trusting model changes.

## MAP@12 Checklist

- A hit counts only the first time a relevant article appears in the prediction list.
- Duplicate predictions should be removed before scoring or treated as invalid output.
- Cut predictions to 12.
- Average precision is normalized by `min(number_of_actual_items, 12)`.
- Customers with empty actual sets are excluded from the final mean when mirroring Kaggle.

## Diagnostics

Track:

- MAP@12 for final ranked predictions.
- Recall@12, recall@50, and recall@100 for candidate generation.
- Recall by candidate source and cumulative recall after adding each source.
- Coverage: customers with 12 predictions, items represented, and cold/customer-history groups.
- Failure slices for cold customers, sparse customers, dense customers, and low-history articles.
- Baseline comparison against recent global popularity and customer repeat purchases.

## Leakage Red Flags

- Features computed with transactions inside the validation target window.
- Popularity counts that include validation purchases.
- Customer histories updated after the split date.
- Article availability assumptions derived from future purchases.
