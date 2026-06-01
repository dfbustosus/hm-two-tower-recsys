---
name: hm-competition
description: Use for H&M Kaggle competition rules, dataset contract, MAP@12 objective, and submission-format decisions.
license: MIT
compatibility: opencode
metadata:
  domain: recommender-systems
  competition: h-and-m-personalized-fashion-recommendations
---

# H&M Competition Skill

Use this skill whenever work touches the competition objective, data contract, validation target, or submission format.

## Objective

Predict which `article_id` values each `customer_id` will purchase in the 7-day period immediately after the training data ends.

## Evaluation

The competition uses MAP@12. Each customer can receive up to 12 ranked predictions. There is no penalty for providing 12 predictions when the customer buys fewer than 12 items, so final submissions should backfill to 12 valid article IDs whenever possible.

Customers with no purchases in the hidden test period are excluded from scoring, but predictions must still be made for all customers in `sample_submission.csv`.

## Required Files

- `transactions_train.csv`: purchase history with duplicates allowed for repeated purchases.
- `articles.csv`: article metadata.
- `customers.csv`: customer metadata.
- `sample_submission.csv`: authoritative customer list and required submission shape.
- `images/`: optional article images, not guaranteed for every item.

## Submission Contract

CSV header:

```csv
customer_id,prediction
```

Rules:

- One row per required customer.
- `prediction` is a single string of space-separated article IDs.
- Use no more than 12 article IDs per customer.
- Preserve leading zeroes in every ID.
- Remove duplicate article IDs within each prediction string.

## Competition Hygiene

- Do not use data from after the training period for offline validation features or model training.
- Do not redistribute Kaggle data or images in the repository.
- If using external data for a competition submission, it must be public, free, and equally available to all competitors.
