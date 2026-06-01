---
description: Implement or review a popularity/repeat-purchase baseline before deep modeling
agent: hm-recsys-orchestrator
---

Implement or review the simplest strong baseline for H&M recommendations:

- Recent global popularity.
- Customer repeat purchases ranked by recency/frequency.
- Backfill to 12 predictions.
- Offline validation through MAP@12.
- Submission generation checks.

Prefer small, testable modules and add metric/submission tests where useful.
