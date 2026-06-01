## Summary

- What changed:
- Why it matters:
- Project layer affected:

## H&M Recommender Guardrails

- [ ] Customer and article IDs remain strings; leading zeroes are preserved.
- [ ] No validation or test-window data leaks into training features, candidates, negatives, or labels.
- [ ] Candidate generation and ranking changes report separate diagnostics when applicable.
- [ ] MAP@12 behavior is unchanged or explicitly tested and documented.
- [ ] Final prediction/submission logic still removes duplicate article IDs per customer.

## Data, Artifact, and Security Impact

- [ ] No raw Kaggle CSVs, article images, generated features, checkpoints, logs, or submissions are committed.
- [ ] No credentials, tokens, cookies, private URLs, or local machine paths are committed.
- [ ] External data or code is public, free, equally accessible, and license-compatible if used.
- [ ] New dependencies or GitHub Actions are justified and covered by Dependabot when possible.

## Validation

Commands or checks run:

```text

```

Relevant metrics, reports, or artifacts:

```text

```

## Review Notes

- Known risks:
- Follow-up work:
- Related issue or experiment:
