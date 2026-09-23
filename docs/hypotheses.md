# Hypotheses

## H1 — Semantic subgroup hidden failure

OOD errors are not uniformly distributed across semantic subgroups. Some coherent groups exhibit substantially worse AUROC and/or FPR95 than the aggregate result suggests.

## H2 — Prompt sensitivity predicts failure

For sample \(x\) and prompt-specific score \(s_p(x)\):

\[
\mu(x)=\operatorname{mean}_p s_p(x)
\]

\[
\sigma(x)=\operatorname{std}_p s_p(x)
\]

Samples or groups with larger \(\sigma(x)\) are expected to have higher OOD failure rates.

## H3 — Robust aggregation can reduce worst-group failure

Candidate score:

\[
S_{robust}(x)=\mu(x)-\lambda\sigma(x)
\]

Alternative: use a lower prompt-score quantile.

\(\lambda\) must be selected using a limited validation condition and never tuned on final test OOD data.

## Null / competing hypotheses

### H0-1

Aggregate and subgroup performance are not meaningfully different.

### H0-2

Prompt sensitivity has no reproducible relationship with OOD failure.

### Alternative explanation

Observed subgroup failure may be explained by generic dataset difficulty or ID–OOD visual similarity rather than semantic structure itself.

## Falsification principle

The hypotheses are fixed before examining final test outcomes. If subgroup failure does not persist across reasonable subgroup definitions and confidence intervals, H1 should be rejected rather than reformulated post hoc.
