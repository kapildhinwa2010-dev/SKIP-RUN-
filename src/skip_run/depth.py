"""Depth fakeness test: skip every k-th block and watch the behavior.

A deep stack of blocks may contain a lot of depth that does almost
nothing. SKIP-RUN runs a stack of additive blocks twice: once
complete, and once with every k-th block removed. If the output
barely changes, that depth was cosmetic. Pure NumPy.
"""

import numpy as np


def run_stream(blocks, x):
    """Run a stack of additive blocks: x = x + block(x)."""
    for block in blocks:
        x = x + block(x)
    return x


def skipped_indices(num_blocks, k):
    """Indices removed when every k-th block is skipped (1-based)."""
    if k < 1:
        raise ValueError("k must be at least 1")
    return [i for i in range(num_blocks) if (i + 1) % k == 0]


def skip_sweep(blocks, x, score_fn, k_values=(1, 2, 3, 4)):
    """Run the stack with every k-th block removed, for each k."""
    if not blocks:
        raise ValueError("blocks must be a non-empty list")
    x = np.asarray(x, dtype=float)
    baseline = run_stream(blocks, x)
    base_score = float(score_fn(baseline))
    rows = []
    for k in k_values:
        removed = skipped_indices(len(blocks), k)
        kept = [b for i, b in enumerate(blocks) if i not in removed]
        out = run_stream(kept, x)
        score = float(score_fn(out))
        rows.append({
            "k": int(k),
            "removed": removed,
            "score": score,
            "delta": abs(base_score - score),
            "retained_fraction": len(kept) / len(blocks),
        })
    return {"baseline_score": base_score, "sweep": rows}


def real_depth(report, tolerance=0.05):
    """Largest k whose skip still keeps the score within tolerance."""
    if not isinstance(report, dict) or "sweep" not in report:
        raise ValueError("report must come from skip_sweep")
    safe = [row["k"] for row in report["sweep"] if row["delta"] <= tolerance]
    return int(max(safe)) if safe else None