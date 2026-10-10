"""Mechanical verdict on the pre-registered SACP criteria (docs/algorithm.md).

The criteria were written before any real-data run. This module turns them into code so the
verdict is computed from `analyze.py` output, not argued after the fact. It reads
`report["shift_conformal"]` and returns SUPPORTED or NOT SUPPORTED with the reason for each rule.

Where the pre-registration left a detail open, the choice made here is the strict one and is
listed under `interpretation` in the result:
- "the best of cond_novelty and cond_entropy" -> SACP must satisfy criteria 1-3 against BOTH.
- criterion 3's confidence interval -> the paired-bootstrap interval on the set-size difference that
  `analyze.py` records for each held-out condition (nominal alpha); its mean upper bound must be < 0.
- "match its coverage at equal size" -> mean coverage >= SACP's and mean set size <= SACP's.
"""
from __future__ import annotations

import math

BASELINES = ("cond_novelty", "cond_entropy")
SPOILERS = ("raps", "weighted", "pooled_aug")
SIZE_TOLERANCE = 1.10   # criterion 2: no more than 10% larger
INTERPRETATION = [
    "'best of cond_novelty and cond_entropy' -> SACP must pass criteria 1-3 against both.",
    "criterion 3 interval -> paired-bootstrap CI of the set-size difference at nominal alpha; mean upper bound < 0.",
    "'match its coverage at equal size' -> mean coverage >= SACP's and mean set size <= SACP's.",
]


def _ci_upper_bounds(report: dict, base: str) -> list[float]:
    out = []
    for family in report["held_out"].values():
        for cond in family.values():
            out.append(cond["sacp"]["vs"][base]["size_ci"][1])
    return out


def sacp_verdict(shift_conformal: dict) -> dict:
    summ = shift_conformal["summary"]
    s = summ["sacp"]
    rules, ok = [], True

    for base in BASELINES:
        b = summ[base]
        c1 = s["worst_undercoverage"] <= b["worst_undercoverage"]
        c2 = s["mean_size"] <= SIZE_TOLERANCE * b["mean_size"]
        s_eff, b_eff = s["mean_size_at_target_coverage"], b["mean_size_at_target_coverage"]
        ups = _ci_upper_bounds(shift_conformal, base)
        mean_up = sum(ups) / len(ups) if ups else math.nan
        c3 = s_eff < b_eff and mean_up < 0
        for n, passed, detail in (
            (1, c1, f"worst under-coverage {s['worst_undercoverage']:.4f} vs {b['worst_undercoverage']:.4f}"),
            (2, c2, f"mean size {s['mean_size']:.3f} vs {b['mean_size']:.3f} (limit x{SIZE_TOLERANCE})"),
            (3, c3, f"size at 90% coverage {s_eff:.3f} vs {b_eff:.3f}; mean CI upper bound of size diff {mean_up:.3f}"),
        ):
            rules.append({"criterion": n, "against": base, "passed": bool(passed), "detail": detail})
            ok &= bool(passed)

    for sp in SPOILERS:
        if sp not in summ:
            continue
        m = summ[sp]
        matches = m["mean_coverage"] >= s["mean_coverage"] and m["mean_size"] <= s["mean_size"]
        rules.append({"criterion": "no spoiler", "against": sp, "passed": not matches,
                      "detail": f"coverage {m['mean_coverage']:.4f} / size {m['mean_size']:.3f} "
                                f"vs SACP {s['mean_coverage']:.4f} / {s['mean_size']:.3f}"})
        ok &= not matches

    return {"verdict": "SUPPORTED" if ok else "NOT SUPPORTED", "rules": rules,
            "interpretation": INTERPRETATION}
