"""Apply the pre-registered SACP criteria (docs/algorithm.md) to an analyze.py report.

    python scripts/sacp_verdict.py runs/analysis_vit_mae_s0.json
"""
import json
import sys

from terraforge.training.preregistered import sacp_verdict


def main():
    report = json.load(open(sys.argv[1]))
    v = sacp_verdict(report["shift_conformal"])
    for r in v["rules"]:
        print(f"[{'pass' if r['passed'] else 'FAIL'}] criterion {r['criterion']} vs {r['against']}: {r['detail']}")
    print("interpretation choices:", *v["interpretation"], sep="\n  - ")
    print(f"\nVERDICT: {v['verdict']}")


if __name__ == "__main__":
    main()
