from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REQUIRED = [
    "00_config.json", "00_sources.json", "01_single_condition_fisher_metrics.csv",
    "02_all_three_condition_designs.csv", "03_selected_design_comparison.csv",
    "04_crlb_monte_carlo_validation.csv", "01_fisher_condition_landscape.png",
    "02_crlb_design_comparison.png", "03_crlb_monte_carlo_validation.png",
    "05_summary.json", "run_manifest.json", "review_notes.md", "output_sha256.json",
]


def rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser(); p.add_argument("--run-dir", type=Path, required=True); a = p.parse_args()
    run = a.run_dir.resolve(); missing = [x for x in REQUIRED if not (run / x).exists()]
    summary = json.loads((run / "05_summary.json").read_text(encoding="utf-8")) if not missing else {}
    singles = rows(run / "01_single_condition_fisher_metrics.csv") if not missing else []
    subsets = rows(run / "02_all_three_condition_designs.csv") if not missing else []
    mc = rows(run / "04_crlb_monte_carlo_validation.csv") if not missing else []
    designs = summary.get("designs", {})
    robust = designs.get("robust_D_optimal", {})
    baseline = designs.get("single_0s", {})
    ratios = [float(x["empirical_to_crlb"]) for x in mc]
    checks = {
        "required_files_present": not missing,
        "candidate_conditions_13": len(singles) == 13,
        "three_condition_combinations_286": len(subsets) == 286,
        "five_designs_compared": len(designs) == 5,
        "robust_design_has_three_conditions": len(robust.get("conditions", [])) == 3,
        "robust_logdet_exceeds_baseline": robust.get("logdet", -1e99) > baseline.get("logdet", 1e99),
        "robust_min_eigen_exceeds_baseline": robust.get("min_eigenvalue", -1) > baseline.get("min_eigenvalue", 1e99),
        "monte_carlo_nine_parameter_checks": len(mc) == 9,
        "monte_carlo_matches_crlb_scale": bool(ratios) and all(0.90 <= x <= 1.10 for x in ratios),
        "finite_metrics": all(np == np and abs(np) != float("inf") for row in singles for np in [float(row["logdet_F"]), float(row["min_eigenvalue"])]),
    }
    report = {"schema": "project_a_information_design_p1_verification_v1", "run_dir": str(run),
              "missing": missing, "checks": checks, "passed": all(checks.values()),
              "note": "passed confirms the P1 mathematical, numeric and output-structure gates."}
    (run / "verification_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2)); raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
