from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REQUIRED = [
    "00_config.json", "00_sources.json",
    "01_A01_solver_validation.csv", "01_A01_solver_validation.png",
    "02_A02_material_indices.csv", "02_A02_material_dispersion.png",
    "03_A02_dispersion_comparison.csv",
    "04_A03_angle_polarization_cube.csv", "03_A03_angle_polarization_cube.png",
    "05_A04_thickness_response_surface.csv", "04_A04_thickness_coupling.png",
    "06_A04_near_degenerate_pairs.csv", "07_A04_condition_comparison.csv",
    "05_A04_ambiguity_resolution.png", "08_summary.json", "run_manifest.json",
    "review_notes.md", "output_sha256.json",
]


def read_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    missing = [name for name in REQUIRED if not (run / name).exists()]
    summary = json.loads((run / "08_summary.json").read_text(encoding="utf-8")) if not missing else {}
    cube_rows = read_rows(run / "04_A03_angle_polarization_cube.csv") if not missing else []
    pair_rows = read_rows(run / "06_A04_near_degenerate_pairs.csv") if not missing else []
    condition_rows = read_rows(run / "07_A04_condition_comparison.csv") if not missing else []
    selected_conditions = {
        (row["angle_deg"], row["polarization"])
        for row in condition_rows if row.get("selected_for_universal_set") == "yes"
    }
    range_ok = all(
        -1e-9 <= float(row[key]) <= 1.0 + 1e-9
        for row in cube_rows for key in ("R", "T", "A")
    ) if cube_rows else False
    sum_error = max((abs(float(r["R"]) + float(r["T"]) + float(r["A"]) - 1.0) for r in cube_rows), default=999.0)
    checks = {
        "required_files_present": not missing,
        "solver_agreement": summary.get("A01", {}).get("max_solver_R_difference", 1.0) < 1e-10,
        "energy_conservation": summary.get("A01", {}).get("max_energy_error", 1.0) < 1e-10,
        "legacy_vectorized_agreement": summary.get("A01", {}).get("normal_incidence_vectorized_difference", 1.0) < 1e-10,
        "cube_expected_rows": len(cube_rows) == 14 * 186,
        "cube_R_T_A_range": range_ok,
        "cube_identity": sum_error < 1e-9,
        "thickness_grid_expected": summary.get("A04", {}).get("grid_samples") == 441,
        "near_degenerate_pairs_found": len(pair_rows) >= 2,
        "condition_search_table_complete": len(condition_rows) == 5 * 14,
        "three_universal_conditions_selected": len(selected_conditions) == 3,
        "all_difficult_pairs_improve": summary.get("A04", {}).get("minimum_separation_gain", 0.0) > 1.0,
    }
    report = {
        "schema": "project_a_physics_v2_verification_v1",
        "run_dir": str(run),
        "required_files": len(REQUIRED),
        "missing": missing,
        "checks": checks,
        "cube_rows": len(cube_rows),
        "near_degenerate_pairs": len(pair_rows),
        "selected_universal_conditions": sorted(selected_conditions),
        "max_cube_identity_error": sum_error,
        "passed": all(checks.values()),
        "note": "passed confirms the software, numeric, cross-check and output-structure gates.",
    }
    (run / "verification_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
