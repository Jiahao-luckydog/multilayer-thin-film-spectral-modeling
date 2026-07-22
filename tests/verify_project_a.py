from pathlib import Path
import argparse
import csv
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    required = [
        "01_angle_polarization_spectra.csv", "02_angle_summary.csv",
        "03_high_layer_thickness_sensitivity.csv", "04_monte_carlo_uncertainty.csv",
        "01_angle_polarization_spectra.png", "02_high_layer_thickness_sensitivity.png",
        "03_monte_carlo_uncertainty.png", "run_manifest.json", "output_sha256.json",
        "review_notes.md",
    ]
    missing = [name for name in required if not (args.run_dir / name).exists()]
    with (args.run_dir / "01_angle_polarization_spectra.csv").open("r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    numeric_ok = all(0.0 <= float(row[key]) <= 1.0 for row in rows for key in row if key.startswith("R_"))
    report = {"required_files": len(required), "missing": missing, "spectral_rows": len(rows), "reflectance_range_ok": numeric_ok,
              "passed": not missing and len(rows) >= 150 and numeric_ok,
              "note": "passed confirms only the required output structure and numeric ranges."}
    (args.run_dir / "verification_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
