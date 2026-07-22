from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import socket
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "src"))
from tmm_model import quarter_wave_stack, reflectance_spectrum, spectral_summary


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def write_csv(path: Path, header, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Guided TMM thin-film project")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--periods", type=int, default=6)
    parser.add_argument("--design-wavelength-nm", type=float, default=550.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--monte-carlo-samples", type=int, default=200)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    wavelengths = np.arange(400.0, 801.0, 2.0)
    n_layers, d_layers = quarter_wave_stack(args.periods, args.design_wavelength_nm)

    curves = {}
    angle_rows = []
    for polarization in ("s", "p"):
        for angle in (0, 15, 30, 45):
            key = f"R_{polarization}_{angle}deg"
            curve = reflectance_spectrum(wavelengths, n_layers, d_layers, angle, polarization)
            curves[key] = curve
            summary = spectral_summary(wavelengths, curve)
            angle_rows.append([polarization, angle, *summary.values()])

    write_csv(out / "01_angle_polarization_spectra.csv",
              ["wavelength_nm", *curves.keys()],
              zip(wavelengths, *curves.values()))
    write_csv(out / "02_angle_summary.csv",
              ["polarization", "angle_deg", "peak_wavelength_nm", "peak_reflectance",
               "mean_reflectance_500_600", "bandwidth_above_0_9_nm"], angle_rows)

    plt.figure(figsize=(9, 5.2))
    for key, values in curves.items():
        if "0deg" in key or "30deg" in key or "45deg" in key:
            plt.plot(wavelengths, values, label=key.replace("R_", ""))
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.ylim(0, 1.03)
    plt.title("Angle and polarisation response of an ideal H/L stack")
    plt.grid(alpha=0.25)
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "01_angle_polarization_spectra.png", dpi=180)
    plt.close()

    thickness_rows = []
    plt.figure(figsize=(9, 5.2))
    for scale in (0.90, 0.95, 1.00, 1.05, 1.10):
        n_test, d_test = quarter_wave_stack(args.periods, args.design_wavelength_nm, high_scale=scale)
        curve = reflectance_spectrum(wavelengths, n_test, d_test, 0, "s")
        summary = spectral_summary(wavelengths, curve)
        thickness_rows.append([scale, *summary.values()])
        plt.plot(wavelengths, curve, label=f"H thickness x {scale:.2f}")
    write_csv(out / "03_high_layer_thickness_sensitivity.csv",
              ["high_layer_scale", "peak_wavelength_nm", "peak_reflectance",
               "mean_reflectance_500_600", "bandwidth_above_0_9_nm"], thickness_rows)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.ylim(0, 1.03)
    plt.title("Sensitivity to high-index-layer thickness")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "02_high_layer_thickness_sensitivity.png", dpi=180)
    plt.close()

    mc = []
    for _ in range(args.monte_carlo_samples):
        perturbation = rng.normal(1.0, 0.02, size=len(d_layers))
        mc.append(reflectance_spectrum(wavelengths, n_layers, d_layers * perturbation, 0, "s"))
    mc = np.asarray(mc)
    mean = np.mean(mc, axis=0)
    p05 = np.percentile(mc, 5, axis=0)
    p95 = np.percentile(mc, 95, axis=0)
    baseline = curves["R_s_0deg"]
    write_csv(out / "04_monte_carlo_uncertainty.csv",
              ["wavelength_nm", "baseline", "mean", "p05", "p95"],
              zip(wavelengths, baseline, mean, p05, p95))
    plt.figure(figsize=(9, 5.2))
    plt.fill_between(wavelengths, p05, p95, alpha=0.25, label="5th-95th percentile")
    plt.plot(wavelengths, baseline, linewidth=1.5, label="Nominal stack")
    plt.plot(wavelengths, mean, linewidth=1.0, linestyle="--", label="Monte Carlo mean")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.ylim(0, 1.03)
    plt.title("Uncertainty from 2% layer-thickness variation")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "03_monte_carlo_uncertainty.png", dpi=180)
    plt.close()

    manifest = {
        "schema": "guided_project_a_tmm_v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": "software_run_complete_pending_scientific_review",
        "claim_boundary": "The files document a numerical run; they do not constitute experimental validation.",
        "environment": {"python": sys.version, "platform": platform.platform(), "hostname": socket.gethostname()},
        "parameters": vars(args) | {"output_dir": str(out), "angles_deg": [0, 15, 30, 45], "polarizations": ["s", "p"], "thickness_sigma": 0.02},
        "model_limits": ["lossless isotropic layers", "ideal flat interfaces", "constant refractive indices", "no fabrication measurement"],
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "review_notes.md").write_text(
        "# Review notes\n\n"
        "1. 我亲自运行的时间：\n2. 我修改的参数及原因：\n3. 我观察到的三条规律：\n"
        "4. 我最初不理解、后来弄懂的概念：\n5. 一张我能独立解释的图：\n"
        "6. 当前模型的限制：\n7. 下一步实验或COMSOL验证计划：\n", encoding="utf-8")
    hashes = {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file()}
    (out / "output_sha256.json").write_text(json.dumps(hashes, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(out), "files": len(hashes), "status": manifest["status"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
