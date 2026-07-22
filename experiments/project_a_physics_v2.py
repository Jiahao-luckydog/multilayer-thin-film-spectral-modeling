from __future__ import annotations

import argparse
import csv
import hashlib
from itertools import combinations
import json
import platform
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "src"))
from tmm_model import normal_incidence_reflectance
from tmm_v2 import (
    MATERIALS,
    MaterialModel,
    characteristic_rt,
    constant_index,
    quarter_wave_thicknesses,
    recursive_reflectance,
    spectral_metrics,
    stack_indices,
)

plt.rcParams.update({
    "font.family": "Times New Roman",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 8,
    "figure.dpi": 120,
})


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def write_csv(path: Path, header, rows) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def save_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def constant_material(key: str, label: str, value: float) -> MaterialModel:
    return MaterialModel(
        key=key,
        label=label,
        source_url="project baseline at design wavelength",
        valid_range_nm=(400.0, 800.0),
        function=constant_index(value),
        limitation="Constant-index numerical baseline, not a material claim.",
    )


def pairwise_rmse(x: np.ndarray) -> np.ndarray:
    norms = np.sum(x * x, axis=1)
    d2 = (norms[:, None] + norms[None, :] - 2.0 * (x @ x.T)) / x.shape[1]
    return np.sqrt(np.maximum(d2, 0.0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Project A v2 physics and identifiability study")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--periods", type=int, default=6)
    parser.add_argument("--design-wavelength-nm", type=float, default=550.0)
    parser.add_argument("--wavelength-start-nm", type=float, default=430.0)
    parser.add_argument("--wavelength-end-nm", type=float, default=800.0)
    parser.add_argument("--wavelength-step-nm", type=float, default=2.0)
    parser.add_argument("--thickness-grid-points", type=int, default=21)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-origin", choices=["precheck", "research_run"], default="precheck")
    args = parser.parse_args()
    started = time.perf_counter()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    wavelengths = np.arange(
        args.wavelength_start_nm,
        args.wavelength_end_nm + 0.5 * args.wavelength_step_nm,
        args.wavelength_step_nm,
    )
    angles = [0, 10, 20, 30, 40, 50, 60]
    polarizations = ["s", "p"]
    high = MATERIALS["tio2_rutile_devore"]
    low = MATERIALS["sio2_malitson"]
    n_high_design = float(np.real(high.evaluate([args.design_wavelength_nm])[0]))
    n_low_design = float(np.real(low.evaluate([args.design_wavelength_nm])[0]))
    thicknesses = quarter_wave_thicknesses(
        args.periods, args.design_wavelength_nm, n_high_design, n_low_design
    )
    indices_disp = stack_indices(wavelengths, args.periods, high, low)
    high_const = constant_material("high_constant", "High-index constant baseline", n_high_design)
    low_const = constant_material("low_constant", "Low-index constant baseline", n_low_design)
    indices_const = stack_indices(wavelengths, args.periods, high_const, low_const)

    config = {
        "schema": "project_a_physics_v2_config_v1",
        "periods": args.periods,
        "design_wavelength_nm": args.design_wavelength_nm,
        "wavelengths_nm": {
            "start": args.wavelength_start_nm,
            "end": args.wavelength_end_nm,
            "step": args.wavelength_step_nm,
            "count": len(wavelengths),
        },
        "angles_deg": angles,
        "polarizations": polarizations,
        "materials": {"high": high.key, "low": low.key},
        "design_indices": {"high": n_high_design, "low": n_low_design},
        "quarter_wave_thickness_nm": {
            "high": float(thicknesses[0]), "low": float(thicknesses[1])
        },
        "substrate_index": 1.52,
        "thickness_grid": {"min_scale": 0.90, "max_scale": 1.10, "points": args.thickness_grid_points},
        "seed": args.seed,
        "run_origin": args.run_origin,
    }
    save_json(out / "00_config.json", config)
    save_json(out / "00_sources.json", {
        "TiO2": {
            "citation": "J. R. DeVore, Refractive Indices of Rutile and Sphalerite, JOSA 41, 416-419 (1951).",
            "url": high.source_url,
            "use": "Ordinary-index dispersion of bulk rutile as a high-dispersion stress test.",
            "limitation": high.limitation,
        },
        "SiO2": {
            "citation": "I. H. Malitson, Interspecimen Comparison of the Refractive Index of Fused Silica, JOSA 55, 1205-1209 (1965).",
            "url": low.source_url,
            "use": "Sellmeier dispersion of fused silica.",
            "limitation": low.limitation,
        },
        "TMM": {
            "citation": "S. J. Byrnes, Multilayer optical calculations, arXiv:1603.02720.",
            "url": "https://arxiv.org/abs/1603.02720",
            "use": "Definition and implementation cross-check reference.",
        },
    })

    # A01: independent solver agreement and lossless energy conservation.
    validation_rows = []
    all_matrix = {}; all_recursive = {}; all_T = {}; all_A = {}
    for pol in polarizations:
        for angle in [0, 30, 45, 60]:
            key = f"{pol}_{angle}deg"
            R, T, A = characteristic_rt(wavelengths, indices_const, thicknesses, angle, pol)
            R2 = recursive_reflectance(wavelengths, indices_const, thicknesses, angle, pol)
            all_matrix[key], all_recursive[key], all_T[key], all_A[key] = R, R2, T, A
            validation_rows.append([
                pol, angle, float(np.max(np.abs(R - R2))),
                float(np.sqrt(np.mean((R - R2) ** 2))),
                float(np.max(np.abs(R + T - 1.0))),
                float(np.min(R)), float(np.max(R)), float(np.min(T)), float(np.max(T)),
            ])
    vectorized = normal_incidence_reflectance(
        wavelengths, np.tile([n_high_design, n_low_design], args.periods), thicknesses
    )
    vectorized_difference = float(np.max(np.abs(vectorized - all_matrix["s_0deg"])))
    write_csv(out / "01_A01_solver_validation.csv",
              ["polarization", "angle_deg", "max_abs_R_difference", "R_rmse",
               "max_abs_energy_error", "R_min", "R_max", "T_min", "T_max"], validation_rows)
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 7.1), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    axes[0].plot(wavelengths, all_matrix["s_45deg"], label="Characteristic matrix")
    axes[0].plot(wavelengths, all_recursive["s_45deg"], "--", label="Fresnel recursion")
    axes[0].set_ylabel("Reflectance"); axes[0].set_ylim(0, 1.02); axes[0].legend(); axes[0].grid(alpha=.25)
    diff = np.maximum(np.abs(all_matrix["s_45deg"] - all_recursive["s_45deg"]), 1e-17)
    axes[1].semilogy(wavelengths, diff, color="#B22222")
    axes[1].set_xlabel("Wavelength (nm)"); axes[1].set_ylabel("Absolute difference"); axes[1].grid(alpha=.25)
    fig.suptitle("A01 Independent TMM solver cross-check (s, 45 deg)")
    fig.tight_layout(); fig.savefig(out / "01_A01_solver_validation.png", dpi=200); plt.close(fig)

    # A02: material dispersion versus design-index constants.
    high_n = np.real(high.evaluate(wavelengths)); low_n = np.real(low.evaluate(wavelengths))
    R_const, T_const, _ = characteristic_rt(wavelengths, indices_const, thicknesses, 0, "s")
    R_disp, T_disp, A_disp = characteristic_rt(wavelengths, indices_disp, thicknesses, 0, "s")
    write_csv(out / "02_A02_material_indices.csv", ["wavelength_nm", "n_TiO2_rutile", "n_SiO2_fused"], zip(wavelengths, high_n, low_n))
    write_csv(out / "03_A02_dispersion_comparison.csv",
              ["wavelength_nm", "R_constant", "T_constant", "R_dispersive", "T_dispersive", "A_dispersive", "R_difference"],
              zip(wavelengths, R_const, T_const, R_disp, T_disp, A_disp, R_disp - R_const))
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 7.3), sharex=True)
    axes[0].plot(wavelengths, high_n, label="TiO2 rutile (ordinary)")
    axes[0].plot(wavelengths, low_n, label="Fused silica")
    axes[0].axvline(args.design_wavelength_nm, color="gray", linestyle=":")
    axes[0].set_ylabel("Refractive index"); axes[0].legend(); axes[0].grid(alpha=.25)
    axes[1].plot(wavelengths, R_const, label="Constant at 550 nm")
    axes[1].plot(wavelengths, R_disp, "--", label="Dispersive material models")
    axes[1].set_xlabel("Wavelength (nm)"); axes[1].set_ylabel("Reflectance"); axes[1].set_ylim(0,1.02); axes[1].legend(); axes[1].grid(alpha=.25)
    fig.suptitle("A02 Material-dispersion stress test")
    fig.tight_layout(); fig.savefig(out / "02_A02_material_dispersion.png", dpi=200); plt.close(fig)

    # A03: wavelength-angle-polarization data cube.
    cube_rows = []; cube = {}
    for pol in polarizations:
        cube[pol] = []
        for angle in angles:
            R, T, A = characteristic_rt(wavelengths, indices_disp, thicknesses, angle, pol)
            cube[pol].append(R)
            metrics = spectral_metrics(wavelengths, R)
            for lam, rv, tv, av in zip(wavelengths, R, T, A):
                cube_rows.append([lam, angle, pol, rv, tv, av])
    write_csv(out / "04_A03_angle_polarization_cube.csv",
              ["wavelength_nm", "angle_deg", "polarization", "R", "T", "A"], cube_rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, pol in zip(axes, polarizations):
        image = ax.imshow(np.asarray(cube[pol]), aspect="auto", origin="lower",
                          extent=[wavelengths[0], wavelengths[-1], angles[0], angles[-1]],
                          vmin=0, vmax=1, cmap="viridis")
        ax.set_title(f"{pol.upper()} polarization"); ax.set_xlabel("Wavelength (nm)")
    axes[0].set_ylabel("Angle (deg)"); fig.colorbar(image, ax=axes, label="Reflectance", shrink=.88)
    fig.suptitle("A03 Wavelength-angle-polarization reflectance cube")
    fig.subplots_adjust(left=.08, right=.91, bottom=.13, top=.84, wspace=.08)
    fig.savefig(out / "03_A03_angle_polarization_cube.png", dpi=200); plt.close(fig)

    # A04: coupled thickness response and near-degenerate spectra.
    scales = np.linspace(0.90, 1.10, args.thickness_grid_points)
    spectra = []; params = []; surface_rows = []
    for hs in scales:
        for ls in scales:
            d = quarter_wave_thicknesses(args.periods, args.design_wavelength_nm,
                                         n_high_design, n_low_design, hs, ls)
            R, _, _ = characteristic_rt(wavelengths, indices_disp, d, 0, "s")
            metrics = spectral_metrics(wavelengths, R)
            spectra.append(R); params.append([hs, ls])
            surface_rows.append([hs, ls, float(d[0]), float(d[1]), *metrics.values(),
                                 float(np.sqrt(np.mean((R - R_disp) ** 2)))])
    spectra = np.asarray(spectra); params = np.asarray(params)
    write_csv(out / "05_A04_thickness_response_surface.csv",
              ["high_scale", "low_scale", "dH_nm", "dL_nm", "peak_wavelength_nm", "peak_reflectance",
               "mean_reflectance_500_600", "bandwidth_above_0_9_nm", "rmse_from_nominal"], surface_rows)
    distances = pairwise_rmse(spectra)
    param_distance = np.sqrt(((params[:, None, :] - params[None, :, :]) ** 2).sum(axis=2))
    invalid = np.tril(np.ones_like(distances, dtype=bool)) | (param_distance < 0.05)
    distances[invalid] = np.inf
    candidates = np.argsort(distances, axis=None)
    selected = []; used = set()
    for flat in candidates:
        i, j = np.unravel_index(flat, distances.shape)
        if not np.isfinite(distances[i, j]): break
        if i in used or j in used: continue
        selected.append((i, j)); used.update([i, j])
        if len(selected) == 5: break
    # Instead of hand-picking extra observations, choose one universal three-condition
    # set for all difficult pairs. The objective is maximin: improve the hardest pair
    # first, then use mean gain as the tie-breaker. Baseline 0 deg/s is always retained.
    available_conditions = [(int(angle), pol) for angle in angles for pol in polarizations]
    baseline_condition = (0, "s")
    condition_rmse_by_pair = []
    for i, j in selected:
        rmse_map = {}
        for angle, pol in available_conditions:
            da = quarter_wave_thicknesses(args.periods, args.design_wavelength_nm, n_high_design, n_low_design, *params[i])
            db = quarter_wave_thicknesses(args.periods, args.design_wavelength_nm, n_high_design, n_low_design, *params[j])
            ra, _, _ = characteristic_rt(wavelengths, indices_disp, da, angle, pol)
            rb, _, _ = characteristic_rt(wavelengths, indices_disp, db, angle, pol)
            rmse_map[(angle, pol)] = float(np.sqrt(np.mean((ra - rb) ** 2)))
        condition_rmse_by_pair.append(rmse_map)

    best_selection = None
    best_score = (-np.inf, -np.inf)
    extras = [condition for condition in available_conditions if condition != baseline_condition]
    for extra_pair in combinations(extras, 2):
        trial = (baseline_condition, *extra_pair)
        trial_gains = []
        for pair_index, (i, j) in enumerate(selected):
            fused = float(np.sqrt(np.mean([condition_rmse_by_pair[pair_index][c] ** 2 for c in trial])))
            trial_gains.append(fused / max(float(distances[i, j]), 1e-15))
        score = (float(np.min(trial_gains)), float(np.mean(trial_gains)))
        if score > best_score:
            best_score = score
            best_selection = trial
    fusion_conditions = best_selection or (baseline_condition,)

    pair_rows = []; condition_rows = []; gains = []
    for rank, (i, j) in enumerate(selected, 1):
        single_rmse = float(distances[i, j])
        rmse_map = condition_rmse_by_pair[rank - 1]
        for angle, pol in available_conditions:
            condition_rows.append([rank, angle, pol, rmse_map[(angle, pol)],
                                   "yes" if (angle, pol) in fusion_conditions else "no"])
        fused_rmse = float(np.sqrt(np.mean([rmse_map[c] ** 2 for c in fusion_conditions])))
        gain = fused_rmse / max(single_rmse, 1e-15); gains.append(gain)
        pair_rows.append([rank, *params[i], *params[j], float(param_distance[i, j]), single_rmse, fused_rmse, gain])
    write_csv(out / "06_A04_near_degenerate_pairs.csv",
              ["rank", "high_scale_A", "low_scale_A", "high_scale_B", "low_scale_B", "parameter_distance",
               "single_0deg_s_rmse", "fused_rmse", "separation_gain"], pair_rows)
    write_csv(out / "07_A04_condition_comparison.csv",
              ["pair_rank", "angle_deg", "polarization", "spectral_rmse", "selected_for_universal_set"], condition_rows)
    peak_map = np.asarray([r[4] for r in surface_rows]).reshape(len(scales), len(scales))
    nominal_map = np.asarray([r[-1] for r in surface_rows]).reshape(len(scales), len(scales))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    im0=axes[0].imshow(peak_map,origin="lower",extent=[scales[0],scales[-1],scales[0],scales[-1]],aspect="auto",cmap="plasma")
    axes[0].set_title("Peak wavelength (nm)"); axes[0].set_xlabel("Low-layer scale"); axes[0].set_ylabel("High-layer scale"); fig.colorbar(im0,ax=axes[0])
    im1=axes[1].imshow(nominal_map,origin="lower",extent=[scales[0],scales[-1],scales[0],scales[-1]],aspect="auto",cmap="magma_r")
    axes[1].set_title("Spectral RMSE from nominal"); axes[1].set_xlabel("Low-layer scale"); axes[1].set_ylabel("High-layer scale"); fig.colorbar(im1,ax=axes[1])
    fig.suptitle("A04 Coupled high/low thickness response")
    fig.tight_layout(); fig.savefig(out / "04_A04_thickness_coupling.png", dpi=200); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    if selected:
        i, j = selected[0]
        axes[0].plot(wavelengths, spectra[i], label=f"A: H {params[i,0]:.2f}, L {params[i,1]:.2f}")
        axes[0].plot(wavelengths, spectra[j], "--", label=f"B: H {params[j,0]:.2f}, L {params[j,1]:.2f}")
        axes[0].set_xlabel("Wavelength (nm)"); axes[0].set_ylabel("Reflectance"); axes[0].set_ylim(0,1.02); axes[0].legend(); axes[0].grid(alpha=.25)
        axes[1].bar(np.arange(1, len(gains)+1), gains, color="#2F75B5")
        axes[1].axhline(1, color="black", linewidth=.8); axes[1].set_xlabel("Near-degenerate pair rank"); axes[1].set_ylabel("Fused / single-condition RMSE")
        axes[1].set_title("Separation gain from extra conditions"); axes[1].grid(axis="y",alpha=.25)
    fig.suptitle("A04 Near-degenerate parameters and multi-condition separation")
    fig.tight_layout(); fig.savefig(out / "05_A04_ambiguity_resolution.png", dpi=200); plt.close(fig)

    summary = {
        "schema": "project_a_physics_v2_summary_v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": "engineering_precheck" if args.run_origin == "precheck" else "research_run_pending_review",
        "A01": {
            "max_solver_R_difference": max(r[2] for r in validation_rows),
            "max_energy_error": max(r[4] for r in validation_rows),
            "normal_incidence_vectorized_difference": vectorized_difference,
        },
        "A02": {
            "TiO2_n_range": [float(np.min(high_n)), float(np.max(high_n))],
            "SiO2_n_range": [float(np.min(low_n)), float(np.max(low_n))],
            "max_abs_R_change_from_dispersion": float(np.max(np.abs(R_disp - R_const))),
            "peak_constant_nm": spectral_metrics(wavelengths, R_const)["peak_wavelength_nm"],
            "peak_dispersive_nm": spectral_metrics(wavelengths, R_disp)["peak_wavelength_nm"],
        },
        "A03": {"spectral_rows": len(cube_rows), "conditions": len(angles) * len(polarizations)},
        "A04": {
            "grid_samples": len(surface_rows),
            "near_degenerate_pairs": len(pair_rows),
            "best_single_condition_rmse": float(pair_rows[0][6]) if pair_rows else None,
            "selection_objective": "maximize minimum separation gain, then maximize mean gain",
            "selected_universal_conditions": [
                {"angle_deg": angle, "polarization": pol} for angle, pol in fusion_conditions
            ],
            "minimum_separation_gain": float(np.min(gains)) if gains else None,
            "mean_separation_gain": float(np.mean(gains)) if gains else None,
            "max_separation_gain": float(np.max(gains)) if gains else None,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": "This run verifies software and numerical experiments; it does not constitute experimental validation.",
        "limitations": [
            high.limitation,
            low.limitation,
            "Planar isotropic layers with ideal interfaces; no roughness, scattering or measured sample.",
            "No COMSOL result is claimed in this run.",
        ],
    }
    save_json(out / "08_summary.json", summary)
    manifest = {
        "schema": "project_a_physics_v2_manifest_v1",
        "generated_at": summary["generated_at"],
        "environment": {"python": sys.version, "platform": platform.platform(), "hostname": socket.gethostname()},
        "script": str(HERE), "output_dir": str(out), "parameters": vars(args), "status": summary["status"],
    }
    manifest["parameters"]["output_dir"] = str(out)
    save_json(out / "run_manifest.json", manifest)
    (out / "review_notes.md").write_text(
        "# Project A v2 review notes\n\n"
        "1. 我亲自运行的日期与时间：\n"
        "2. 我确认的材料模型及其局限：\n"
        "3. A01两种求解器为什么要一致：\n"
        "4. A02色散模型相对固定折射率改变了什么：\n"
        "5. A03哪两个角度或偏振提供了互补信息：\n"
        "6. A04第一组近似同谱异参是什么，为什么单条件难区分：\n"
        "7. 加入多条件后分离增益是多少，我如何解释：\n"
        "8. 我认为当前模型最重要的三个限制：\n"
        "9. 我下一步准备修改的一个参数及原因：\n",
        encoding="utf-8",
    )
    hashes = {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != "output_sha256.json"}
    save_json(out / "output_sha256.json", hashes)
    print(json.dumps({"output_dir": str(out), "summary": summary, "files": len(hashes)+1}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
