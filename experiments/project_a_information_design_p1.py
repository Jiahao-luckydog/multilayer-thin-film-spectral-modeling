from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import socket
import sys
import time
from datetime import datetime
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "src"))
from tmm_v2 import MATERIALS, characteristic_rt, quarter_wave_thicknesses, stack_indices

plt.rcParams.update({"font.family": "Times New Roman", "figure.dpi": 120})


def write_csv(path: Path, header, rows) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


def save_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def model_spectrum(wavelengths, condition, theta, periods, design_wavelength_nm):
    """theta = [dH_scale, dL_scale, nH_scale], all dimensionless."""
    angle, pol = condition
    high = MATERIALS["tio2_rutile_devore"]
    low = MATERIALS["sio2_malitson"]
    n_high_design = float(np.real(high.evaluate([design_wavelength_nm])[0]))
    n_low_design = float(np.real(low.evaluate([design_wavelength_nm])[0]))
    thickness = quarter_wave_thicknesses(
        periods, design_wavelength_nm, n_high_design, n_low_design, theta[0], theta[1]
    )
    indices = stack_indices(wavelengths, periods, high, low)
    indices = indices.copy()
    indices[:, 1:-1:2] *= theta[2]
    R, _, _ = characteristic_rt(wavelengths, indices, thickness, angle, pol)
    return R


def jacobian_and_fisher(wavelengths, condition, theta, steps, periods, design_wavelength_nm):
    base = model_spectrum(wavelengths, condition, theta, periods, design_wavelength_nm)
    cols = []
    for k, h in enumerate(steps):
        plus = theta.copy(); minus = theta.copy()
        plus[k] += h; minus[k] -= h
        cols.append((model_spectrum(wavelengths, condition, plus, periods, design_wavelength_nm)
                     - model_spectrum(wavelengths, condition, minus, periods, design_wavelength_nm)) / (2 * h))
    J = np.column_stack(cols)
    sigma = 0.003 + 0.007 * np.sqrt(np.clip(base * (1 - base), 0, None))
    F = J.T @ (J / sigma[:, None] ** 2)
    return base, J, sigma, F


def fim_metrics(F):
    eigenvalues = np.linalg.eigvalsh((F + F.T) / 2)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    sign, logdet = np.linalg.slogdet(F + np.eye(F.shape[0]) * 1e-12)
    positive = eigenvalues[eigenvalues > max(eigenvalues.max(initial=0) * 1e-14, 1e-12)]
    condition = float(positive.max() / positive.min()) if len(positive) == F.shape[0] else float("inf")
    covariance = np.linalg.pinv(F, rcond=1e-12)
    return {
        "logdet": float(logdet if sign > 0 else -np.inf),
        "min_eigenvalue": float(eigenvalues.min()),
        "condition_number": condition,
        "crlb_std_dH_scale": float(np.sqrt(max(covariance[0, 0], 0))),
        "crlb_std_dL_scale": float(np.sqrt(max(covariance[1, 1], 0))),
        "crlb_std_nH_scale": float(np.sqrt(max(covariance[2, 2], 0))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Project A P1 Fisher-information experimental design")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--periods", type=int, default=6)
    parser.add_argument("--design-wavelength-nm", type=float, default=550.0)
    parser.add_argument("--monte-carlo", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-origin", choices=["precheck", "research_run"], default="precheck")
    args = parser.parse_args()
    started = time.perf_counter(); out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    wavelengths = np.arange(430.0, 800.0 + 1e-9, 2.0)
    conditions = [(0, "s")] + [(a, p) for a in range(10, 61, 10) for p in ("s", "p")]
    theta0 = np.array([1.0, 1.0, 1.0])
    steps = np.array([0.002, 0.002, 0.001])
    scenario_thetas = [np.array([hs, ls, 1.0]) for hs in (0.95, 1.0, 1.05) for ls in (0.95, 1.0, 1.05)]

    save_json(out / "00_config.json", {
        "schema": "project_a_information_design_p1_config_v1", "periods": args.periods,
        "design_wavelength_nm": args.design_wavelength_nm, "wavelength_nm": [430, 800, 2],
        "parameters": ["dH_scale", "dL_scale", "nH_scale"], "finite_difference_steps": steps.tolist(),
        "noise_model": "sigma_R = 0.003 + 0.007*sqrt(R*(1-R))", "candidate_conditions": conditions,
        "subset_size": 3, "robust_scenarios": [x.tolist() for x in scenario_thetas], "seed": args.seed,
    })
    save_json(out / "00_sources.json", {
        "Byrnes_TMM": "https://arxiv.org/abs/1603.02720",
        "Fisher_reflectometry": "https://doi.org/10.1107/S1600576721003126",
        "Bayesian_scatterometry": "https://arxiv.org/abs/1707.08467",
        "DeVore_TiO2": "https://doi.org/10.1364/JOSA.41.000416",
        "Malitson_SiO2": "https://doi.org/10.1364/JOSA.55.001205",
    })

    condition_cache = {}
    sensitivity_rows = []
    for condition in conditions:
        base, J, sigma, F = jacobian_and_fisher(wavelengths, condition, theta0, steps, args.periods, args.design_wavelength_nm)
        condition_cache[condition] = (base, J, sigma, F)
        m = fim_metrics(F)
        sensitivity_rows.append([condition[0], condition[1], *[m[k] for k in (
            "logdet", "min_eigenvalue", "condition_number", "crlb_std_dH_scale", "crlb_std_dL_scale", "crlb_std_nH_scale")]])
    write_csv(out / "01_single_condition_fisher_metrics.csv",
              ["angle_deg", "polarization", "logdet_F", "min_eigenvalue", "condition_number",
               "crlb_std_dH_scale", "crlb_std_dL_scale", "crlb_std_nH_scale"], sensitivity_rows)

    scenario_fims = {}
    for sidx, theta in enumerate(scenario_thetas):
        for condition in conditions:
            scenario_fims[(sidx, condition)] = jacobian_and_fisher(
                wavelengths, condition, theta, steps, args.periods, args.design_wavelength_nm
            )[3]

    subset_rows = []
    candidate_records = []
    for subset in combinations(conditions, 3):
        F_nom = sum((condition_cache[c][3] for c in subset), np.zeros((3, 3)))
        nominal = fim_metrics(F_nom)
        scenario_metrics = [fim_metrics(sum((scenario_fims[(sidx, c)] for c in subset), np.zeros((3, 3))))
                            for sidx in range(len(scenario_thetas))]
        worst_logdet = min(m["logdet"] for m in scenario_metrics)
        worst_min_eig = min(m["min_eigenvalue"] for m in scenario_metrics)
        record = {"subset": subset, "nominal": nominal, "worst_logdet": worst_logdet, "worst_min_eig": worst_min_eig}
        candidate_records.append(record)
        subset_rows.append([" | ".join(f"{a}deg-{p}" for a, p in subset), nominal["logdet"],
                            nominal["min_eigenvalue"], nominal["condition_number"], worst_logdet, worst_min_eig])
    write_csv(out / "02_all_three_condition_designs.csv",
              ["conditions", "nominal_logdet", "nominal_min_eigenvalue", "nominal_condition_number",
               "worst_scenario_logdet", "worst_scenario_min_eigenvalue"], subset_rows)

    d_opt = max(candidate_records, key=lambda x: x["nominal"]["logdet"])
    e_opt = max(candidate_records, key=lambda x: x["nominal"]["min_eigenvalue"])
    robust = max(candidate_records, key=lambda x: (x["worst_logdet"], x["worst_min_eig"]))
    p0_set = ((0, "s"), (50, "s"), (60, "s"))
    named = {
        "single_0s": ((0, "s"),), "P0_spectral_maximin": p0_set,
        "D_optimal_nominal": d_opt["subset"], "E_optimal_nominal": e_opt["subset"],
        "robust_D_optimal": robust["subset"],
    }
    comparison_rows = []
    comparison = {}
    for name, subset in named.items():
        F = sum((condition_cache[c][3] for c in subset), np.zeros((3, 3)))
        m = fim_metrics(F); comparison[name] = {"conditions": subset, **m}
        comparison_rows.append([name, " | ".join(f"{a}deg-{p}" for a, p in subset), *[m[k] for k in (
            "logdet", "min_eigenvalue", "condition_number", "crlb_std_dH_scale", "crlb_std_dL_scale", "crlb_std_nH_scale")]])
    write_csv(out / "03_selected_design_comparison.csv",
              ["design", "conditions", "logdet_F", "min_eigenvalue", "condition_number",
               "crlb_std_dH_scale", "crlb_std_dL_scale", "crlb_std_nH_scale"], comparison_rows)

    # Linearized Monte Carlo checks whether the predicted CRLB scale agrees with repeated noisy observations.
    mc_rows = []
    mc_errors = {}
    for name in ("single_0s", "P0_spectral_maximin", "robust_D_optimal"):
        subset = named[name]
        Js = np.vstack([condition_cache[c][1] for c in subset])
        sigmas = np.concatenate([condition_cache[c][2] for c in subset])
        F = Js.T @ (Js / sigmas[:, None] ** 2)
        gain = np.linalg.pinv(F, rcond=1e-12) @ (Js.T / sigmas[None, :] ** 2)
        noise = rng.normal(size=(args.monte_carlo, len(sigmas))) * sigmas
        errors = noise @ gain.T
        mc_errors[name] = errors
        crlb = np.sqrt(np.maximum(np.diag(np.linalg.pinv(F, rcond=1e-12)), 0))
        empirical = errors.std(axis=0, ddof=1)
        for k, parameter in enumerate(("dH_scale", "dL_scale", "nH_scale")):
            mc_rows.append([name, parameter, crlb[k], empirical[k], empirical[k] / max(crlb[k], 1e-15)])
    write_csv(out / "04_crlb_monte_carlo_validation.csv",
              ["design", "parameter", "crlb_std", "empirical_std", "empirical_to_crlb"], mc_rows)

    single = np.asarray(sensitivity_rows, dtype=object)
    labels = [f"{a}°-{p}" for a, p in conditions]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].bar(np.arange(len(labels)), single[:, 3].astype(float), color="#2F75B5")
    axes[0].set_xticks(np.arange(len(labels)), labels, rotation=55, ha="right")
    axes[0].set_ylabel("Minimum eigenvalue of F")
    axes[0].set_title("Worst identifiable parameter direction")
    axes[0].grid(axis="y", alpha=.25)
    axes[1].bar(np.arange(len(labels)), single[:, 2].astype(float), color="#70AD47")
    axes[1].set_xticks(np.arange(len(labels)), labels, rotation=55, ha="right")
    axes[1].set_ylabel("log det(F)")
    axes[1].set_title("Total local information")
    axes[1].grid(axis="y", alpha=.25)
    fig.suptitle("A-P1 Fisher information of candidate measurements")
    fig.tight_layout(); fig.savefig(out / "01_fisher_condition_landscape.png", dpi=200); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    design_names = list(named)
    for k, (metric, title) in enumerate((("crlb_std_dH_scale", "dH scale"), ("crlb_std_dL_scale", "dL scale"), ("crlb_std_nH_scale", "nH scale"))):
        vals = [comparison[n][metric] for n in design_names]
        axes[k].barh(np.arange(len(vals)), vals, color="#5B9BD5")
        axes[k].set_yticks(np.arange(len(vals)), design_names if k == 0 else [])
        axes[k].invert_yaxis(); axes[k].set_xlabel("CRLB standard deviation"); axes[k].set_title(title); axes[k].grid(axis="x", alpha=.25)
    fig.suptitle("A-P1 Predicted parameter precision by measurement design")
    fig.tight_layout(); fig.savefig(out / "02_crlb_design_comparison.png", dpi=200); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for k, parameter in enumerate(("dH scale error", "dL scale error", "nH scale error")):
        for name, errors in mc_errors.items():
            axes[k].hist(errors[:, k], bins=45, density=True, alpha=.42, label=name)
        axes[k].set_title(parameter); axes[k].grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.suptitle("A-P1 Linearized Monte Carlo validation of uncertainty scale")
    fig.tight_layout(); fig.savefig(out / "03_crlb_monte_carlo_validation.png", dpi=200); plt.close(fig)

    summary = {
        "schema": "project_a_information_design_p1_summary_v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": "engineering_precheck" if args.run_origin == "precheck" else "research_run_pending_review",
        "research_question": "How should angle-polarization measurements be selected to improve joint identifiability of dH, dL and high-index dispersion scale?",
        "designs": {k: {"conditions": [list(x) for x in v["conditions"]], **{mk: mv for mk, mv in v.items() if mk != "conditions"}} for k, v in comparison.items()},
        "robust_design_worst_scenario_logdet": robust["worst_logdet"],
        "monte_carlo_repetitions": args.monte_carlo,
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": [
            "Fisher and CRLB results are local and depend on the stated noise model.",
            "Monte Carlo uses a linearized estimator, not a full nonlinear posterior sampler.",
            "Optical constants represent literature bulk models, not a measured deposited film.",
            "No fabricated sample or instrument measurement is claimed.",
        ],
    }
    save_json(out / "05_summary.json", summary)
    manifest_parameters = vars(args).copy()
    manifest_parameters["output_dir"] = str(out)
    save_json(out / "run_manifest.json", {
        "schema": "project_a_information_design_p1_manifest_v1", "generated_at": summary["generated_at"],
        "environment": {"python": sys.version, "platform": platform.platform(), "hostname": socket.gethostname()},
        "parameters": manifest_parameters, "output_dir": str(out), "status": summary["status"],
    })
    (out / "review_notes.md").write_text(
        "# Project A P1 review notes\n\n"
        "1. 我亲自运行的时间：\n2. 三个待估参数分别是什么：\n3. 雅可比矩阵每一列表示什么：\n"
        "4. Fisher矩阵为什么等于加权灵敏度内积：\n5. D-opt与E-opt分别优化什么：\n"
        "6. 稳健设计为什么要看9个厚度场景中的最差情况：\n7. CRLB能说明什么、不能说明什么：\n"
        "8. 蒙特卡洛经验标准差与CRLB是否一致：\n9. 我认为下一步最重要的模型误差是：\n",
        encoding="utf-8")
    stable = [p for p in out.iterdir() if p.is_file() and p.name != "output_sha256.json"]
    save_json(out / "output_sha256.json", {p.name: sha256(p) for p in sorted(stable)})
    print(json.dumps({"output_dir": str(out), "robust_design": summary["designs"]["robust_D_optimal"],
                      "elapsed_seconds": summary["elapsed_seconds"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
