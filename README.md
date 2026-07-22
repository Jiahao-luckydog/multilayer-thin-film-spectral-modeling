# Multilayer Thin-Film Spectral Modeling

Reproducible numerical experiments for an ideal multilayer dielectric stack. The repository studies forward modeling, solver cross-checks, material-dispersion sensitivity, parameter ambiguity and Fisher-information-based measurement design.

## Research questions

1. Can two independently organized transfer-matrix solvers reproduce the same reflectance spectrum and satisfy energy conservation?
2. How strongly do dispersion, angle, polarization and coupled layer-thickness changes affect the observable spectrum?
3. When different parameter combinations produce nearly indistinguishable spectra, which angle-polarization conditions improve identifiability?
4. How well do local Cramer-Rao predictions agree with Monte Carlo estimates in the tested neighborhood?

## Repository structure

- `src/`: transfer-matrix and material-dispersion utilities.
- `experiments/`: baseline, physics/identifiability and Fisher-design experiments.
- `tests/`: output and numerical acceptance checks.
- `requirements.txt`: minimal Python dependencies.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Run the baseline

```bash
python experiments/project_a_tmm_study.py --output-dir outputs/baseline
python tests/verify_project_a.py --run-dir outputs/baseline
```

## Run the dispersion and identifiability study

```bash
python experiments/project_a_physics_v2.py --output-dir outputs/physics_v2 --run-origin research_run
python tests/verify_project_a_v2.py --run-dir outputs/physics_v2
```

## Run the Fisher-information study

```bash
python experiments/project_a_information_design_p1.py --output-dir outputs/fisher_p1 --run-origin research_run
python tests/verify_project_a_p1.py --run-dir outputs/fisher_p1
```

## Scope and limitations

- The models assume planar, isotropic layers and ideal interfaces.
- Bulk-material dispersion equations are sensitivity models, not guaranteed optical constants for deposited films.
- Numerical agreement between solvers is verification, not experimental validation.
- COMSOL Wave Optics and measured reflectance or ellipsometry remain separate validation stages.
- Generated outputs may contain the local machine name in a run manifest; inspect outputs before publishing them.

## Reproducibility

Each experiment writes its configuration, summaries, figures, tabular data, hashes and review notes to a user-selected output directory. Randomized experiments use an explicit seed.

## Author

Wang Jiahao

## License

No open-source license has been selected yet. The code is publicly viewable, but reuse rights are not granted until a license is added.
