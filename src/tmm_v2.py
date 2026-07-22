"""Validated transfer-matrix utilities for the project-A v2 experiments.

All wavelengths and thicknesses use nanometres.  The module implements two
independent reflectance solvers: a characteristic-matrix method and a Fresnel
recursion.  Agreement between them is used as a numerical cross-check.

The bundled TiO2 model represents the ordinary refractive index of bulk rutile
reported by DeVore (1951), not a deposited thin-film guarantee.  The fused
silica model uses the Malitson (1965) Sellmeier equation.  Real deposited films
can differ because of phase, porosity, roughness, temperature and process.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

IndexModel = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class MaterialModel:
    key: str
    label: str
    source_url: str
    valid_range_nm: tuple[float, float]
    function: IndexModel
    limitation: str

    def evaluate(self, wavelengths_nm: Iterable[float]) -> np.ndarray:
        wavelengths = np.asarray(wavelengths_nm, dtype=float)
        if np.any((wavelengths < self.valid_range_nm[0]) | (wavelengths > self.valid_range_nm[1])):
            raise ValueError(
                f"{self.key} model is restricted to {self.valid_range_nm[0]}-"
                f"{self.valid_range_nm[1]} nm in this project"
            )
        values = np.asarray(self.function(wavelengths), dtype=complex)
        if values.shape != wavelengths.shape or not np.all(np.isfinite(values)):
            raise ValueError(f"{self.key} returned invalid refractive indices")
        return values


def constant_index(value: complex) -> IndexModel:
    def model(wavelengths_nm: np.ndarray) -> np.ndarray:
        return np.full_like(np.asarray(wavelengths_nm, dtype=float), value, dtype=complex)
    return model


def fused_silica_malitson(wavelengths_nm: np.ndarray) -> np.ndarray:
    """Malitson Sellmeier equation; wavelength is converted to micrometres."""
    lam_um = np.asarray(wavelengths_nm, dtype=float) / 1000.0
    lam2 = lam_um**2
    n2 = (
        1.0
        + 0.6961663 * lam2 / (lam2 - 0.0684043**2)
        + 0.4079426 * lam2 / (lam2 - 0.1162414**2)
        + 0.8974794 * lam2 / (lam2 - 9.896161**2)
    )
    return np.sqrt(n2).astype(complex)


def rutile_tio2_devore(wavelengths_nm: np.ndarray) -> np.ndarray:
    """DeVore ordinary-index dispersion for bulk rutile; lambda in micrometres."""
    lam_um = np.asarray(wavelengths_nm, dtype=float) / 1000.0
    n2 = 5.913 + 0.2441 / (lam_um**2 - 0.0803)
    return np.sqrt(n2).astype(complex)


MATERIALS = {
    "sio2_malitson": MaterialModel(
        key="sio2_malitson",
        label="Fused silica (Malitson Sellmeier)",
        source_url="https://doi.org/10.1364/JOSA.55.001205",
        valid_range_nm=(400.0, 800.0),
        function=fused_silica_malitson,
        limitation="Bulk fused-silica dispersion; deposited SiO2 films may differ.",
    ),
    "tio2_rutile_devore": MaterialModel(
        key="tio2_rutile_devore",
        label="Rutile TiO2 ordinary index (DeVore)",
        source_url="https://doi.org/10.1364/JOSA.41.000416",
        valid_range_nm=(430.0, 800.0),
        function=rutile_tio2_devore,
        limitation=(
            "Bulk anisotropic rutile model with absorption neglected here; it is a "
            "dispersion stress-test model, not a deposited-film optical constant."
        ),
    ),
}


def _admittance(n: complex, theta: complex, polarization: str) -> complex:
    if polarization == "s":
        return n * np.cos(theta)
    if polarization == "p":
        return n / np.cos(theta)
    raise ValueError("polarization must be 's' or 'p'")


def _angles(indices: np.ndarray, n_incident: complex, angle_deg: float) -> np.ndarray:
    invariant = complex(n_incident) * np.sin(np.deg2rad(angle_deg))
    return np.asarray([np.arcsin(invariant / complex(n)) for n in indices], dtype=complex)


def stack_indices(
    wavelengths_nm: np.ndarray,
    periods: int,
    high_model: MaterialModel,
    low_model: MaterialModel,
) -> np.ndarray:
    """Return wavelength x layer complex-index matrix for an H/L periodic stack."""
    if periods < 1:
        raise ValueError("periods must be >= 1")
    high = high_model.evaluate(wavelengths_nm)
    low = low_model.evaluate(wavelengths_nm)
    result = np.empty((len(wavelengths_nm), 2 * periods), dtype=complex)
    result[:, 0::2] = high[:, None]
    result[:, 1::2] = low[:, None]
    return result


def quarter_wave_thicknesses(
    periods: int,
    design_wavelength_nm: float,
    n_high_design: float,
    n_low_design: float,
    high_scale: float = 1.0,
    low_scale: float = 1.0,
) -> np.ndarray:
    d_high = design_wavelength_nm / (4.0 * n_high_design) * high_scale
    d_low = design_wavelength_nm / (4.0 * n_low_design) * low_scale
    return np.tile([d_high, d_low], periods).astype(float)


def characteristic_rt(
    wavelengths_nm: Iterable[float],
    n_layers_by_wavelength: np.ndarray,
    d_layers_nm: Iterable[float],
    angle_deg: float = 0.0,
    polarization: str = "s",
    n_incident: complex = 1.0,
    n_substrate: complex = 1.52,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return power R, T and A using the characteristic-matrix formulation."""
    wavelengths = np.asarray(wavelengths_nm, dtype=float)
    indices = np.asarray(n_layers_by_wavelength, dtype=complex)
    thicknesses = np.asarray(d_layers_nm, dtype=float)
    if indices.shape != (len(wavelengths), len(thicknesses)):
        raise ValueError("n_layers_by_wavelength must have shape (wavelengths, layers)")
    R = np.empty(len(wavelengths)); T = np.empty(len(wavelengths))
    for wi, wavelength in enumerate(wavelengths):
        all_n = np.concatenate(([complex(n_incident)], indices[wi], [complex(n_substrate)]))
        theta = _angles(all_n, n_incident, angle_deg)
        q0 = _admittance(all_n[0], theta[0], polarization)
        qs = _admittance(all_n[-1], theta[-1], polarization)
        matrix = np.eye(2, dtype=complex)
        for li, thickness in enumerate(thicknesses, start=1):
            q = _admittance(all_n[li], theta[li], polarization)
            delta = 2.0 * np.pi * all_n[li] * thickness * np.cos(theta[li]) / wavelength
            layer = np.array(
                [[np.cos(delta), 1j * np.sin(delta) / q],
                 [1j * q * np.sin(delta), np.cos(delta)]],
                dtype=complex,
            )
            matrix = matrix @ layer
        b = matrix[0, 0] + matrix[0, 1] * qs
        c = matrix[1, 0] + matrix[1, 1] * qs
        denominator = q0 * b + c
        r = (q0 * b - c) / denominator
        t = 2.0 * q0 / denominator
        R[wi] = float(abs(r) ** 2)
        T[wi] = float(np.real(qs / q0) * abs(t) ** 2)
    A = 1.0 - R - T
    return R, T, A


def recursive_reflectance(
    wavelengths_nm: Iterable[float],
    n_layers_by_wavelength: np.ndarray,
    d_layers_nm: Iterable[float],
    angle_deg: float = 0.0,
    polarization: str = "s",
    n_incident: complex = 1.0,
    n_substrate: complex = 1.52,
) -> np.ndarray:
    """Independent Fresnel recursion used to cross-check matrix reflectance."""
    wavelengths = np.asarray(wavelengths_nm, dtype=float)
    indices = np.asarray(n_layers_by_wavelength, dtype=complex)
    thicknesses = np.asarray(d_layers_nm, dtype=float)
    result = np.empty(len(wavelengths))
    for wi, wavelength in enumerate(wavelengths):
        all_n = np.concatenate(([complex(n_incident)], indices[wi], [complex(n_substrate)]))
        theta = _angles(all_n, n_incident, angle_deg)
        q = np.asarray([_admittance(n, t, polarization) for n, t in zip(all_n, theta)])
        fresnel = (q[:-1] - q[1:]) / (q[:-1] + q[1:])
        r_eff = fresnel[-1]
        for interface in range(len(thicknesses) - 1, -1, -1):
            layer_index = interface + 1
            beta = 2.0 * np.pi * all_n[layer_index] * thicknesses[interface] * np.cos(theta[layer_index]) / wavelength
            phase = np.exp(2j * beta)
            r_eff = (fresnel[interface] + r_eff * phase) / (1.0 + fresnel[interface] * r_eff * phase)
        result[wi] = float(abs(r_eff) ** 2)
    return result


def spectral_metrics(wavelengths_nm: np.ndarray, reflectance: np.ndarray) -> dict[str, float]:
    wavelengths = np.asarray(wavelengths_nm, dtype=float)
    values = np.asarray(reflectance, dtype=float)
    peak = int(np.argmax(values))
    mask = (wavelengths >= 500.0) & (wavelengths <= 600.0)
    above = values >= 0.9
    step = float(np.mean(np.diff(wavelengths)))
    return {
        "peak_wavelength_nm": float(wavelengths[peak]),
        "peak_reflectance": float(values[peak]),
        "mean_reflectance_500_600": float(np.mean(values[mask])),
        "bandwidth_above_0_9_nm": float(np.sum(above) * step),
    }
