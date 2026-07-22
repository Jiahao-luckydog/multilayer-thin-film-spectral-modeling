"""Minimal transfer-matrix utilities for the guided thin-film projects.

The model is intentionally transparent rather than production-grade. It uses
lossless, isotropic layers and a characteristic-matrix formulation. All units
for wavelength and thickness must be the same; this package uses nanometres.
"""
from __future__ import annotations

import numpy as np


def quarter_wave_stack(periods: int, design_wavelength_nm: float = 550.0,
                       n_high: float = 2.10, n_low: float = 1.45,
                       high_scale: float = 1.0, low_scale: float = 1.0):
    """Return H/L layer refractive indices and quarter-wave thicknesses."""
    if periods < 1:
        raise ValueError("periods must be >= 1")
    d_high = design_wavelength_nm / (4.0 * n_high) * high_scale
    d_low = design_wavelength_nm / (4.0 * n_low) * low_scale
    n_layers = []
    d_layers = []
    for _ in range(periods):
        n_layers.extend([n_high, n_low])
        d_layers.extend([d_high, d_low])
    return np.asarray(n_layers, dtype=float), np.asarray(d_layers, dtype=float)


def _admittance(n: complex, theta: complex, polarization: str) -> complex:
    if polarization == "s":
        return n * np.cos(theta)
    if polarization == "p":
        return n / np.cos(theta)
    raise ValueError("polarization must be 's' or 'p'")


def reflectance_spectrum(wavelengths_nm, n_layers, d_layers,
                         angle_deg: float = 0.0, polarization: str = "s",
                         n_incident: float = 1.0, n_substrate: float = 1.52):
    """Calculate power reflectance for a stratified, lossless film stack."""
    wavelengths = np.asarray(wavelengths_nm, dtype=float)
    n_layers = np.asarray(n_layers, dtype=complex)
    d_layers = np.asarray(d_layers, dtype=float)
    if len(n_layers) != len(d_layers):
        raise ValueError("n_layers and d_layers must have equal length")
    theta0 = np.deg2rad(angle_deg)
    invariant = n_incident * np.sin(theta0)
    all_n = np.concatenate(([complex(n_incident)], n_layers, [complex(n_substrate)]))
    angles = np.asarray([np.arcsin(invariant / n) for n in all_n], dtype=complex)
    q0 = _admittance(all_n[0], angles[0], polarization)
    qs = _admittance(all_n[-1], angles[-1], polarization)
    result = np.empty_like(wavelengths)

    for wi, wavelength in enumerate(wavelengths):
        matrix = np.eye(2, dtype=complex)
        for li, (n_value, thickness) in enumerate(zip(n_layers, d_layers), start=1):
            theta = angles[li]
            q = _admittance(n_value, theta, polarization)
            delta = 2.0 * np.pi * n_value * thickness * np.cos(theta) / wavelength
            layer = np.asarray([
                [np.cos(delta), 1j * np.sin(delta) / q],
                [1j * q * np.sin(delta), np.cos(delta)],
            ], dtype=complex)
            matrix = matrix @ layer
        b = matrix[0, 0] + matrix[0, 1] * qs
        c = matrix[1, 0] + matrix[1, 1] * qs
        reflection = (q0 * b - c) / (q0 * b + c)
        result[wi] = float(np.abs(reflection) ** 2)
    return np.clip(result, 0.0, 1.0)


def normal_incidence_reflectance(wavelengths_nm, n_layers, d_layers,
                                 n_incident: float = 1.0, n_substrate: float = 1.52):
    """Vectorised normal-incidence reflectance used for ML data generation."""
    wavelengths = np.asarray(wavelengths_nm, dtype=float)
    n_layers = np.asarray(n_layers, dtype=float)
    d_layers = np.asarray(d_layers, dtype=float)
    m00 = np.ones_like(wavelengths, dtype=complex)
    m01 = np.zeros_like(wavelengths, dtype=complex)
    m10 = np.zeros_like(wavelengths, dtype=complex)
    m11 = np.ones_like(wavelengths, dtype=complex)
    for n_value, thickness in zip(n_layers, d_layers):
        delta = 2.0 * np.pi * n_value * thickness / wavelengths
        c = np.cos(delta)
        s = np.sin(delta)
        a00, a01 = c, 1j * s / n_value
        a10, a11 = 1j * n_value * s, c
        n00 = m00 * a00 + m01 * a10
        n01 = m00 * a01 + m01 * a11
        n10 = m10 * a00 + m11 * a10
        n11 = m10 * a01 + m11 * a11
        m00, m01, m10, m11 = n00, n01, n10, n11
    b = m00 + m01 * n_substrate
    c = m10 + m11 * n_substrate
    r = (n_incident * b - c) / (n_incident * b + c)
    return np.clip(np.abs(r) ** 2, 0.0, 1.0)


def spectral_summary(wavelengths_nm, reflectance, band=(500.0, 600.0)):
    wavelengths = np.asarray(wavelengths_nm, dtype=float)
    reflectance = np.asarray(reflectance, dtype=float)
    mask = (wavelengths >= band[0]) & (wavelengths <= band[1])
    peak_index = int(np.argmax(reflectance))
    return {
        "peak_wavelength_nm": float(wavelengths[peak_index]),
        "peak_reflectance": float(reflectance[peak_index]),
        "mean_reflectance_500_600": float(np.mean(reflectance[mask])),
        "bandwidth_above_0_9_nm": float(np.sum(reflectance >= 0.9) * np.mean(np.diff(wavelengths))),
    }
