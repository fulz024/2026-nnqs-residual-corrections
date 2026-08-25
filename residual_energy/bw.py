"""Diagonal Brillouin--Wigner fixed point from owner-local residual arrays."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .distributed import Communicator
from .internal import regularize_denominator


@dataclass(frozen=True)
class DiagonalBW:
    correction: float
    energy: float
    response_norm_sq: float
    iterations: int
    fixed_point_error: float
    minimum_abs_denominator: float


def _self_energy(
    omega: float,
    internal_reference: np.ndarray,
    internal_residual: np.ndarray,
    internal_diagonal: np.ndarray,
    external_residual: np.ndarray,
    external_diagonal: np.ndarray,
    communicator: Communicator,
    denominator_epsilon: float,
) -> tuple[float, float, float]:
    denominator_i = regularize_denominator(
        omega - internal_diagonal,
        denominator_epsilon,
    )
    inverse_i = internal_residual / denominator_i
    weighted_overlap = communicator.sum_complex(
        complex(np.vdot(internal_reference, inverse_i))
    )
    weighted_norm = communicator.sum_complex(
        complex(np.vdot(internal_reference, internal_reference / denominator_i))
    )
    if abs(weighted_norm) <= np.finfo(np.float64).eps:
        raise ValueError("the diagonal-BW internal projection is singular")
    multiplier = -weighted_overlap / weighted_norm
    response_i = (internal_residual + multiplier * internal_reference) / denominator_i

    denominator_e = regularize_denominator(
        omega - external_diagonal,
        denominator_epsilon,
    )
    response_e = external_residual / denominator_e
    local = np.array(
        [
            np.vdot(internal_residual, response_i).real
            + np.vdot(external_residual, response_e).real,
            np.vdot(response_i, response_i).real + np.vdot(response_e, response_e).real,
        ],
        dtype=np.float64,
    )
    totals = communicator.sum_values(local)
    minimum = communicator.min_float(
        min(
            float(np.min(np.abs(denominator_i), initial=np.inf)),
            float(np.min(np.abs(denominator_e), initial=np.inf)),
        )
    )
    return float(totals[0]), float(totals[1]), float(minimum)


def solve_diagonal_bw(
    *,
    energy: float,
    pt2: float,
    response_norm_sq: float,
    internal_reference: np.ndarray,
    internal_residual: np.ndarray,
    internal_diagonal: np.ndarray,
    external_residual: np.ndarray,
    external_diagonal: np.ndarray,
    communicator: Communicator,
    denominator_epsilon: float,
    tolerance: float = 5.0e-14,
    maximum_iterations: int = 50,
) -> DiagonalBW:
    """Solve ``delta=Sigma_D(E+delta)`` by scalar Newton iteration."""

    at_reference = _self_energy(
        energy,
        internal_reference,
        internal_residual,
        internal_diagonal,
        external_residual,
        external_diagonal,
        communicator,
        denominator_epsilon,
    )
    scale = max(1.0, abs(pt2), response_norm_sq)
    if abs(at_reference[0] - pt2) > 2.0e-10 * scale:
        raise ValueError("diagonal self-energy does not reproduce PT2 at omega=E")
    if abs(at_reference[1] - response_norm_sq) > 2.0e-10 * scale:
        raise ValueError("diagonal self-energy derivative does not reproduce the response norm")

    shift = pt2 / (1.0 + response_norm_sq)
    for iteration in range(1, int(maximum_iterations) + 1):
        correction, norm_sq, minimum = _self_energy(
            energy + shift,
            internal_reference,
            internal_residual,
            internal_diagonal,
            external_residual,
            external_diagonal,
            communicator,
            denominator_epsilon,
        )
        error = shift - correction
        if abs(error) <= tolerance * max(1.0, abs(shift)):
            return DiagonalBW(
                correction=float(shift),
                energy=float(energy + shift),
                response_norm_sq=float(norm_sq),
                iterations=iteration,
                fixed_point_error=float(error),
                minimum_abs_denominator=float(minimum),
            )
        shift -= error / (1.0 + norm_sq)
    raise RuntimeError(
        f"diagonal-BW fixed point did not converge after {maximum_iterations} iterations"
    )
