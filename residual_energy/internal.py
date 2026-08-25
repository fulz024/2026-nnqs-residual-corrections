"""Projected diagonal correction for residual left inside the retained support."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .distributed import Communicator


@dataclass(frozen=True)
class InternalCorrection:
    energy: float
    reference_local: np.ndarray
    residual_local: np.ndarray
    diagonal_local: np.ndarray
    correction: float
    response_local: np.ndarray
    response_norm_sq: float
    residual_norm_sq: float
    minimum_abs_denominator: float
    orthogonality_error: float
    equation_error: float


def regularize_denominator(values: np.ndarray, epsilon: float) -> np.ndarray:
    output = np.asarray(values, dtype=np.float64).copy()
    if epsilon <= 0.0:
        return output
    small = np.abs(output) < float(epsilon)
    signs = np.sign(output[small])
    signs[signs == 0.0] = 1.0
    output[small] = signs * float(epsilon)
    return output


def projected_internal_correction(
    vector_local: np.ndarray,
    h_vector_local: np.ndarray,
    diagonal_local: np.ndarray,
    communicator: Communicator,
    *,
    denominator_epsilon: float,
) -> InternalCorrection:
    """Solve ``Q(D-E)Q z=-r`` without gathering retained-space rows."""

    vector = np.asarray(vector_local, dtype=np.complex128).reshape(-1)
    h_vector = np.asarray(h_vector_local, dtype=np.complex128).reshape(-1)
    diagonal = np.asarray(diagonal_local, dtype=np.float64).reshape(-1)
    if vector.shape != h_vector.shape or vector.shape != diagonal.shape:
        raise ValueError("internal arrays must have identical shapes")

    norm_sq = communicator.sum_float(float(np.vdot(vector, vector).real))
    if not np.isfinite(norm_sq) or norm_sq <= 0.0:
        raise ValueError("the retained-space vector has zero or invalid norm")
    norm = np.sqrt(norm_sq)
    reference = vector / norm
    h_reference = h_vector / norm
    energy = communicator.sum_float(float(np.vdot(reference, h_reference).real))
    residual = h_reference - energy * reference
    overlap = communicator.sum_complex(complex(np.vdot(reference, residual)))
    residual -= reference * overlap

    denominator = regularize_denominator(
        energy - diagonal,
        denominator_epsilon,
    )
    inverse_residual = residual / denominator
    weighted_overlap = communicator.sum_complex(
        complex(np.vdot(reference, inverse_residual))
    )
    weighted_norm = communicator.sum_complex(
        complex(np.vdot(reference, reference / denominator))
    )
    if abs(weighted_norm) <= np.finfo(np.float64).eps:
        raise ValueError("the projected internal diagonal problem is singular")
    multiplier = -weighted_overlap / weighted_norm
    response = (residual + multiplier * reference) / denominator

    correction = communicator.sum_float(float(np.vdot(residual, response).real))
    response_norm_sq = communicator.sum_float(float(np.vdot(response, response).real))
    residual_norm_sq = communicator.sum_float(float(np.vdot(residual, residual).real))
    orthogonality = communicator.sum_complex(complex(np.vdot(reference, response)))

    lhs = (diagonal - energy) * response
    lhs -= reference * communicator.sum_complex(complex(np.vdot(reference, lhs)))
    equation_error_sq = communicator.sum_float(float(np.vdot(lhs + residual, lhs + residual).real))
    minimum = communicator.min_float(
        float(np.min(np.abs(denominator), initial=np.inf))
    )
    return InternalCorrection(
        energy=float(energy),
        reference_local=reference,
        residual_local=residual,
        diagonal_local=diagonal,
        correction=float(correction),
        response_local=response,
        response_norm_sq=float(response_norm_sq),
        residual_norm_sq=float(residual_norm_sq),
        minimum_abs_denominator=float(minimum),
        orthogonality_error=float(abs(orthogonality)),
        equation_error=float(np.sqrt(max(equation_error_sq, 0.0))),
    )
