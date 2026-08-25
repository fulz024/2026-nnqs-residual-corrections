"""End-to-end frozen-support residual correction evaluator."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np

from .bw import solve_diagonal_bw
from .coupling import CouplingGenerator
from .distributed import Communicator
from .internal import projected_internal_correction, regularize_denominator


@dataclass(frozen=True)
class ResidualReport:
    support_size: int
    external_size: int
    mpi_ranks: int
    restricted_energy: float
    internal_pt2: float
    external_pt2: float
    pt2_correction: float
    pt2_energy: float
    first_order_norm_sq: float
    rpt2_correction: float
    rpt2_energy: float
    dbw_correction: float | None
    dbw_energy: float | None
    dbw_iterations: int | None
    dbw_error: str | None
    residual_norm_sq: float
    minimum_abs_denominator: float
    internal_orthogonality_error: float
    internal_equation_error: float
    coupling_seconds: float
    total_seconds: float

    def to_dict(self) -> dict[str, float | int | str | None]:
        return asdict(self)


def _validate_support(states: np.ndarray, vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    states = np.ascontiguousarray(states, dtype=np.uint64).reshape(-1)
    vector = np.ascontiguousarray(vector, dtype=np.complex128).reshape(-1)
    if states.shape != vector.shape:
        raise ValueError("support states and coefficients must have the same length")
    if not states.size:
        raise ValueError("the retained support is empty")
    if not np.all(np.isfinite(vector)):
        raise ValueError("the retained vector contains non-finite coefficients")
    order = np.argsort(states, kind="stable")
    states = states[order]
    vector = vector[order]
    if np.any(states[1:] == states[:-1]):
        raise ValueError("the retained support contains duplicate determinants")
    return states, vector


def evaluate_residual_report(
    fcidump: str,
    states: np.ndarray,
    vector: np.ndarray,
    *,
    denominator_epsilon: float = 1.0e-12,
    communicator: Communicator | None = None,
) -> ResidualReport:
    """Evaluate restricted ``E_S`` and full-residual PT2, rPT2, and dBW.

    The support and coefficient arrays are replicated metadata. Source rows are
    partitioned over MPI ranks, while external residuals are coherently summed
    on deterministic hash owners before any quadratic contribution is formed.
    """

    started = perf_counter()
    comm = communicator or Communicator()
    states, vector = _validate_support(states, vector)
    local_slice = comm.partition(states.size)
    local_states = states[local_slice]
    local_vector = vector[local_slice]

    generator = CouplingGenerator(fcidump)
    coupling_started = perf_counter()
    graph = generator.generate(local_states)
    coupling_seconds = perf_counter() - coupling_started
    lengths = np.asarray(graph["coupled_states_length"], dtype=np.int64)
    targets = np.asarray(graph["coupled_states"], dtype=np.uint64)
    coefficients = np.asarray(graph["coefficients"], dtype=np.float64)
    target_diagonal = np.asarray(graph["coupled_diagonal"], dtype=np.float64)
    if lengths.size != local_states.size:
        raise RuntimeError("the coupling backend returned an invalid row partition")
    if targets.shape != coefficients.shape or targets.shape != target_diagonal.shape:
        raise RuntimeError("the coupling backend returned inconsistent edge arrays")

    row_offsets = np.concatenate(([0], np.cumsum(lengths)))
    if int(row_offsets[-1]) != targets.size:
        raise RuntimeError("the coupling backend returned invalid row lengths")
    if local_states.size:
        first_edges = row_offsets[:-1]
        if not np.array_equal(targets[first_edges], local_states):
            raise RuntimeError("each coupling row must begin with its diagonal element")
        diagonal_local = target_diagonal[first_edges]
    else:
        diagonal_local = np.empty(0, dtype=np.float64)

    edge_rows = np.repeat(np.arange(local_states.size, dtype=np.int64), lengths)
    positions = np.searchsorted(states, targets)
    inside = positions < states.size
    safe_positions = np.minimum(positions, states.size - 1)
    inside &= states[safe_positions] == targets

    h_vector_local = np.zeros(local_states.size, dtype=np.complex128)
    np.add.at(
        h_vector_local,
        edge_rows[inside],
        coefficients[inside] * vector[safe_positions[inside]],
    )
    internal = projected_internal_correction(
        local_vector,
        h_vector_local,
        diagonal_local,
        comm,
        denominator_epsilon=float(denominator_epsilon),
    )

    external_states = targets[~inside]
    external_partial = coefficients[~inside] * internal.reference_local[edge_rows[~inside]]
    external_diagonal = target_diagonal[~inside]
    owner = comm.owner_reduce(
        external_states,
        external_partial,
        external_diagonal,
    )

    denominator_e = regularize_denominator(
        internal.energy - owner.diagonal,
        float(denominator_epsilon),
    )
    response_e = owner.residual / denominator_e
    external_values = comm.sum_values(
        np.array(
            [
                np.vdot(owner.residual, response_e).real,
                np.vdot(response_e, response_e).real,
                np.vdot(owner.residual, owner.residual).real,
            ],
            dtype=np.float64,
        )
    )
    external_pt2 = float(external_values[0])
    response_norm_sq = float(internal.response_norm_sq + external_values[1])
    residual_norm_sq = float(internal.residual_norm_sq + external_values[2])
    pt2 = float(internal.correction + external_pt2)
    rpt2 = float(pt2 / (1.0 + response_norm_sq))

    try:
        dbw = solve_diagonal_bw(
            energy=internal.energy,
            pt2=pt2,
            response_norm_sq=response_norm_sq,
            internal_reference=internal.reference_local,
            internal_residual=internal.residual_local,
            internal_diagonal=internal.diagonal_local,
            external_residual=owner.residual,
            external_diagonal=owner.diagonal,
            communicator=comm,
            denominator_epsilon=float(denominator_epsilon),
        )
        dbw_correction: float | None = dbw.correction
        dbw_energy: float | None = dbw.energy
        dbw_iterations: int | None = dbw.iterations
        dbw_error = None
        dbw_minimum = dbw.minimum_abs_denominator
    except (RuntimeError, ValueError) as error:
        dbw_correction = None
        dbw_energy = None
        dbw_iterations = None
        dbw_error = str(error)
        dbw_minimum = np.inf

    external_minimum = comm.min_float(
        float(np.min(np.abs(denominator_e), initial=np.inf))
    )
    total_seconds = perf_counter() - started
    return ResidualReport(
        support_size=int(states.size),
        external_size=comm.sum_int(int(owner.states.size)),
        mpi_ranks=comm.size,
        restricted_energy=float(internal.energy),
        internal_pt2=float(internal.correction),
        external_pt2=external_pt2,
        pt2_correction=pt2,
        pt2_energy=float(internal.energy + pt2),
        first_order_norm_sq=response_norm_sq,
        rpt2_correction=rpt2,
        rpt2_energy=float(internal.energy + rpt2),
        dbw_correction=dbw_correction,
        dbw_energy=dbw_energy,
        dbw_iterations=dbw_iterations,
        dbw_error=dbw_error,
        residual_norm_sq=residual_norm_sq,
        minimum_abs_denominator=float(
            min(internal.minimum_abs_denominator, external_minimum, dbw_minimum)
        ),
        internal_orthogonality_error=internal.orthogonality_error,
        internal_equation_error=internal.equation_error,
        coupling_seconds=comm.max_float(coupling_seconds),
        total_seconds=comm.max_float(total_seconds),
    )
