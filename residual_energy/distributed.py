"""Distributed hash-owner reduction for coherent external residuals."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import numpy as np
import torch
import torch.distributed as dist


@dataclass(frozen=True)
class OwnerResiduals:
    states: np.ndarray
    residual: np.ndarray
    diagonal: np.ndarray


@dataclass(frozen=True)
class _HierarchicalOwnerGroups:
    local_world_size: int
    node_count: int
    node_index: int
    local_rank: int
    node_rank_counts: tuple[int, ...]
    intra_node: Any
    inter_node_lane: Any | None


_HIERARCHICAL_OWNER_GROUPS: dict[
    tuple[int, int, tuple[int, ...]], _HierarchicalOwnerGroups
] = {}


class Communicator:
    """Torch-distributed reductions used by the production residual report."""

    def __init__(self, device: torch.device | str | None = None):
        requested_world = int(os.environ.get("WORLD_SIZE", "1"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        automatic_device = device is None
        if device is None:
            configured = os.environ.get("RESIDUAL_DEVICE", "").strip()
            if configured:
                device = configured
                automatic_device = False
            elif torch.cuda.is_available():
                device = torch.device("cuda", local_rank % torch.cuda.device_count())
            else:
                device = torch.device("cpu")
        self.device = torch.device(device)

        if requested_world > 1 and not dist.is_initialized():
            use_nccl = self.device.type == "cuda" and dist.is_nccl_available()
            if automatic_device and not use_nccl:
                self.device = torch.device("cpu")
            backend = "nccl" if use_nccl else "gloo"
            dist.init_process_group(backend=backend, init_method="env://")
        elif dist.is_initialized() and automatic_device:
            if dist.get_backend() != "nccl":
                self.device = torch.device("cpu")
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)

    @property
    def rank(self) -> int:
        return int(dist.get_rank()) if _dist_active() else 0

    @property
    def size(self) -> int:
        return int(dist.get_world_size()) if _dist_active() else 1

    def sum_float(self, value: float) -> float:
        return float(self._reduce([value], dist.ReduceOp.SUM, torch.float64)[0])

    def sum_complex(self, value: complex) -> complex:
        result = self._reduce([value], dist.ReduceOp.SUM, torch.complex128)
        return complex(result[0])

    def sum_int(self, value: int) -> int:
        return int(self._reduce([value], dist.ReduceOp.SUM, torch.int64)[0])

    def min_float(self, value: float) -> float:
        return float(self._reduce([value], dist.ReduceOp.MIN, torch.float64)[0])

    def max_float(self, value: float) -> float:
        return float(self._reduce([value], dist.ReduceOp.MAX, torch.float64)[0])

    def sum_values(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(
            self._reduce(values, dist.ReduceOp.SUM, torch.float64),
            dtype=np.float64,
        )

    def barrier(self) -> None:
        if _dist_active():
            dist.barrier()

    def partition(self, length: int) -> slice:
        start = int(length) * self.rank // self.size
        stop = int(length) * (self.rank + 1) // self.size
        return slice(start, stop)

    def owner_reduce(
        self,
        states: np.ndarray,
        residual: np.ndarray,
        diagonal: np.ndarray,
        *,
        diagonal_tolerance: float = 1.0e-10,
    ) -> OwnerResiduals:
        """Coherently sum Q-space residuals on deterministic hash owners."""

        states_u64 = np.ascontiguousarray(states, dtype=np.uint64).reshape(-1)
        residual_np = np.ascontiguousarray(residual, dtype=np.complex128).reshape(-1)
        diagonal_np = np.ascontiguousarray(diagonal, dtype=np.float64).reshape(-1)
        if not (
            states_u64.shape == residual_np.shape == diagonal_np.shape
        ):
            raise ValueError("external state, residual, and diagonal arrays must match")

        states_t = torch.as_tensor(
            states_u64.view(np.int64), dtype=torch.long, device=self.device
        )
        residual_t = torch.as_tensor(
            residual_np, dtype=torch.complex128, device=self.device
        )
        diagonal_t = torch.as_tensor(
            diagonal_np, dtype=torch.float64, device=self.device
        )
        unique, partial_residual, partial_diagonal = _reduce_q_partials_torch(
            states_t,
            residual_t,
            diagonal_t,
        )
        owner_states, owner_residual, owner_diagonal = _route_q_partials(
            unique,
            partial_residual,
            partial_diagonal,
        )

        # H_aa is a state property. Check all duplicates before returning the
        # owner-local arrays used by PT2 and the dBW scalar iterations.
        mismatch = _maximum_diagonal_mismatch(
            owner_states,
            owner_diagonal,
        )
        mismatch = self.max_float(mismatch)
        if mismatch > float(diagonal_tolerance):
            raise ValueError(
                "inconsistent Hamiltonian diagonal for duplicate external states: "
                f"maximum difference {mismatch:.3e}"
            )
        owner_states, owner_residual, owner_diagonal = _reduce_q_partials_torch(
            owner_states,
            owner_residual,
            owner_diagonal,
        )
        _sync_torch_device(self.device)
        return OwnerResiduals(
            states=np.ascontiguousarray(
                owner_states.detach().cpu().numpy().view(np.uint64)
            ),
            residual=np.ascontiguousarray(
                owner_residual.detach().cpu().numpy(), dtype=np.complex128
            ),
            diagonal=np.ascontiguousarray(
                owner_diagonal.detach().cpu().numpy(), dtype=np.float64
            ),
        )

    def _reduce(self, values, operation, dtype: torch.dtype) -> np.ndarray:
        tensor = torch.as_tensor(values, dtype=dtype, device=self.device)
        if _dist_active():
            dist.all_reduce(tensor, op=operation)
        return np.ascontiguousarray(tensor.detach().cpu().numpy())


def _dist_active() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def _sync_torch_device(device: torch.device | str) -> None:
    target = torch.device(device)
    if target.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(target)


def _logical_right_shift_int64(values: torch.Tensor, shift: int) -> torch.Tensor:
    if not 0 < int(shift) < 64:
        raise ValueError("shift must lie strictly between zero and 64")
    mask = (1 << (64 - int(shift))) - 1
    return torch.bitwise_and(torch.bitwise_right_shift(values, int(shift)), mask)


def _splitmix64_torch(values: torch.Tensor) -> torch.Tensor:
    mixed = values.to(dtype=torch.long).clone()
    mixed = mixed + (-7046029254386353131)  # 0x9E3779B97F4A7C15
    mixed = (mixed ^ _logical_right_shift_int64(mixed, 30)) * (
        -4658895280553007687  # 0xBF58476D1CE4E5B9
    )
    mixed = (mixed ^ _logical_right_shift_int64(mixed, 27)) * (
        -7723592293110705685  # 0x94D049BB133111EB
    )
    return mixed ^ _logical_right_shift_int64(mixed, 31)


def _unsigned_remainder_int64(values: torch.Tensor, modulus: int) -> torch.Tensor:
    modulus = int(modulus)
    if modulus <= 0:
        raise ValueError("modulus must be positive")
    remainder = torch.remainder(values, modulus)
    unsigned_correction = pow(2, 64, modulus)
    if unsigned_correction:
        remainder = torch.where(
            values < 0,
            torch.remainder(remainder + unsigned_correction, modulus),
            remainder,
        )
    return remainder


def _splitmix64_owner_indices_torch(
    keys: torch.Tensor,
    world: int,
) -> torch.Tensor:
    if int(world) <= 0:
        raise ValueError("world size must be positive")
    mixed = _splitmix64_torch(keys.to(dtype=torch.long).reshape(-1))
    return _unsigned_remainder_int64(mixed, int(world))


def _scatter_add_complex_torch(
    index: torch.Tensor,
    values: torch.Tensor,
    n_out: int,
) -> torch.Tensor:
    index = index.to(dtype=torch.long)
    values = values.to(dtype=torch.complex128)
    real = torch.zeros(int(n_out), dtype=torch.float64, device=values.device)
    imag = torch.zeros(int(n_out), dtype=torch.float64, device=values.device)
    real.scatter_add_(0, index, values.real)
    imag.scatter_add_(0, index, values.imag)
    return real.to(torch.complex128) + 1j * imag.to(torch.complex128)


def _reduce_q_partials_torch(
    states: torch.Tensor,
    residual: torch.Tensor,
    diagonal: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    states = states.to(dtype=torch.long).reshape(-1)
    residual = residual.to(dtype=torch.complex128).reshape(-1)
    diagonal = diagonal.to(dtype=torch.float64).reshape(-1)
    if not (states.numel() == residual.numel() == diagonal.numel()):
        raise ValueError("Q partial arrays must have the same length")
    if states.numel() == 0:
        return states, residual, diagonal
    unique, group = torch.unique(states, sorted=True, return_inverse=True)
    reduced_residual = _scatter_add_complex_torch(
        group,
        residual,
        int(unique.numel()),
    )
    reduced_diagonal = torch.empty(
        int(unique.numel()), dtype=torch.float64, device=states.device
    )
    reduced_diagonal[group] = diagonal
    return unique, reduced_residual, reduced_diagonal


def _maximum_diagonal_mismatch(
    states: torch.Tensor,
    diagonal: torch.Tensor,
) -> float:
    if states.numel() == 0:
        return 0.0
    unique, group = torch.unique(states, sorted=True, return_inverse=True)
    minimum = torch.full(
        (int(unique.numel()),),
        float("inf"),
        dtype=torch.float64,
        device=states.device,
    )
    maximum = torch.full(
        (int(unique.numel()),),
        -float("inf"),
        dtype=torch.float64,
        device=states.device,
    )
    minimum.scatter_reduce_(0, group, diagonal, reduce="amin", include_self=True)
    maximum.scatter_reduce_(0, group, diagonal, reduce="amax", include_self=True)
    return float(torch.max(maximum - minimum).detach().cpu().item())


def _all_to_all_fixed_rows_torch(
    rows: torch.Tensor,
    send_counts: torch.Tensor,
    *,
    group: Any = None,
) -> torch.Tensor:
    rows = rows.contiguous()
    if not _dist_active():
        return rows
    world = dist.get_world_size(group=group)
    send_counts = send_counts.to(device=rows.device, dtype=torch.long)
    if int(send_counts.numel()) != world:
        raise ValueError("send_counts must contain one entry per rank")
    recv_counts_t = torch.empty_like(send_counts)
    dist.all_to_all_single(recv_counts_t, send_counts, group=group)
    recv_counts = [int(value) for value in recv_counts_t.cpu().tolist()]
    send_counts_list = [int(value) for value in send_counts.cpu().tolist()]
    recv = torch.empty(
        (sum(recv_counts), rows.shape[1]),
        dtype=rows.dtype,
        device=rows.device,
    )
    dist.all_to_all_single(
        recv,
        rows,
        output_split_sizes=recv_counts,
        input_split_sizes=send_counts_list,
        group=group,
    )
    return recv


def _all_to_all_int_float_rows_torch(
    int_rows: torch.Tensor,
    float_rows: torch.Tensor,
    send_counts: torch.Tensor,
    *,
    group: Any = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    int_rows = int_rows.contiguous().to(dtype=torch.long)
    float_rows = float_rows.contiguous().to(dtype=torch.float64)
    if int_rows.shape[0] != float_rows.shape[0]:
        raise ValueError("int_rows and float_rows must have the same row count")
    if int_rows.device != float_rows.device:
        raise ValueError("int_rows and float_rows must be on the same device")
    int_cols = int(int_rows.shape[1])
    float_cols = int(float_rows.shape[1])
    packed = torch.empty(
        (int_rows.shape[0], int_cols + float_cols),
        dtype=torch.long,
        device=int_rows.device,
    )
    if int_cols:
        packed[:, :int_cols] = int_rows
    if float_cols:
        packed[:, int_cols:] = float_rows.view(torch.long)
    recv = _all_to_all_fixed_rows_torch(packed, send_counts, group=group)
    recv_int = recv[:, :int_cols].contiguous()
    recv_float = (
        recv[:, int_cols:].contiguous().view(torch.float64)
        if float_cols
        else torch.empty((recv.shape[0], 0), dtype=torch.float64, device=recv.device)
    )
    return recv_int, recv_float


def _hierarchical_node_rank_counts(world: int) -> tuple[int, ...] | None:
    explicit = os.environ.get("NNQS_NODE_RANKS", "").strip()
    if explicit:
        try:
            counts = tuple(int(value) for value in explicit.split(","))
        except ValueError as error:
            raise ValueError(f"invalid NNQS_NODE_RANKS={explicit!r}") from error
        if not counts or any(value <= 0 for value in counts):
            raise ValueError("NNQS_NODE_RANKS entries must be positive")
        if sum(counts) != int(world):
            raise ValueError("NNQS_NODE_RANKS must sum to WORLD_SIZE")
        return counts
    try:
        local_world = int(os.environ.get("LOCAL_WORLD_SIZE", "0"))
    except ValueError:
        return None
    if local_world <= 0 or local_world >= int(world):
        return None
    if int(world) % local_world != 0:
        return None
    return (local_world,) * (int(world) // local_world)


def _hierarchical_owner_groups(world: int) -> _HierarchicalOwnerGroups | None:
    if not _dist_active() or int(world) <= 1:
        return None
    node_rank_counts = _hierarchical_node_rank_counts(world)
    if node_rank_counts is None:
        return None
    rank = dist.get_rank()
    node_count = len(node_rank_counts)
    node_offsets = np.cumsum((0, *node_rank_counts), dtype=np.int64)
    node_index = int(np.searchsorted(node_offsets[1:], rank, side="right"))
    local_rank = int(rank - node_offsets[node_index])
    local_world = int(node_rank_counts[node_index])
    cache_key = (id(dist.group.WORLD), int(world), node_rank_counts)
    cached = _HIERARCHICAL_OWNER_GROUPS.get(cache_key)
    if cached is not None:
        return cached

    intra_node = None
    for node, node_size in enumerate(node_rank_counts):
        start = int(node_offsets[node])
        ranks = list(range(start, start + int(node_size)))
        group = dist.new_group(ranks=ranks)
        if node == node_index:
            intra_node = group

    inter_node_lane = None
    if len(set(node_rank_counts)) == 1:
        for lane in range(node_rank_counts[0]):
            ranks = [int(node_offsets[node]) + lane for node in range(node_count)]
            group = dist.new_group(ranks=ranks)
            if lane == local_rank:
                inter_node_lane = group
    if intra_node is None:
        raise RuntimeError("failed to construct hierarchical owner process groups")
    result = _HierarchicalOwnerGroups(
        local_world_size=local_world,
        node_count=node_count,
        node_index=node_index,
        local_rank=local_rank,
        node_rank_counts=node_rank_counts,
        intra_node=intra_node,
        inter_node_lane=inter_node_lane,
    )
    _HIERARCHICAL_OWNER_GROUPS[cache_key] = result
    return result


def _route_q_partials(
    states: torch.Tensor,
    residual: torch.Tensor,
    diagonal: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    world = dist.get_world_size() if _dist_active() else 1
    if world == 1:
        return states, residual, diagonal
    hierarchy = _hierarchical_owner_groups(world)
    if hierarchy is None:
        owners = _splitmix64_owner_indices_torch(states, world)
        order = torch.argsort(owners, stable=True)
        counts = torch.bincount(owners, minlength=world).to(torch.long)
        int_rows = states[order].reshape(-1, 1).contiguous()
        float_rows = torch.empty((states.numel(), 3), dtype=torch.float64, device=states.device)
        float_rows[:, 0] = residual.real
        float_rows[:, 1] = residual.imag
        float_rows[:, 2] = diagonal
        return _unpack_owner_rows(
            *_all_to_all_int_float_rows_torch(
                int_rows,
                float_rows[order].contiguous(),
                counts,
            )
        )

    local_world = hierarchy.local_world_size
    global_owners = _splitmix64_owner_indices_torch(states, world)
    local_owners = torch.remainder(global_owners, local_world)
    order = torch.argsort(local_owners, stable=True)
    counts = torch.bincount(local_owners, minlength=local_world).to(torch.long)
    int_rows = states[order].reshape(-1, 1).contiguous()
    float_rows = torch.empty((states.numel(), 3), dtype=torch.float64, device=states.device)
    float_rows[:, 0] = residual.real
    float_rows[:, 1] = residual.imag
    float_rows[:, 2] = diagonal
    node_int, node_float = _all_to_all_int_float_rows_torch(
        int_rows,
        float_rows[order].contiguous(),
        counts,
        group=hierarchy.intra_node,
    )
    node_states, node_residual, node_diagonal = _unpack_owner_rows(node_int, node_float)
    node_states, node_residual, node_diagonal = _reduce_q_partials_torch(
        node_states,
        node_residual,
        node_diagonal,
    )
    node_global_owners = _splitmix64_owner_indices_torch(node_states, world)
    if hierarchy.inter_node_lane is not None:
        destinations = torch.div(
            node_global_owners,
            local_world,
            rounding_mode="floor",
        )
        second_world = hierarchy.node_count
    else:
        destinations = node_global_owners
        second_world = world
    order = torch.argsort(destinations, stable=True)
    counts = torch.bincount(destinations, minlength=second_world).to(torch.long)
    int_rows = node_states[order].reshape(-1, 1).contiguous()
    float_rows = torch.empty(
        (node_states.numel(), 3), dtype=torch.float64, device=states.device
    )
    float_rows[:, 0] = node_residual.real
    float_rows[:, 1] = node_residual.imag
    float_rows[:, 2] = node_diagonal
    return _unpack_owner_rows(
        *_all_to_all_int_float_rows_torch(
            int_rows,
            float_rows[order].contiguous(),
            counts,
            group=hierarchy.inter_node_lane,
        )
    )


def _unpack_owner_rows(
    int_rows: torch.Tensor,
    float_rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    states = int_rows[:, 0]
    residual = float_rows[:, 0].to(torch.complex128)
    residual += 1j * float_rows[:, 1].to(torch.complex128)
    return states, residual, float_rows[:, 2]
