"""Generate local Hubbard FCIDUMP files for the PT2 local-energy benchmarks.

The generated files intentionally live under this benchmark folder so runs do
not depend on older diagnostic/cache directories elsewhere in the workspace.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "fcidump"


@dataclass(frozen=True)
class HubbardCase:
    filename: str
    lx: int
    ly: int
    nelec: int
    periodic: bool
    hopping: float
    write_lower_triangle: bool


CASES = (
    HubbardCase(
        filename="Hub1d4_U4.0_t1.0_Ne4_Sz0_obc.FCIDUMP",
        lx=4,
        ly=1,
        nelec=4,
        periodic=False,
        hopping=1.0,
        write_lower_triangle=False,
    ),
    HubbardCase(
        filename="Hub1d6_U4.0_t1.0_Ne6_Sz0_obc.FCIDUMP",
        lx=6,
        ly=1,
        nelec=6,
        periodic=False,
        hopping=1.0,
        write_lower_triangle=False,
    ),
    HubbardCase(
        filename="Hub1d8_U4.0_t1.0_Ne8_Sz0_obc.FCIDUMP",
        lx=8,
        ly=1,
        nelec=8,
        periodic=False,
        hopping=1.0,
        write_lower_triangle=False,
    ),
    HubbardCase(
        filename="Hub44_U4_Ne16_pbc.FCIDUMP",
        lx=4,
        ly=4,
        nelec=16,
        periodic=True,
        hopping=-1.0,
        write_lower_triangle=True,
    ),
)


def site_index(x: int, y: int, lx: int) -> int:
    return x + lx * y + 1


def nearest_neighbor_edges(lx: int, ly: int, periodic: bool) -> list[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for x in range(lx):
        for y in range(ly):
            i = site_index(x, y, lx)
            for dx, dy in ((1, 0), (0, 1)):
                xn = x + dx
                yn = y + dy
                if periodic:
                    xn %= lx
                    yn %= ly
                elif not (0 <= xn < lx and 0 <= yn < ly):
                    continue
                j = site_index(xn, yn, lx)
                if i != j:
                    edges.add(tuple(sorted((i, j))))
    return sorted(edges)


def write_fcidump(case: HubbardCase, path: Path) -> None:
    nsites = case.lx * case.ly
    with path.open("w", encoding="utf-8") as f:
        f.write(f"&FCI NORB={nsites}, NELEC={case.nelec}, MS2=0,\n")
        f.write(" ORBSYM={}\n".format(" ".join(["1,"] * nsites)))
        f.write(" ISYM=1,\n")
        f.write("&END\n")

        for i in range(1, nsites + 1):
            f.write(f" 4.0 {i:3d} {i:3d} {i:3d} {i:3d}\n")

        for i, j in nearest_neighbor_edges(case.lx, case.ly, case.periodic):
            a, b = (max(i, j), min(i, j)) if case.write_lower_triangle else (i, j)
            f.write(f" {case.hopping:.16g} {a:3d} {b:3d}  0  0\n")

        f.write(" 0.0  0  0  0  0\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        path = OUT / case.filename
        write_fcidump(case, path)
        print(path)


if __name__ == "__main__":
    main()
