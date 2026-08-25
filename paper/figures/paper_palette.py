"""Frozen article-wide Spring Cross H palette.

The system uses saturated dark red/teal anchors, brighter everyday coral/aqua/
mint/peach lines, and a small warm-yellow accent family.  Vanilla is reserved
for regions and uncertainty bands, not thin lines.  Colour identity must still
be paired with a marker or line style whenever several series share a panel.

Do not change individual figure colours locally.  Revise this module and the
semantic mapping below if the paper palette is intentionally updated.
"""

from __future__ import annotations


PALETTE_VERSION = "spring-cross-h-2026-08-17"

SPRING = {
    # Saturated dark anchors: chroma, rather than low lightness alone, carries
    # their emphasis.
    "deep_coral": "#B73736",
    "deep_aqua": "#007E86",
    # Bright everyday line colours.
    "coral": "#F0917E",
    "aqua": "#36C9D1",
    "mint": "#78C4B7",
    "peach": "#EE987F",
    "rose": "#F392B9",
    # Intermediate shades for denser categorical figures.
    "red_coral": "#C84E47",
    "teal": "#168E96",
    "soft_coral": "#E47F6D",
    "soft_aqua": "#58BBC0",
    # Warm accents.  Yellow is for sparse marks; vanilla is an area colour.
    "dark_ochre": "#B7831E",
    "ochre": "#D8AC37",
    "bright_gold": "#EAB51A",
    "yellow": "#F0CF58",
    "vanilla": "#F8E8CB",
}

# Ordered to alternate red and blue-green families before using warm accents.
# The cycle intentionally excludes vanilla because it is too pale for a thin
# line on white.  Markers and line styles remain mandatory for dense panels.
SERIES_CYCLE = (
    SPRING["deep_coral"],
    SPRING["deep_aqua"],
    SPRING["coral"],
    SPRING["aqua"],
    SPRING["red_coral"],
    SPRING["teal"],
    SPRING["mint"],
    SPRING["rose"],
    SPRING["peach"],
    SPRING["dark_ochre"],
    SPRING["soft_aqua"],
    SPRING["soft_coral"],
    SPRING["ochre"],
)

NEUTRALS = {
    "black": "#1F2224",
    "charcoal": "#303538",
    "gray": "#707070",
    "graphite": "#666666",
    "blue_gray": "#667A80",
    "mid_dark": "#62696C",
    "mid": "#858B8D",
    "light": "#BEC1C0",
    "grid": "#DADCD9",
    "off_white": "#F9F8F4",
    "white": "#FFFFFF",
}

SURFACES = {
    "neutral": SPRING["vanilla"],
    "aqua": "#C9ECE9",
    "mint": "#D9EEE8",
    "coral": "#F5C9BF",
    "peach": "#F8D9CD",
    "vanilla": SPRING["vanilla"],
    # Kept as a compatibility role for existing method diagrams; this is now
    # a warm coral surface rather than a separate purple family.
    "plum": "#F2D3CB",
}

# Frozen semantic mapping for all paper figures.
METHOD_COLORS = {
    "restricted": NEUTRALS["graphite"],
    "external": SPRING["coral"],
    "full": SPRING["coral"],
    "rpt2": SPRING["aqua"],
    "dbw": SPRING["rose"],
    "internal": SPRING["deep_aqua"],
    "accent": SPRING["ochre"],
    "missing_data": NEUTRALS["mid"],
    # High-order analysis: primary methods stay in the light, energetic
    # red/aqua/yellow triad; black and gray are reserved for construction
    # lines, axes, and other auxiliary structure.
    "pt_path": SPRING["coral"],
    "rr": SPRING["aqua"],
    "rr_residual": SPRING["ochre"],
    "healthy": SPRING["aqua"],
    "intruder": SPRING["deep_coral"],
    "physical_path": SPRING["bright_gold"],
    # Distributed-performance figures use bright series colours.  Neutral
    # tones are reserved for ideal lines, grids, axes, and style-only keys.
    "hpc_exact": NEUTRALS["gray"],
    "hpc_sketch": SPRING["deep_aqua"],
    "hpc_budget_30k": SPRING["coral"],
    "hpc_budget_100k": SPRING["aqua"],
    "hpc_budget_300k": SPRING["rose"],
    "hpc_scan_bucket": SPRING["coral"],
    "hpc_scan_replica": SPRING["aqua"],
    "hpc_production": NEUTRALS["gray"],
}
