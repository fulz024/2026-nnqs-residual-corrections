"""Regenerate the article figures from the committed source data."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

from paper_palette import METHOD_COLORS, NEUTRALS, SERIES_CYCLE, SURFACES


HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
DATA = PAPER / "figure_data"
OUT = HERE / "generated"
REPO = PAPER.parent
FAILURE_DIAG = (
    REPO / "data" / "diagnostics" / "large_system_failure_diagnosis"
)

COLORS = dict(METHOD_COLORS)
MARKERS = {"restricted": "o", "external": "s", "full": "D", "rpt2": "^", "dbw": "v"}
LABELS = {
    "restricted": r"Restricted $E_{\mathcal{S}}$",
    "external": r"External-only PT2",
    "full": r"PT2 report",
    "rpt2": r"rPT2 report",
    "dbw": r"dBW report",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.25,
            "lines.markersize": 4.5,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 450,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=NEUTRALS["grid"], linewidth=0.50, alpha=0.62, zorder=0)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.04,
        label,
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=9,
        va="bottom",
        ha="left",
    )


def missing_data_panel(ax: plt.Axes, message: str) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(NEUTRALS["light"])
        spine.set_linewidth(0.7)
    ax.text(
        0.5,
        0.53,
        "DATA FILE MISSING",
        transform=ax.transAxes,
        ha="center",
        va="center",
        color=COLORS["missing_data"],
        fontsize=10,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.40,
        message,
        transform=ax.transAxes,
        ha="center",
        va="center",
        color=NEUTRALS["mid_dark"],
        fontsize=7,
        wrap=True,
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{stem}.{suffix}")
    plt.close(fig)


def load_pt_generated_rr_hierarchy() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the committed exact-small-space PT/RR convergence scan."""

    stable = pd.read_csv(DATA / "small_pt_generated_rr.csv")
    stable = stable.rename(columns={"information_order": "order"})
    return (
        stable[stable["method"].isin(("rPT2", "PT partial sum"))].copy(),
        stable[~stable["method"].isin(("rPT2", "PT partial sum"))].copy(),
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_large_pt2_checkpoints() -> pd.DataFrame:
    """Load the committed checkpoint table used by the figures."""

    columns = [
        "system",
        "label",
        "step",
        "seed",
        "support_size",
        "full_residual_error_mha",
        "pt2_error_mha",
        "rpt2_error_mha",
        "dbw_error_mha",
        "dbw_converged",
        "total_sec",
    ]
    csv_path = DATA / "large_pt2_checkpoints.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"missing frozen paper data: {csv_path}")
    stable = pd.read_csv(csv_path)
    if "pt2_error_mha" not in stable:
        stable["pt2_error_mha"] = stable["full_residual_error_mha"]
    for name in ("rpt2_error_mha", "dbw_error_mha", "dbw_converged"):
        if name not in stable:
            stable[name] = np.nan
    return (
        stable[columns]
        .drop_duplicates(["system", "step", "seed"], keep="last")
        .sort_values(["system", "step", "seed"])
    )


def load_large_training_trajectories() -> pd.DataFrame:
    """Load the committed training-trajectory table used by the figures."""

    csv_path = DATA / "large_training_trajectory.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"missing frozen paper data: {csv_path}")
    return (
        pd.read_csv(csv_path)
        .drop_duplicates(["system", "step"], keep="last")
        .sort_values(["system", "step"])
    )








def _box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    edge: str,
    face: str = NEUTRALS["white"],
    fontsize: float = 7.2,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.018",
        linewidth=0.9,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
    )
    return patch


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = NEUTRALS["charcoal"],
    style: str = "-|>",
    connectionstyle: str = "arc3",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=8,
            linewidth=0.85,
            color=color,
            connectionstyle=connectionstyle,
        )
    )


def figure1_method() -> None:
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 4.92), gridspec_kw={"hspace": 0.18})
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    ax = axes[0]
    panel_label(ax, "a")
    _box(
        ax,
        (0.025, 0.34),
        0.17,
        0.32,
        r"BAS support $\mathcal{S}$" "\n" r"NNQS amplitudes $c_i$",
        edge=COLORS["restricted"],
        face=SURFACES["neutral"],
    )
    _box(
        ax,
        (0.245, 0.34),
        0.18,
        0.32,
        "Hamiltonian edge stream" "\n" r"$i\rightarrow j$ and $i\rightarrow a$",
        edge=COLORS["external"],
        face=SURFACES["aqua"],
    )
    _arrow(ax, (0.195, 0.50), (0.245, 0.50))

    _box(
        ax,
        (0.485, 0.62),
        0.19,
        0.27,
        r"Internal $j\in\mathcal{S}$" "\n" r"$E_{\mathcal{S}},\;r_i^{\mathcal{S}}$",
        edge=COLORS["internal"],
        face=SURFACES["mint"],
    )
    _box(
        ax,
        (0.485, 0.11),
        0.19,
        0.27,
        r"External $a\notin\mathcal{S}$" "\n" r"partial $H_{ai}c_i$",
        edge=COLORS["external"],
        face=SURFACES["aqua"],
    )
    _arrow(ax, (0.425, 0.56), (0.485, 0.74), color=COLORS["internal"])
    _arrow(ax, (0.425, 0.44), (0.485, 0.25), color=COLORS["external"])

    _box(
        ax,
        (0.755, 0.32),
        0.22,
        0.36,
        r"$E_{\rm PT2}=E_{\mathcal{S}}$" "\n" r"$+\Delta E_{\rm int}^{(2)}+\Delta E_{\rm ext}^{(2)}$",
        edge=COLORS["full"],
        face=SURFACES["coral"],
        fontsize=7.0,
    )
    _arrow(ax, (0.675, 0.75), (0.755, 0.58), color=COLORS["internal"])
    _arrow(ax, (0.675, 0.25), (0.755, 0.42), color=COLORS["external"])
    ax.text(
        0.58,
        0.015,
        "No NNQS evaluation on external determinants",
        ha="center",
        va="bottom",
        fontsize=6.8,
        color=NEUTRALS["charcoal"],
    )

    ax = axes[1]
    panel_label(ax, "b")
    ax.text(
        0.105,
        0.93,
        "PT2",
        ha="center",
        va="top",
        fontsize=7.4,
        fontweight="bold",
    )
    _box(
        ax,
        (0.015, 0.37),
        0.18,
        0.27,
        r"$\Sigma_2(E_{\mathcal{S}})=E^{(2)}$"
        "\n"
        r"$E_{\rm PT2}=E_{\mathcal{S}}+E^{(2)}$",
        edge=COLORS["restricted"],
        face=SURFACES["neutral"],
        fontsize=6.5,
    )

    ax.text(
        0.335,
        0.93,
        "rPT2",
        ha="center",
        va="top",
        fontsize=7.4,
        fontweight="bold",
    )
    _box(
        ax,
        (0.255, 0.37),
        0.16,
        0.27,
        r"$Z=[1-\Sigma_2'(E_{\mathcal{S}})]^{-1}$"
        "\n"
        r"$E_{\rm rPT2}=E_{\mathcal{S}}+ZE^{(2)}$",
        edge=COLORS["accent"],
        face=SURFACES["plum"],
        fontsize=6.3,
    )

    ax.text(
        0.585,
        0.93,
        "dBW",
        ha="center",
        va="top",
        fontsize=7.4,
        fontweight="bold",
    )
    _box(
        ax,
        (0.495, 0.37),
        0.18,
        0.27,
        r"$\delta=\Sigma_2(E_{\mathcal{S}}+\delta)$"
        "\n"
        r"$E_{\rm dBW}=E_{\mathcal{S}}+\delta$",
        edge=COLORS["internal"],
        face=SURFACES["mint"],
        fontsize=6.5,
    )
    ax.text(
        0.335,
        0.30,
        r"parallel readings of $\Sigma_2(E)$",
        ha="center",
        va="top",
        fontsize=6.4,
        color=NEUTRALS["charcoal"],
    )
    ax.text(
        0.86,
        0.93,
        r"$Q_K$ diagnostic",
        ha="center",
        va="top",
        fontsize=7.4,
        fontweight="bold",
    )
    _box(
        ax,
        (0.75, 0.32),
        0.22,
        0.36,
        r"screened residual shell $Q_K$"
        "\n"
        r"Ritz roots of $H_{\mathcal{S}\oplus Q_K}$"
        "\n"
        "reference overlap",
        edge=COLORS["full"],
        face=SURFACES["coral"],
        fontsize=6.3,
    )

    ax = axes[2]
    panel_label(ax, "c")
    rank_y = [0.78, 0.50, 0.22]
    owner_y = [0.78, 0.50, 0.22]
    rank_names = [r"source rank 0", r"source rank 1", r"source rank $P-1$"]
    owner_names = [r"owner 0", r"owner 1", r"owner $P-1$"]
    for y, name in zip(rank_y, rank_names):
        _box(
            ax,
            (0.025, y - 0.09),
            0.20,
            0.18,
            name + "\nlocal precombine",
            edge=COLORS["restricted"],
            face=SURFACES["neutral"],
            fontsize=6.8,
        )
    for y, name in zip(owner_y, owner_names):
        _box(
            ax,
            (0.755, y - 0.09),
            0.22,
            0.18,
            name + "\ncoherent key reduce",
            edge=COLORS["external"],
            face=SURFACES["aqua"],
            fontsize=6.8,
        )
    _box(
        ax,
        (0.39, 0.35),
        0.22,
        0.30,
        "hash owner routing" "\n" r"$a\mapsto h(a)\;\mathrm{mod}\;P$",
        edge=COLORS["accent"],
        face=SURFACES["plum"],
        fontsize=7.0,
    )
    for y in rank_y:
        _arrow(ax, (0.225, y), (0.39, 0.50), color=COLORS["accent"])
    for y in owner_y:
        _arrow(ax, (0.61, 0.50), (0.755, y), color=COLORS["accent"])
    save_figure(fig, "fig1_method_dataflow")










def figure2_large_trajectories() -> None:
    trajectories = load_large_training_trajectories()
    pt2 = load_large_pt2_checkpoints()
    correction_styles = {
        "full": {"linestyle": "-", "linewidth": 1.20, "marker": "o", "markersize": 5.3, "zorder": 4},
        "rpt2": {"linestyle": "-", "linewidth": 1.30, "marker": None, "markersize": 0.0, "zorder": 5},
        "dbw": {"linestyle": "None", "linewidth": 0.0, "marker": "D", "markersize": 5.3, "zorder": 3},
    }
    systems = [
        ("h2o_ccpvdz", r"H$_2$O/cc-pVDZ"),
        ("nh3_ccpvdz", r"NH$_3$/cc-pVDZ"),
        ("n2_ccpvdz", r"N$_2$/cc-pVDZ"),
        ("hcn_ccpvdz", r"HCN/cc-pVDZ"),
        ("cr2_cas24e30o", r"Cr$_2$ CAS(24e,30o)"),
        ("fe2s2_cas30e20o", r"Fe$_2$S$_2$ CAS(30e,20o)"),
    ]
    y_limits = {
        "n2_ccpvdz": (-15.0, 80.0),
        "h2o_ccpvdz": (-2.5, 45.0),
        "cr2_cas24e30o": (-230.0, 390.0),
        "fe2s2_cas30e20o": (175.0, 270.0),
    }
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(7.0, 6.45),
        sharex=True,
        gridspec_kw={"hspace": 0.25, "wspace": 0.22},
    )
    for panel, (ax, (system, title)) in enumerate(zip(axes.flat, systems)):
        subset = trajectories[trajectories.system == system].sort_values("step")
        points = pt2[pt2.system == system].sort_values(["step", "seed"])
        ax.axhspan(-1.6, 1.6, color=SURFACES["neutral"], alpha=0.38, zorder=0)
        ax.axhline(0.0, color=NEUTRALS["mid"], linewidth=0.65, zorder=1)
        if not subset.empty:
            smooth = subset["restricted_error_mha"].rolling(21, center=True, min_periods=1).median()
            ax.plot(
                subset["step"],
                smooth,
                color=COLORS["restricted"],
                linewidth=1.80,
                label=r"Restricted $E_{\mathcal{S}}$" if panel == 0 else None,
                zorder=2,
            )
        if not points.empty:
            for key, column in (
                ("full", "pt2_error_mha"),
                ("rpt2", "rpt2_error_mha"),
                ("dbw", "dbw_error_mha"),
            ):
                report = points.groupby("step")[column].mean().dropna().reset_index()
                if report.empty:
                    continue
                style = correction_styles[key]
                ax.plot(
                    report["step"],
                    report[column],
                    color=COLORS[key],
                    linestyle=style["linestyle"],
                    linewidth=style["linewidth"],
                    marker=style["marker"],
                    markersize=style["markersize"],
                    markerfacecolor=NEUTRALS["white"],
                    markeredgecolor=COLORS[key],
                    markeredgewidth=1.2,
                    label=LABELS[key] if panel == 0 else None,
                    zorder=style["zorder"],
                )
        if points.empty:
            ax.text(
                0.97,
                0.06,
                "PT2 data unavailable",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=6.4,
                color=NEUTRALS["mid"],
            )
        ax.set_title(title, pad=3)
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=1.6, linscale=0.8, base=10)
        if system == "fe2s2_cas30e20o":
            ax.set_yscale("linear")
            ax.set_yticks([180, 200, 220, 240, 260])
        if system in y_limits:
            ax.set_ylim(*y_limits[system])
        ax.set_xlim(1.0e2, 3.5e4)
        clean_axis(ax)
        panel_label(ax, chr(ord("a") + panel))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        columnspacing=1.0,
        handletextpad=0.35,
    )
    axes[2, 0].set_xlabel("Restricted-training updates")
    axes[2, 1].set_xlabel("Restricted-training updates")
    axes[0, 0].set_ylabel(r"Signed energy error (m$E_h$)")
    axes[1, 0].set_ylabel(r"Signed energy error (m$E_h$)")
    axes[2, 0].set_ylabel(r"Signed energy error (m$E_h$)")
    fig.subplots_adjust(top=0.91, bottom=0.08)
    save_figure(fig, "fig2_large_training_trajectories")


def figure_large_root_locality() -> None:
    """Show whether a frozen NNQS and its first residual shell contain the target."""

    systems = [
        {
            "title": r"N$_2$",
            "support": "n2_internal_ground.json",
            "augmented": "n2_screened_q_k300000_spectrum.json",
        },
        {
            "title": r"Cr$_2$",
            "support": "cr2_spectrum16.json",
            "augmented": "cr2_screened_q_k1000000_spectrum.json",
        },
        {
            "title": r"Fe$_2$S$_2$",
            "support": "fe2s2_spectrum16.json",
            "augmented": "fe2s2_screened_q_k300000_spectrum.json",
        },
        {
            "title": "Mn(salen)",
            "support": "mn_salen_spectrum16.json",
            "augmented": "mn_salen_screened_q_k300000_spectrum.json",
        },
    ]

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(7.0, 3.20),
        sharey=True,
        gridspec_kw={"wspace": 0.08},
    )
    x_support, x_es, x_pt2, x_augmented = 0.18, 0.72, 1.12, 1.72
    y_min, y_max = -12.0, 315.0
    root_half_width = 0.17

    for panel, (ax, meta) in enumerate(zip(axes, systems)):
        support = _read_json(FAILURE_DIAG / meta["support"])
        augmented = _read_json(FAILURE_DIAG / meta["augmented"])
        support_errors = np.asarray(support["low_spectrum"]["errors_mha"], dtype=float)
        support_weights = np.asarray(support["low_spectrum"]["nnqs_weights"], dtype=float)
        support_ref = int(np.argmax(support_weights))
        augmented_roots = augmented["augmented"]["roots"]
        augmented_ref = int(
            augmented["augmented"]["largest_reference_overlap_root"]["index"]
        )

        ax.axhline(0.0, color=NEUTRALS["black"], linewidth=0.9, zorder=1)

        # Exact Ritz roots in the sampled support.  Aqua identifies the root
        # carrying the NNQS; coral identifies a lower competing root.
        for index, (error, weight) in enumerate(zip(support_errors[:8], support_weights[:8])):
            if not (y_min <= error <= y_max):
                continue
            if index == support_ref:
                color, linewidth, alpha = COLORS["healthy"], 2.25, 1.0
            elif index == 0 and support_ref != 0:
                color, linewidth, alpha = COLORS["pt_path"], 1.75, 1.0
            else:
                color, linewidth, alpha = NEUTRALS["mid"], 0.9, 0.92
            ax.hlines(
                error,
                x_support - root_half_width,
                x_support + root_half_width,
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                zorder=3,
            )

        # Exact roots in the fixed S + Q_K space use the same root-locality
        # encoding.  The first six roots are enough to show the relevant split.
        for root in augmented_roots[:8]:
            index = int(root["index"])
            error = float(root["energy_error_mha"])
            if not (y_min <= error <= y_max):
                continue
            if index == augmented_ref:
                color, linewidth, alpha = COLORS["healthy"], 2.25, 1.0
            elif index == 0 and augmented_ref != 0:
                color, linewidth, alpha = COLORS["pt_path"], 1.75, 1.0
            else:
                color, linewidth, alpha = NEUTRALS["mid"], 0.9, 0.92
            ax.hlines(
                error,
                x_augmented - root_half_width,
                x_augmented + root_half_width,
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                zorder=3,
            )

        e_s_error = float(support["E_S_error_mha"])
        pt2_error = float(support["projected_internal_plus_external_error_mha"])
        ax.scatter(
            [x_es],
            [e_s_error],
            s=23,
            marker="D",
            facecolor=NEUTRALS["black"],
            edgecolor=NEUTRALS["black"],
            linewidth=0.6,
            zorder=5,
        )
        ax.scatter(
            [x_pt2],
            [pt2_error],
            s=24,
            marker="o",
            facecolor=NEUTRALS["black"],
            edgecolor=NEUTRALS["black"],
            linewidth=0.6,
            zorder=5,
        )
        ax.annotate(
            "",
            xy=(x_pt2, pt2_error),
            xytext=(x_es, e_s_error),
            arrowprops={
                "arrowstyle": "-|>",
                "color": NEUTRALS["black"],
                "linewidth": 0.65,
                "mutation_scale": 7.0,
                "shrinkA": 5.0,
                "shrinkB": 5.0,
            },
            zorder=4,
        )

        ax.set_title(meta["title"], pad=6)
        ax.set_xlim(-0.08, 2.05)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(
            [x_support, x_es, x_pt2, x_augmented],
            [
                r"$H_{\mathcal{SS}}$",
                r"$E_{\mathcal{S}}$",
                "PT2",
                r"$H_{\mathcal{S}\oplus Q_K}$",
            ],
        )
        ax.tick_params(axis="x", labelsize=6.1, pad=3, length=0)
        ax.set_yticks(np.arange(0.0, 301.0, 50.0))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if panel > 0:
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", left=False)
        panel_label(ax, chr(ord("a") + panel))

    axes[0].set_ylabel(r"Energy error relative to reference (m$E_h$)")
    legend_handles = [
        Line2D([0], [0], color=COLORS["healthy"], linewidth=2.25, label="NNQS-connected root"),
        Line2D([0], [0], color=COLORS["pt_path"], linewidth=1.75, label="lower competing root"),
        Line2D([0], [0], color=NEUTRALS["mid"], linewidth=0.9, label="other Ritz roots"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        columnspacing=1.4,
        handlelength=2.0,
        fontsize=6.4,
    )
    fig.subplots_adjust(left=0.075, right=0.99, top=0.88, bottom=0.21)
    save_figure(fig, "fig_large_root_locality")


def figure3_sample_budget() -> None:
    data = pd.read_csv(DATA / "n2_support_budget_triad.csv")
    expected_budgets = [3_000, 6_000, 12_000, 30_000, 100_000, 300_000]
    if len(data) != 30 or sorted(data["requested_budget"].unique()) != expected_budgets:
        raise ValueError("N2 support-budget study must contain six budgets and five seeds")
    if not data["dbw_converged"].all() or set(data["world_size"]) != {4}:
        raise ValueError("N2 support-budget study requires converged dBW at fixed four-rank cost")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.0, 2.75),
        gridspec_kw={"wspace": 0.32},
    )
    budget_labels = ["3k", "6k", "12k", "30k", "100k", "300k"]
    mean_support = data.groupby("requested_budget")["support_size"].mean().reindex(expected_budgets)

    ax = axes[0]
    panel_label(ax, "a")
    ax.axhspan(-1.6, 1.6, color=SURFACES["neutral"], alpha=0.38, zorder=0)
    ax.axhline(0.0, color=NEUTRALS["mid"], linewidth=0.65, zorder=1)
    report_styles = {
        "restricted": ("E_S_error_mha", None, None, "-", 1.65, 6),
        "full": ("pt2_error_mha", "o", "o", "-", 1.15, 4),
        "rpt2": ("rpt2_error_mha", None, None, "-", 1.25, 5),
        "dbw": ("dbw_error_mha", "D", "D", "None", 0.0, 3),
    }
    for key, (column, raw_marker, mean_marker, linestyle, linewidth, zorder) in report_styles.items():
        if raw_marker is not None:
            ax.scatter(
                data["support_size"],
                data[column],
                marker=raw_marker,
                s=16,
                facecolor=NEUTRALS["white"],
                edgecolor=COLORS[key],
                linewidth=0.9,
                alpha=0.58,
                zorder=zorder,
            )
        summary = (
            data.groupby("requested_budget")[column]
            .agg(["mean", "std", "min", "max"])
            .reindex(expected_budgets)
        )
        if key == "rpt2":
            ax.fill_between(
                mean_support,
                summary["min"],
                summary["max"],
                color=COLORS[key],
                alpha=0.13,
                linewidth=0.0,
                zorder=zorder - 1,
            )
        ax.errorbar(
            mean_support,
            summary["mean"],
            yerr=summary["std"] if key in {"full", "dbw"} else None,
            color=COLORS[key],
            linestyle=linestyle,
            linewidth=linewidth,
            marker=mean_marker,
            markersize=5.0,
            markerfacecolor=NEUTRALS["white"],
            markeredgecolor=COLORS[key],
            markeredgewidth=1.25,
            capsize=1.8,
            elinewidth=0.75,
            label=LABELS[key],
            zorder=zorder + 2,
        )
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1.6, linscale=0.85, base=10)
    ax.set_xlim(2.5e3, 3.7e5)
    ax.set_ylim(-2.4, 45.0)
    ax.set_xticks(expected_budgets, budget_labels)
    ax.set_xlabel(r"Realized unique support $M$")
    ax.set_ylabel(r"Signed energy error (m$E_h$)")
    clean_axis(ax)
    ax.legend(
        loc="center right",
        bbox_to_anchor=(0.99, 0.64),
        ncol=2,
        columnspacing=0.8,
        handletextpad=0.35,
    )

    ax = axes[1]
    panel_label(ax, "b")
    time_summary = (
        data.groupby("requested_budget")["correction_total_sec"]
        .agg(["mean", "std"])
        .reindex(expected_budgets)
    )
    ax.scatter(
        data["support_size"],
        data["correction_total_sec"],
        marker="o",
        s=18,
        facecolor=NEUTRALS["white"],
        edgecolor=COLORS["internal"],
        linewidth=0.95,
        alpha=0.62,
        zorder=2,
    )
    ax.errorbar(
        mean_support,
        time_summary["mean"],
        yerr=time_summary["std"],
        color=COLORS["internal"],
        linewidth=1.35,
        marker="o",
        markersize=5.0,
        markerfacecolor=NEUTRALS["white"],
        markeredgecolor=COLORS["internal"],
        markeredgewidth=1.25,
        capsize=1.8,
        elinewidth=0.75,
        label="Four-rank correction",
        zorder=4,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(2.5e3, 3.7e5)
    ax.set_xticks(expected_budgets, budget_labels)
    ax.set_xlabel(r"Realized unique support $M$")
    ax.set_ylabel("Correction wall time (s)")
    clean_axis(ax)
    ax.legend(loc="upper left")
    save_figure(fig, "fig3_sample_budget_scaling")






def figure4_hpc() -> None:
    """Communication architecture plus the three primary scaling panels."""

    strong_path = DATA / "hpc_pt2_strong_budgets.csv"
    weak_path = DATA / "hpc_pt2_weak_scaling.csv"
    strong = pd.read_csv(strong_path) if strong_path.exists() else pd.DataFrame()
    weak = pd.read_csv(weak_path) if weak_path.exists() else pd.DataFrame()

    fig = plt.figure(figsize=(7.0, 4.15))
    grid = fig.add_gridspec(
        2,
        6,
        height_ratios=(1.00, 2.15),
        hspace=0.22,
        wspace=1.02,
    )
    architecture = fig.add_subplot(grid[0, :])
    strong_ax = fig.add_subplot(grid[1, 0:2])
    weak_ax = fig.add_subplot(grid[1, 2:4])
    memory_ax = fig.add_subplot(grid[1, 4:6])

    architecture.set_xlim(0.0, 1.0)
    architecture.set_ylim(0.0, 1.0)
    architecture.axis("off")
    architecture.text(
        -0.018,
        1.01,
        "a",
        fontweight="bold",
        fontsize=9,
        va="bottom",
        ha="left",
    )
    architecture.plot(
        [0.5, 0.5],
        [0.10, 0.95],
        color=NEUTRALS["light"],
        linewidth=0.75,
        zorder=0,
    )

    def communication_box(
        x: float,
        y: float,
        width: float,
        height: float,
        label: str,
        *,
        face: str,
        edge: str,
        fontsize: float = 6.25,
    ) -> None:
        architecture.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.006,rounding_size=0.010",
                facecolor=face,
                edgecolor=edge,
                linewidth=0.85,
                zorder=4,
            )
        )
        architecture.text(
            x + width / 2,
            y + height / 2,
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=NEUTRALS["charcoal"],
            linespacing=1.05,
            zorder=5,
        )

    def communication_arrow(
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        color: str,
        alpha: float = 0.72,
        radius: float = 0.0,
        width: float = 0.88,
    ) -> None:
        architecture.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=5.8,
                connectionstyle=f"arc3,rad={radius}",
                linewidth=width,
                color=color,
                alpha=alpha,
                shrinkA=1.4,
                shrinkB=1.4,
                zorder=1,
            )
        )

    exact_color = COLORS["hpc_exact"]
    sketch_color = COLORS["hpc_sketch"]
    source_face = SURFACES["vanilla"]
    exact_face = SURFACES["coral"]
    sketch_face = SURFACES["aqua"]
    box_width = 0.090
    box_height = 0.15
    exact_x = (0.028, 0.148, 0.268)
    sketch_x = (0.528, 0.648, 0.768)
    source_y = 0.68
    target_y = 0.20

    architecture.text(
        0.245,
        0.975,
        "Exact key ownership",
        ha="center",
        va="top",
        fontsize=7.9,
        fontweight="bold",
        color=NEUTRALS["charcoal"],
    )
    architecture.text(
        0.745,
        0.975,
        "Random-phase CountSketch",
        ha="center",
        va="top",
        fontsize=7.9,
        fontweight="bold",
        color=NEUTRALS["charcoal"],
    )

    rank_labels = ("rank 0", "rank 1", r"rank $P-1$")
    owner_labels = ("owner 0", "owner 1", r"owner $P-1$")
    for x, rank_label in zip(exact_x, rank_labels):
        communication_box(
            x,
            source_y,
            box_width,
            box_height,
            rank_label,
            face=source_face,
            edge=exact_color,
        )
    for x, owner_label in zip(exact_x, owner_labels):
        communication_box(
            x,
            target_y,
            box_width,
            box_height,
            owner_label,
            face=exact_face,
            edge=exact_color,
        )
    for source_index, source_x in enumerate(exact_x):
        for owner_index, owner_x in enumerate(exact_x):
            communication_arrow(
                (source_x + box_width / 2, source_y - 0.004),
                (owner_x + box_width / 2, target_y + box_height + 0.004),
                color=exact_color,
                radius=0.045 * (owner_index - source_index),
                alpha=0.54 if owner_index != source_index else 0.76,
            )
    architecture.text(
        0.205,
        0.515,
        "keyed all-to-all",
        ha="center",
        va="center",
        fontsize=6.60,
        fontweight="bold",
        color=NEUTRALS["charcoal"],
        bbox={"facecolor": NEUTRALS["white"], "edgecolor": "none", "pad": 0.7},
        zorder=3,
    )
    communication_box(
        0.395,
        target_y,
        0.080,
        box_height,
        "scalar\nsum",
        face=NEUTRALS["white"],
        edge=exact_color,
        fontsize=6.25,
    )
    for owner_x in exact_x:
        communication_arrow(
            (owner_x + box_width, target_y + box_height / 2),
            (0.395, target_y + box_height / 2),
            color=exact_color,
            alpha=0.48,
            width=0.62,
        )
    for x, rank_label in zip(sketch_x, rank_labels):
        communication_box(
            x,
            source_y,
            box_width,
            box_height,
            rank_label,
            face=source_face,
            edge=sketch_color,
        )
    shard_labels = ("shard 0", "shard 1", r"shard $P-1$")
    for x, shard_label in zip(sketch_x, shard_labels):
        communication_box(
            x,
            target_y,
            box_width,
            box_height,
            shard_label,
            face=sketch_face,
            edge=sketch_color,
        )
    for source_index, source_x in enumerate(sketch_x):
        for shard_index, shard_x in enumerate(sketch_x):
            communication_arrow(
                (source_x + box_width / 2, source_y - 0.004),
                (shard_x + box_width / 2, target_y + box_height + 0.004),
                color=sketch_color,
                radius=0.045 * (shard_index - source_index),
                alpha=0.54 if shard_index != source_index else 0.76,
            )
    architecture.text(
        0.705,
        0.515,
        r"reduce-scatter ($B\times R$)",
        ha="center",
        va="center",
        fontsize=6.60,
        fontweight="bold",
        color=sketch_color,
        bbox={"facecolor": NEUTRALS["white"], "edgecolor": "none", "pad": 0.7},
        zorder=3,
    )
    communication_box(
        0.895,
        target_y,
        0.080,
        box_height,
        "$R$ norms",
        face=NEUTRALS["white"],
        edge=sketch_color,
        fontsize=6.0,
    )
    for shard_x in sketch_x:
        communication_arrow(
            (shard_x + box_width, target_y + box_height / 2),
            (0.895, target_y + box_height / 2),
            color=sketch_color,
            alpha=0.48,
            width=0.62,
        )
    budgets = [
        (30_000, "30k", "o"),
        (100_000, "100k", "s"),
        (300_000, "300k", "D"),
    ]
    estimator_styles = {
        "exact": (exact_color, "-", NEUTRALS["white"], "Exact owner"),
        "countsketch": (sketch_color, "--", sketch_color, "CountSketch"),
    }
    required = {
        "estimator",
        "budget",
        "ranks",
        "total_sec",
        "total_sec_std",
        "peak_device_allocated_bytes",
    }
    if not strong.empty and not required.issubset(strong.columns):
        missing = sorted(required - set(strong.columns))
        raise ValueError(f"hpc_pt2_strong_budgets.csv lacks columns: {missing}")

    panel_label(strong_ax, "b")
    if strong.empty:
        missing_data_panel(strong_ax, "production strong scaling")
    else:
        for budget, _, marker in budgets:
            for estimator, (color, linestyle, face, _) in estimator_styles.items():
                frame = strong[
                    (strong["budget"] == budget)
                    & (strong["estimator"] == estimator)
                ].sort_values("ranks")
                if frame.empty:
                    continue
                strong_ax.errorbar(
                    frame["ranks"],
                    frame["total_sec"],
                    yerr=frame["total_sec_std"],
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.25,
                    marker=marker,
                    markerfacecolor=face,
                    markeredgecolor=color,
                    markeredgewidth=0.95,
                    markersize=4.4,
                    capsize=1.6,
                    zorder=3,
                )
        ranks = sorted(int(value) for value in strong["ranks"].unique())
        strong_ax.set_xscale("log", base=2)
        strong_ax.set_yscale("log")
        strong_ax.set_xticks(ranks, [str(rank) for rank in ranks])
        strong_ax.set_xlim(0.9, max(ranks) * 1.10)
        strong_ax.set_xlabel("GPU ranks")
        strong_ax.set_ylabel("Full-report wall time (s)")
        clean_axis(strong_ax)
        estimator_handles = [
            Line2D(
                [0], [0], color=color, linestyle=linestyle, linewidth=1.25,
                marker="o", markerfacecolor=face, markeredgecolor=color,
                label=label,
            )
            for color, linestyle, face, label in estimator_styles.values()
        ]
        estimator_legend = strong_ax.legend(
            handles=estimator_handles,
            loc="upper right",
            fontsize=5.8,
            handlelength=1.8,
        )
        strong_ax.add_artist(estimator_legend)
        budget_handles = [
            Line2D(
                [0], [0], color=NEUTRALS["mid_dark"], linestyle="none",
                marker=marker, markerfacecolor=NEUTRALS["white"],
                markeredgecolor=NEUTRALS["mid_dark"], label=label,
            )
            for _, label, marker in budgets
        ]
        strong_ax.legend(
            handles=budget_handles,
            loc="lower left",
            fontsize=5.7,
            handlelength=1.0,
            ncol=1,
        )

    panel_label(weak_ax, "c")
    weak_required = {
        "estimator",
        "ranks",
        "states_per_rank",
        "external_edges_per_sec",
        "external_sec",
        "external_sec_std",
    }
    if weak.empty:
        missing_data_panel(weak_ax, "weak scaling at 12k retained states per rank")
    elif not weak_required.issubset(weak.columns):
        missing = sorted(weak_required - set(weak.columns))
        raise ValueError(f"hpc_pt2_weak_scaling.csv lacks columns: {missing}")
    else:
        weak_markers = {"exact": "o", "countsketch": "s"}
        for estimator, (color, linestyle, face, label) in estimator_styles.items():
            frame = weak[weak["estimator"] == estimator].sort_values("ranks")
            throughput = frame["external_edges_per_sec"].to_numpy(dtype=float) / 1.0e6
            error = throughput * (
                frame["external_sec_std"].to_numpy(dtype=float)
                / frame["external_sec"].to_numpy(dtype=float)
            )
            weak_ax.errorbar(
                frame["ranks"],
                throughput,
                yerr=error,
                color=color,
                linestyle=linestyle,
                linewidth=1.25,
                marker=weak_markers[estimator],
                markerfacecolor=face,
                markeredgecolor=color,
                markeredgewidth=0.95,
                markersize=4.4,
                capsize=1.6,
                label=label,
                zorder=3,
            )
        ranks = sorted(int(value) for value in weak["ranks"].unique())
        weak_ax.set_xscale("log", base=2)
        weak_ax.set_yscale("log")
        weak_ax.set_xticks(ranks, [str(rank) for rank in ranks])
        weak_ax.set_yticks([20, 50, 100, 200], ["20", "50", "100", "200"])
        weak_ax.set_xlim(0.9, max(ranks) * 1.10)
        weak_ax.set_xlabel("GPU ranks")
        weak_ax.set_ylabel(r"External throughput ($10^6$ edges/s)")
        clean_axis(weak_ax)
        weak_ax.legend(loc="upper left", fontsize=5.8, handlelength=1.8)
        weak_ax.text(
            0.97,
            0.06,
            "12k states/rank",
            transform=weak_ax.transAxes,
            ha="right",
            fontsize=5.9,
            color=NEUTRALS["mid_dark"],
        )

    panel_label(memory_ax, "d")
    if strong.empty:
        missing_data_panel(memory_ax, "peak device memory")
    else:
        memory_rank = int(strong["ranks"].max())
        memory = strong[strong["ranks"] == memory_rank].copy()
        x = np.arange(len(budgets), dtype=float)
        width = 0.34
        maximum = 0.0
        for offset, estimator in zip((-width / 2, width / 2), ("exact", "countsketch")):
            color, _, face, label = estimator_styles[estimator]
            values = [
                float(
                    memory[
                        (memory["budget"] == budget)
                        & (memory["estimator"] == estimator)
                    ]["peak_device_allocated_bytes"].iloc[0]
                )
                / (1024.0**3)
                for budget, _, _ in budgets
            ]
            bars = memory_ax.bar(
                x + offset,
                values,
                width,
                facecolor=face,
                edgecolor=color,
                linewidth=0.95,
                label=label,
                zorder=3,
            )
            memory_ax.bar_label(bars, fmt="%.2g", padding=2, fontsize=5.8)
            maximum = max(maximum, max(values))
        memory_ax.set_xticks(x, [label for _, label, _ in budgets])
        memory_ax.set_xlabel("Support budget")
        memory_ax.set_ylabel("Peak device memory (GiB/rank)")
        memory_ax.set_ylim(0.0, maximum * 1.20)
        clean_axis(memory_ax)
        memory_ax.legend(loc="upper left", fontsize=5.8, handlelength=1.5)
        memory_ax.text(
            0.97,
            0.92,
            f"{memory_rank} GPU ranks",
            transform=memory_ax.transAxes,
            ha="right",
            fontsize=5.9,
            color=NEUTRALS["mid_dark"],
        )

    fig.subplots_adjust(left=0.075, right=0.985, top=0.985, bottom=0.125)
    save_figure(fig, "fig4_distributed_performance")


def figure_s_countsketch() -> None:
    data_path = DATA / "hpc_pt2_countsketch_scan.csv"
    if not data_path.exists():
        return
    data = pd.read_csv(data_path)
    fig, axes = plt.subplots(
        1, 3, figsize=(7.15, 2.55), gridspec_kw={"wspace": 0.38}
    )

    width_scan = data[data["replicas"] == 8].sort_values("buckets")
    replica_scan = data[
        (data["buckets"] == 65536) & (data["replicas"] >= 8)
    ].sort_values("replicas")
    scans = [
        (axes[0], width_scan, "buckets", "Sketch buckets", "a"),
        (axes[1], replica_scan, "replicas", "Sketch replicas", "b"),
    ]
    for ax, frame, x_name, x_label, letter in scans:
        panel_label(ax, letter)
        ax.axhline(0.0, color=NEUTRALS["mid"], linewidth=0.8)
        ax.errorbar(
            frame[x_name],
            frame["delta_vs_exact_mha_median"],
            yerr=frame["reported_standard_error_mha_median"],
            color=COLORS["external"],
            linewidth=0.8,
            alpha=0.30,
            capsize=4.0,
            zorder=1,
        )
        ax.errorbar(
            frame[x_name],
            frame["delta_vs_exact_mha_median"],
            yerr=frame["delta_vs_exact_mha_std"],
            color=COLORS["external"],
            marker="o",
            markerfacecolor=NEUTRALS["white"],
            markeredgecolor=COLORS["external"],
            linewidth=1.2,
            capsize=2.0,
            zorder=2,
        )
        ax.set_xscale("log", base=2)
        if x_name == "buckets":
            tick_labels = [
                f"{int(value / 1024)}k" if value < 100000 else f"{int(round(value / 1024))}k"
                for value in frame[x_name]
            ]
            ax.set_xticks(frame[x_name], tick_labels, rotation=38, ha="right")
        else:
            ax.set_xticks(frame[x_name], [f"{int(value):,}" for value in frame[x_name]])
        ax.set_xlabel(x_label)
        ax.set_ylabel(r"$E^{(2)}_{\rm sketch}-E^{(2)}_{\rm exact}$ (m$E_h$)")
        clean_axis(ax)
        if letter == "a":
            ax.legend(
                handles=[
                    Line2D(
                        [0], [0], color=COLORS["external"], alpha=0.30,
                        linewidth=3.0, label="Reported replica SE",
                    ),
                    Line2D(
                        [0], [0], color=COLORS["external"], marker="o",
                        markerfacecolor=NEUTRALS["white"], linewidth=1.2,
                        label="Hash-seed spread",
                    ),
                ],
                loc="lower right",
            )

    ax = axes[2]
    panel_label(ax, "c")
    ordered = data.sort_values("sketch_entries")
    ax.plot(
        ordered["sketch_entries"],
        ordered["sketch_total_sec_median"],
        color=COLORS["restricted"],
        marker="o",
        linestyle="none",
        label="Reduction time",
    )
    ax.set_xscale("log", base=2)
    ax.set_xlabel(r"Sketch entries $B\times R$")
    ax.set_ylabel("Sketch reduction time (s)")
    clean_axis(ax)
    memory_ax = ax.twinx()
    memory_ax.plot(
        ordered["sketch_entries"],
        ordered["peak_device_allocated_bytes"] / 1.0e9,
        color=COLORS["full"],
        marker="s",
        linestyle="none",
        label="Peak device memory",
    )
    memory_ax.set_ylabel("Peak device memory (GB/rank)", color=COLORS["full"])
    memory_ax.tick_params(axis="y", colors=COLORS["full"])
    memory_ax.spines["top"].set_visible(False)
    handles = ax.get_lines() + memory_ax.get_lines()
    ax.legend(handles, [line.get_label() for line in handles], loc="upper left")
    save_figure(fig, "fig_s_countsketch_convergence")




def _plot_seed_envelope(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    color: str,
    marker: str,
    label: str,
    linestyle: str = "-",
    absolute: bool = False,
    numerical_floor: float = 1.0e-7,
) -> None:
    rows = frame.copy()
    if absolute:
        rows["plot_value"] = np.maximum(
            np.abs(rows["error_microhartree"].to_numpy(dtype=float)),
            numerical_floor,
        )
    else:
        rows["plot_value"] = rows["error_microhartree"].to_numpy(dtype=float)
    summary = (
        rows.groupby("order", as_index=False)["plot_value"]
        .agg(["min", "median", "max"])
        .reset_index()
        .sort_values("order")
    )
    x = summary["order"].to_numpy(dtype=float)
    median = summary["median"].to_numpy(dtype=float)
    lower = summary["min"].to_numpy(dtype=float)
    upper = summary["max"].to_numpy(dtype=float)
    ax.fill_between(x, lower, upper, color=color, alpha=0.13, linewidth=0)
    ax.plot(
        x,
        median,
        color=color,
        marker=marker,
        markerfacecolor=NEUTRALS["white"],
        markeredgewidth=0.9,
        linestyle=linestyle,
        label=label,
        zorder=3,
    )


def figure_results1_pt_generated_rr() -> None:
    """Connect large-space PT4 warnings to an enumerable-space autopsy."""

    pt, _ = load_pt_generated_rr_hierarchy()
    if pt.empty:
        return
    scan_path = DATA / "n2_negative_axis_scan_seed334.csv"
    intruder_path = DATA / "small_intruder_summary.csv"
    large_warning_path = DATA / "large_pt34_warning.csv"
    if (
        not scan_path.exists()
        or not intruder_path.exists()
        or not large_warning_path.exists()
    ):
        return
    scan = pd.read_csv(scan_path)
    intruders = pd.read_csv(intruder_path).set_index("system")
    large_warning = pd.read_csv(large_warning_path)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.0, 5.08),
        gridspec_kw={"wspace": 0.32, "hspace": 0.43},
    )
    large_ax = axes[0, 0]
    large_order = ("h2o_ccpvdz", "n2_ccpvdz")
    latest = (
        large_warning.sort_values("step")
        .groupby("system", as_index=False)
        .tail(1)
        .set_index("system")
        .loc[list(large_order)]
    )
    orders = np.asarray([2.0, 3.0, 4.0])
    large_styles = {
        "h2o_ccpvdz": (COLORS["healthy"], "o", r"H$_2$O (30k)"),
        "n2_ccpvdz": (COLORS["pt_path"], "s", r"N$_2$ (30k)"),
    }
    for system, (color, marker, label) in large_styles.items():
        row = latest.loc[system]
        errors = row[["pt2_error_mha", "pt3_error_mha", "pt4_error_mha"]].to_numpy(
            dtype=float
        )
        large_ax.plot(
            orders,
            errors,
            color=color,
            linewidth=1.25,
            marker=marker,
            markersize=4.6,
            markerfacecolor=NEUTRALS["white"],
            markeredgewidth=1.0,
            label=label,
            zorder=3,
        )
    large_ax.axhline(
        0.0,
        color=NEUTRALS["mid_dark"],
        linewidth=0.85,
        zorder=1,
    )
    large_ax.text(
        4.02,
        0.0,
        "reference",
        color=NEUTRALS["mid_dark"],
        fontsize=6.0,
        ha="left",
        va="center",
        clip_on=False,
    )
    large_ax.set_ylim(-5.0, 5.0)
    large_ax.set_yticks([-4.0, -2.0, 0.0, 2.0, 4.0])
    large_ax.set_xlim(1.85, 4.18)
    large_ax.set_xticks(orders, ["PT2", "PT3", "PT4"])
    large_ax.set_ylabel(r"Signed energy error (m$E_h$)")
    large_ax.set_title("Large spaces: accurate PT2, oscillatory PT3/PT4")
    large_ax.legend(loc="upper left", frameon=False, fontsize=6.0)
    clean_axis(large_ax)
    panel_label(large_ax, "a")

    small_ax = axes[0, 1]
    partial = pt[pt["method"] == "PT partial sum"].copy()
    partial = partial[partial["system"].isin(("h2o_sto3g", "n2_sto3g", "lif_sto3g"))]
    partial = partial.sort_values(["system", "seed", "order"])
    groups = partial.groupby(["system", "seed"], sort=False)
    partial["coefficient"] = groups["error_microhartree"].diff()
    partial["rho"] = groups["coefficient"].transform(
        lambda values: values.abs() / values.shift(1).abs()
    )
    small_styles = {
        "h2o_sto3g": (COLORS["healthy"], "o", "-", r"H$_2$O"),
        "n2_sto3g": (COLORS["intruder"], "s", "-", r"N$_2$"),
        "lif_sto3g": (COLORS["pt_path"], "D", "--", "LiF"),
    }
    small_ax.axhspan(0.3, 1.0, color=SURFACES["aqua"], alpha=0.34, zorder=0)
    small_ax.axhspan(1.0, 1.8, color=SURFACES["coral"], alpha=0.30, zorder=0)
    for system, (color, marker, linestyle, label) in small_styles.items():
        rows = partial[(partial["system"] == system) & partial["rho"].notna()]
        summary = (
            rows.groupby("order", as_index=False)["rho"]
            .agg(["min", "median", "max"])
            .reset_index()
            .sort_values("order")
        )
        order = summary["order"].to_numpy(dtype=float)
        median = summary["median"].to_numpy(dtype=float)
        small_ax.fill_between(
            order,
            summary["min"].to_numpy(dtype=float),
            summary["max"].to_numpy(dtype=float),
            color=color,
            alpha=0.11,
            linewidth=0.0,
        )
        small_ax.plot(
            order,
            median,
            color=color,
            linewidth=1.0,
            linestyle=linestyle,
            marker=marker,
            markersize=3.5,
            markevery=4,
            markerfacecolor=NEUTRALS["white"],
            markeredgewidth=0.75,
            label=label,
            zorder=3,
        )
    small_ax.axhline(
        1.0,
        color=NEUTRALS["mid_dark"],
        linewidth=0.75,
        linestyle=":",
        zorder=1,
    )
    small_ax.text(
        31.8,
        1.69,
        r"sustained $\rho_n>1$: divergent",
        color=NEUTRALS["charcoal"],
        fontsize=6.0,
        ha="right",
        va="top",
    )
    small_ax.text(
        31.8,
        0.40,
        r"$\rho_n<1$: terms decay",
        color=NEUTRALS["charcoal"],
        fontsize=6.0,
        ha="right",
        va="bottom",
    )
    small_ax.set_xlim(3.5, 32.5)
    small_ax.set_xticks([4, 8, 16, 24, 32])
    small_ax.set_ylim(0.3, 1.8)
    small_ax.set_yticks([0.5, 1.0, 1.5])
    small_ax.set_xlabel("Perturbation order $n$")
    small_ax.set_ylabel(r"Successive coefficient ratio $\rho_n$")
    small_ax.set_title("Enumerable spaces: coefficient-growth test")
    small_ax.legend(loc="lower left", frameon=False, ncol=3, fontsize=6.0)
    clean_axis(small_ax)
    panel_label(small_ax, "b")

    analytic_ax = axes[1, 0]
    analytic_ax.axvspan(
        -1.0,
        1.0,
        color=SURFACES["vanilla"],
        alpha=0.42,
        zorder=0,
    )
    analytic_ax.axvline(
        -1.0,
        color=NEUTRALS["mid"],
        linewidth=0.75,
        linestyle=":",
        zorder=0,
    )
    analytic_ax.axvline(
        1.0,
        color=NEUTRALS["mid"],
        linewidth=0.75,
        linestyle=":",
        zorder=0,
    )
    radius_rows = (
        (
            r"H$_2$O",
            2.0,
            float(intruders.loc["h2o_sto3g", "r_ds"]),
            r"$R_{\rm DS}=1.754$",
            None,
            COLORS["healthy"],
            True,
        ),
        (
            r"N$_2$",
            1.0,
            float(intruders.loc["n2_sto3g", "r_ds"]),
            r"$R_{\rm DS}=0.778$",
            float(intruders.loc["n2_sto3g", "eta_ac"]),
            COLORS["pt_path"],
            False,
        ),
        (
            "LiF",
            0.0,
            float(intruders.loc["lif_sto3g", "r_ds"]),
            r"$R_{\rm DS}=0.616$",
            float(intruders.loc["lif_sto3g", "eta_ac"]),
            COLORS["pt_path"],
            False,
        ),
    )
    for label, yi, radius, annotation, intruder_eta, interval_color, success in radius_rows:
        analytic_ax.fill_betweenx(
            [yi - 0.09, yi + 0.09],
            -radius,
            radius,
            color=interval_color,
            alpha=0.10,
            linewidth=0.0,
            zorder=1,
        )
        analytic_ax.hlines(
            yi,
            -radius,
            radius,
            color=interval_color,
            linewidth=0.9,
            zorder=2,
        )
        analytic_ax.vlines(
            [-radius, radius],
            yi - 0.09,
            yi + 0.09,
            color=interval_color,
            linewidth=1.35,
            zorder=2,
        )
        if intruder_eta is not None:
            analytic_ax.plot(
                intruder_eta,
                yi,
                marker="x",
                markersize=7.6,
                color=NEUTRALS["white"],
                markeredgewidth=3.1,
                linestyle="None",
                zorder=4,
            )
            analytic_ax.plot(
                intruder_eta,
                yi,
                marker="x",
                markersize=6.3,
                color=COLORS["intruder"],
                markeredgewidth=1.75,
                linestyle="None",
                zorder=5,
            )
        analytic_ax.annotate(
            "",
            xy=(1.0, yi),
            xytext=(0.0, yi),
            arrowprops={
                "arrowstyle": "-|>",
                "color": NEUTRALS["charcoal"],
                "linewidth": 1.05,
                "linestyle": "-" if success else (0, (1.6, 1.0)),
                "mutation_scale": 8.5,
                "shrinkA": 0.0,
                "shrinkB": 0.0,
            },
            zorder=4,
        )
        analytic_ax.plot(
            0.0,
            yi,
            marker="o",
            markersize=3.8,
            markerfacecolor=NEUTRALS["charcoal"],
            markeredgecolor=NEUTRALS["charcoal"],
            markeredgewidth=0.6,
            linestyle="None",
            zorder=5,
        )
        analytic_ax.text(
            radius - 0.03,
            yi + 0.17,
            annotation,
            ha="right",
            va="center",
            fontsize=6.0,
            color=interval_color,
        )
    analytic_ax.set_xlim(-2.0, 2.0)
    analytic_ax.set_ylim(-0.32, 2.42)
    analytic_ax.set_xticks([-2.0, -1.0, 0.0, 1.0, 2.0])
    analytic_ax.set_yticks([0.0, 1.0, 2.0], ["LiF", r"N$_2$", r"H$_2$O"])
    analytic_ax.set_xlabel(r"Real path coordinate $\eta$")
    analytic_ax.set_title("Taylor-radius control on the real path")
    clean_axis(analytic_ax)
    panel_label(analytic_ax, "c")

    scan_spec = axes[1, 1].get_subplotspec()
    axes[1, 1].remove()
    scan_grid = scan_spec.subgridspec(
        2,
        1,
        height_ratios=(1.25, 1.0),
        hspace=0.08,
    )
    scan_gap_ax = fig.add_subplot(scan_grid[0])
    scan_weight_ax = fig.add_subplot(scan_grid[1], sharex=scan_gap_ax)
    local_scan = scan[(scan["eta"] >= -0.76) & (scan["eta"] <= -0.60)].copy()
    local_scan = local_scan.sort_values("eta")
    gap_mha = 1.0e3 * (local_scan["e1_hartree"] - local_scan["e0_hartree"])
    crossing_eta = -0.675
    crossing_gap = 0.829
    scan_gap_ax.plot(
        local_scan["eta"],
        gap_mha,
        color=COLORS["pt_path"],
        linewidth=1.15,
        marker="o",
        markersize=3.8,
        markerfacecolor=NEUTRALS["white"],
        markeredgewidth=0.8,
        zorder=3,
    )
    scan_gap_ax.scatter(
        [crossing_eta],
        [crossing_gap],
        s=27,
        marker="D",
        facecolor=NEUTRALS["black"],
        edgecolor=NEUTRALS["black"],
        linewidth=0.7,
        zorder=4,
    )
    scan_gap_ax.annotate(
        r"$\eta_{\rm ac}=-0.675$" "\n" r"$\Delta_{01}^{\min}=0.829$ m$E_h$",
        xy=(crossing_eta, crossing_gap),
        xytext=(-0.748, 3.0),
        arrowprops={"arrowstyle": "-", "color": NEUTRALS["mid_dark"], "lw": 0.65},
        ha="left",
        va="bottom",
        fontsize=6.0,
        color=NEUTRALS["charcoal"],
    )
    scan_gap_ax.set_yscale("log")
    scan_gap_ax.set_ylim(0.55, 150.0)
    scan_gap_ax.set_yticks([1.0, 10.0, 100.0])
    scan_gap_ax.set_ylabel(r"Gap $\Delta_{01}$ (m$E_h$)")
    scan_gap_ax.set_title(r"N$_2$/STO-3G: negative-axis avoided crossing")
    scan_gap_ax.tick_params(axis="x", which="both", labelbottom=False)

    scan_weight_ax.plot(
        local_scan["eta"],
        local_scan["w0"],
        color=COLORS["healthy"],
        linewidth=1.05,
        marker="o",
        markersize=3.6,
        markerfacecolor=NEUTRALS["white"],
        markeredgewidth=0.75,
        label=r"$|\langle x|\psi_0\rangle|^2$",
        zorder=3,
    )
    scan_weight_ax.plot(
        local_scan["eta"],
        local_scan["w1"],
        color=COLORS["pt_path"],
        linewidth=1.05,
        marker="s",
        markersize=3.6,
        markerfacecolor=NEUTRALS["white"],
        markeredgewidth=0.75,
        label=r"$|\langle x|\psi_1\rangle|^2$",
        zorder=3,
    )
    for scan_ax in (scan_gap_ax, scan_weight_ax):
        scan_ax.axvline(
            crossing_eta,
            color=NEUTRALS["mid_dark"],
            linewidth=0.75,
            linestyle=":",
            zorder=1,
        )
        clean_axis(scan_ax)
    scan_weight_ax.set_ylim(-0.05, 1.05)
    scan_weight_ax.set_yticks([0.0, 0.5, 1.0])
    scan_weight_ax.set_ylabel("Reference\nweight")
    scan_weight_ax.set_xlim(-0.76, -0.60)
    scan_weight_ax.set_xticks([-0.75, -0.70, -0.65, -0.60])
    scan_weight_ax.set_xlabel(r"Path coordinate $\eta$")
    scan_weight_ax.legend(
        loc="center left",
        frameon=False,
        fontsize=6.0,
        ncol=2,
        handlelength=1.5,
        columnspacing=0.8,
    )
    panel_label(scan_gap_ax, "d")

    fig.subplots_adjust(top=0.94, bottom=0.09)
    save_figure(fig, "fig_results1_pt_generated_rr")


def main() -> None:
    configure_style()
    figure1_method()
    figure2_large_trajectories()
    figure_large_root_locality()
    figure3_sample_budget()
    figure4_hpc()
    figure_s_countsketch()
    figure_results1_pt_generated_rr()
    print(f"wrote paper figures to {OUT}")


if __name__ == "__main__":
    main()
