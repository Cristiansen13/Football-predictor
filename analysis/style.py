"""One look for every figure: quiet axes, thin marks, a fixed colour order."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files, never open a window
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = ROOT / "docs" / "figures"
RESULTS_DIR = ROOT / "analysis" / "results"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # series 1, 2, 3, always in this order
REFERENCE = "#898781"  # things that are not ours: baselines, the bookmaker
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"]  # light to dark, for magnitudes
OUTCOME_COLORS = {"Home win": BLUE, "Draw": ORANGE, "Away win": AQUA}


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"], "font.size": 10,
        "text.color": INK, "axes.labelcolor": INK_SECONDARY, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "semibold", "axes.titlelocation": "left", "axes.titlepad": 10,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK_SECONDARY, "ytick.labelcolor": INK_SECONDARY,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "lines.linewidth": 2, "lines.markersize": 7, "lines.solid_capstyle": "round",
        "legend.frameon": False, "legend.fontsize": 9,
    })


def season_label(season: int) -> str:
    """2023 -> '2023/24'."""
    return f"{season}/{(season + 1) % 100:02d}"


def save(fig, name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path
