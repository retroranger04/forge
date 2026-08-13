"""Generate the four README figures from coverage_results.json (no hardcoded numbers)."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = REPO_ROOT / "docs" / "figures"

COLOR_80 = "#4C6A92"
COLOR_90 = "#B8654A"
COLOR_95 = "#3B7B78"
COLOR_NEUTRAL = "#3A3A3A"

GRID_COLOR = "#E5E5E5"
AXIS_COLOR = "#333333"

LEVELS = ["80", "90", "95"]
LEVEL_COLORS = {"80": COLOR_80, "90": COLOR_90, "95": COLOR_95}


def style_axes(ax):
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("empirical coverage")
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS_COLOR)
    ax.spines["bottom"].set_color(AXIS_COLOR)
    ax.tick_params(direction="out", length=4, color=AXIS_COLOR, labelcolor=AXIS_COLOR)


def plot_axis_a(ax, data, legend=True):
    splits = data["splits"]
    id_cov = [splits["id_test"][f"coverage_at_{lv}"] for lv in LEVELS]
    ood_cov = [splits["ood_fidelity"][f"coverage_at_{lv}"] for lv in LEVELS]

    x = range(len(LEVELS))
    width = 0.35
    ax.bar(
        [i - width / 2 for i in x], id_cov, width,
        color=COLOR_80, edgecolor=COLOR_80, label="ID (plane stress)", zorder=3,
    )
    ax.bar(
        [i + width / 2 for i in x], ood_cov, width,
        color=COLOR_90, edgecolor=COLOR_90, label="OOD (plane strain)", zorder=3,
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(["80", "90", "95"])
    ax.set_xlabel("nominal coverage level")
    style_axes(ax)
    if legend:
        ax.legend(loc="best", fontsize=8, frameon=False)


def plot_axis_b(ax, data, legend=True):
    ratios = data["ratios"]
    r_values = [1.0, 1.1, 1.25, 1.5, 2.0, 3.0]
    keys = [f"R_{r}" for r in r_values]

    for lv in LEVELS:
        y = [ratios[k][f"coverage_at_{lv}"] for k in keys]
        ax.plot(
            r_values, y, color=LEVEL_COLORS[lv], linewidth=2.0,
            marker="o", markersize=7, markerfacecolor=LEVEL_COLORS[lv],
            markeredgecolor=LEVEL_COLORS[lv], label=f"nominal {lv}%", zorder=3,
        )
    ax.set_xlabel("R (E₁/E₂)")
    style_axes(ax)
    if legend:
        ax.legend(loc="best", fontsize=8, frameon=False)


def plot_axis_c(ax, data, legend=True):
    levels = data["levels"]
    p_values = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
    keys = [f"P_{p}" for p in p_values]

    for lv in LEVELS:
        y = [levels[k][f"coverage_at_{lv}"] for k in keys]
        ax.plot(
            p_values, y, color=LEVEL_COLORS[lv], linewidth=2.0,
            marker="o", markersize=7, markerfacecolor=LEVEL_COLORS[lv],
            markeredgecolor=LEVEL_COLORS[lv], label=f"nominal {lv}%", zorder=3,
        )
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlabel("P (% of σ_ref)")
    style_axes(ax)
    if legend:
        ax.legend(loc="best", fontsize=8, frameon=False)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    axis_a = json.loads((REPO_ROOT / "outputs/run_01/axis_4_plane_strain/coverage_results.json").read_text())
    axis_b = json.loads((REPO_ROOT / "outputs/run_01/axis_anisotropy_sweep/coverage_results.json").read_text())
    axis_c = json.loads((REPO_ROOT / "outputs/run_01/axis_residual_stress_sweep/coverage_results.json").read_text())

    # Figure 1
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")
    plot_axis_a(ax, axis_a)
    ax.set_title("Plane stress trained, plane strain tested", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "axis_a_plane_strain.png", dpi=150)
    plt.close(fig)

    # Figure 2
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")
    plot_axis_b(ax, axis_b)
    ax.set_title("Material anisotropy sweep", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "axis_b_anisotropy.png", dpi=150)
    plt.close(fig)

    # Figure 3
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")
    plot_axis_c(ax, axis_c)
    ax.set_title("Residual pre-stress sweep", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "axis_c_residual_stress.png", dpi=150)
    plt.close(fig)

    # Figure 4: headline panel, per-subplot compact legends
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("white")
    plot_axis_a(axes[0], axis_a, legend=True)
    axes[0].set_title("Physical regime", fontsize=11)
    plot_axis_b(axes[1], axis_b, legend=True)
    axes[1].set_title("Material anisotropy", fontsize=11)
    plot_axis_c(axes[2], axis_c, legend=True)
    axes[2].set_title("Residual pre-stress", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "headline_panel.png", dpi=150)
    plt.close(fig)

    for name in ["axis_a_plane_strain.png", "axis_b_anisotropy.png", "axis_c_residual_stress.png", "headline_panel.png"]:
        path = FIG_DIR / name
        print(f"{name}: {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
