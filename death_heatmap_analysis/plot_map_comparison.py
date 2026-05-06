"""Cross-map comparison of queen-death peak density.

For each of the 4 maps:
  1. Build folded queen-death histogram (gold-half, mirror-folded).
  2. Smooth with wall-respecting diffusion using a per-map mask.
  3. Report peak density (kills / px²) — both raw and per-game.
  4. Concentration ratio = peak / mean(playable) — a unitless measure
     of how "spiky" the hottest cluster is.

Output:
  cross_map_heatmaps.png  — 2×2 grid, shared color scale
  cross_map_peaks.png     — bar chart: peak density and concentration
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from PIL import Image
from scipy.ndimage import gaussian_filter

DATA = os.path.join(os.path.dirname(__file__), "all_kills.parquet")
OUT_DIR = os.path.dirname(__file__)

X_MAX = 1920
Y_MAX = 1080
X_HALF = X_MAX // 2
GRID_X = 480     # 2 px / cell on half-frame
GRID_Y = 540
TARGET_SIGMA_PX = 24

MAP_NAMES = ["map_day", "map_night", "map_dusk", "map_twilight"]
MAP_LABELS = ["day", "night", "dusk", "twilight"]
_HERE = os.path.dirname(__file__)
MAP_PNGS = {
    "map_day": os.path.join(_HERE, "maps/Day.png"),
    "map_night": os.path.join(_HERE, "maps/Night.png"),
    "map_dusk": os.path.join(_HERE, "maps/Dusk.png"),
    "map_twilight": os.path.join(_HERE, "maps/twilight.png"),
}


def hist2d(x, y):
    H, _, _ = np.histogram2d(
        x, y, bins=[GRID_X, GRID_Y], range=[[0, X_HALF], [0, Y_MAX]],
    )
    return H.T


def diffusion_kde(H, mask, target_sigma_cells, alpha=0.2):
    iters = max(1, int(round(target_sigma_cells ** 2 / alpha)))
    f = (H * mask).astype(np.float64)
    m = mask.astype(np.float64)
    pad_m = np.pad(m, 1, mode="constant")
    nbr_count = (pad_m[:-2, 1:-1] + pad_m[2:, 1:-1]
                 + pad_m[1:-1, :-2] + pad_m[1:-1, 2:])
    safe = nbr_count > 0
    for _ in range(iters):
        pad_f = np.pad(f, 1, mode="constant")
        nbr_sum = (pad_f[:-2, 1:-1] + pad_f[2:, 1:-1]
                   + pad_f[1:-1, :-2] + pad_f[1:-1, 2:])
        nbr_avg = np.zeros_like(f)
        nbr_avg[safe] = nbr_sum[safe] / nbr_count[safe]
        f = ((1 - alpha) * f + alpha * nbr_avg) * mask
    return f


def colorize(density, cmap, norm, alpha_max=0.92, gamma=0.55):
    cmap = plt.get_cmap(cmap)
    rgba = cmap(norm(density))
    p = np.clip(norm(density), 0.0, 1.0)
    p = np.where(np.isfinite(p), p, 0.0)
    a = alpha_max * np.power(p, gamma)
    rgba[..., 3] = np.where(density > 0, a, 0.0)
    return rgba


def load_left_half_map(name: str) -> np.ndarray:
    """All four screenshots ship in blue-on-left orientation; we mirror
    to gold-on-left then crop to the gold half."""
    img = Image.open(MAP_PNGS[name]).convert("RGB")
    img = img.resize((X_MAX, Y_MAX)).transpose(Image.FLIP_LEFT_RIGHT)
    return np.array(img)[:, :X_HALF]


def fold_x(x_canon: np.ndarray) -> np.ndarray:
    return np.where(x_canon <= X_HALF, x_canon, X_MAX - x_canon)


def main():
    df = pd.read_parquet(DATA)
    print(f"Loaded {len(df):,} kills total across maps")
    cell_px = X_HALF / GRID_X
    sigma_cells = TARGET_SIGMA_PX / cell_px
    cell_area = cell_px * cell_px

    # Per-map computation
    densities = {}
    metrics = []  # rows for summary table
    for midx, name in enumerate(MAP_NAMES):
        sub = df[df["map_idx"] == midx]
        n_games = sub["game_id"].nunique()
        deaths = sub[sub["killed_cat"] == "Queen"]
        if len(deaths) == 0:
            continue
        x = fold_x(deaths["x_canon"].to_numpy(np.float64))
        y = (Y_MAX - deaths["y"]).to_numpy(np.float64)

        # Playable mask: per-map, from ALL kills in that map
        x_all = fold_x(sub["x_canon"].to_numpy(np.float64))
        y_all = (Y_MAX - sub["y"]).to_numpy(np.float64)
        H_all = hist2d(x_all, y_all)
        mask = gaussian_filter(H_all, sigma=2.0) > 0.01

        H_q = hist2d(x, y)
        density = diffusion_kde(H_q, mask, sigma_cells) / cell_area

        # Per-game density (so peaks are comparable across maps)
        density_per_game = density / max(n_games, 1)

        peak = float(density_per_game[mask].max())
        playable_cells = int(mask.sum())
        playable_area = playable_cells * cell_area
        mean_in_playable = float(density_per_game[mask].mean())
        concentration = peak / mean_in_playable if mean_in_playable > 0 else np.nan

        densities[name] = density_per_game
        metrics.append({
            "map": MAP_LABELS[midx],
            "n_games": n_games,
            "queen_deaths": len(deaths),
            "deaths_per_game": len(deaths) / max(n_games, 1),
            "playable_area_px2": playable_area,
            "peak_density": peak,
            "mean_density": mean_in_playable,
            "concentration_ratio": concentration,
        })
    metrics_df = pd.DataFrame(metrics)
    print("\nSummary:")
    pd.set_option("display.float_format", lambda v: f"{v:,.6g}")
    print(metrics_df.to_string(index=False))

    # ---- Heatmap grid with SHARED color scale ----
    # Use the highest-peak map as the reference top of the scale.
    all_vals = np.concatenate([d[d > 0].ravel() for d in densities.values()])
    vmax = float(np.quantile(all_vals, 0.999))
    vmin = float(np.quantile(all_vals, 0.40))
    norm = LogNorm(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(2, 2, figsize=(14, 16),
                             constrained_layout=True)
    for ax, (midx, name) in zip(axes.flat, enumerate(MAP_NAMES)):
        if name not in densities:
            ax.axis("off")
            continue
        bg = load_left_half_map(name)
        ax.imshow(bg, extent=[0, X_HALF, Y_MAX, 0], aspect="equal")
        rgba = colorize(densities[name], "inferno", norm)
        ax.imshow(rgba, extent=[0, X_HALF, Y_MAX, 0], aspect="equal",
                  interpolation="nearest")
        ax.set_xlim(0, X_HALF); ax.set_ylim(Y_MAX, 0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(20, 50, "GOLD", color="#ffd166", fontsize=12,
                fontweight="bold")
        m = metrics_df[metrics_df["map"] == MAP_LABELS[midx]].iloc[0]
        ax.set_title(
            f"{MAP_LABELS[midx]}  |  "
            f"peak={m['peak_density']:.2e}  "
            f"concentration×{m['concentration_ratio']:.1f}"
        )
    sm = plt.cm.ScalarMappable(cmap="inferno", norm=norm)
    fig.colorbar(sm, ax=axes, shrink=0.4,
                 label="density per game (kills / px² / game)")
    fig.suptitle(
        "Queen-death heatmap by map — gold half (mirror-folded)\n"
        f"shared color scale, σ_eff≈{TARGET_SIGMA_PX}px diffusion",
        fontsize=13,
    )
    out_grid = os.path.join(OUT_DIR, "cross_map_heatmaps.png")
    plt.savefig(out_grid, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_grid}")

    # ---- Peak density / concentration bar chart ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    order = metrics_df.sort_values("peak_density", ascending=False)["map"].tolist()
    md = metrics_df.set_index("map").loc[order]

    colors = ["#5fb0d4", "#7c69ad", "#e08750", "#f4c95d"]
    map_to_color = {"day": colors[0], "night": colors[1],
                    "dusk": colors[2], "twilight": colors[3]}
    bar_colors = [map_to_color[m] for m in order]

    axes[0].bar(order, md["peak_density"].values, color=bar_colors)
    axes[0].set_ylabel("peak density (kills / px² / game)")
    axes[0].set_title("Hottest queen-death cluster intensity")
    for i, v in enumerate(md["peak_density"].values):
        axes[0].text(i, v, f"{v:.2e}", ha="center", va="bottom", fontsize=9)

    axes[1].bar(order, md["concentration_ratio"].values, color=bar_colors)
    axes[1].set_ylabel("peak / mean (playable region)")
    axes[1].set_title("Concentration ratio")
    for i, v in enumerate(md["concentration_ratio"].values):
        axes[1].text(i, v, f"×{v:.1f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Peak queen-death cluster intensity by map", fontsize=13)
    out_bars = os.path.join(OUT_DIR, "cross_map_peaks.png")
    plt.savefig(out_bars, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_bars}")


if __name__ == "__main__":
    main()
