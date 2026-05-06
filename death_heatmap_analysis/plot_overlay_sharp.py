"""Geometry-aware (structure-preserving) queen-death heatmap.

Standard Gaussian KDE blurs density across walls and platforms — a kill
on top of a platform spreads into the stone below; a kill in the snail
chamber spreads into the side chambers across the central pillar.

We avoid this two ways, both data-driven (no hand-segmentation):

1. **Mask-renormalized Gaussian.** Build a "playable" mask M from the
   *empirical* total-kills histogram: cells where >=1 kill ever happened.
   Smooth as
       density = gaussian(H * M, sigma) / gaussian(M, sigma)
   masked outside M. This corrects boundary dilution but still leaks
   across thin walls.

2. **Wall-respecting diffusion.** Iteratively diffuse density only
   between adjacent playable cells (no flux into wall cells). Equivalent
   to solving the heat equation with reflective boundaries on the
   playable region. Variance after T iterations of α=0.2 Jacobi steps
   is sigma_eff² ≈ 4 α T cells² (one cell = 4 px), so we pick T to
   match a target sigma.

Both use a fine grid (960x540, 2 px/cell) so the renormalization can
resolve thin walls. Output: 3-panel comparison vs naive KDE.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from PIL import Image
from scipy.ndimage import gaussian_filter

DATA = os.path.join(os.path.dirname(__file__), "night_kills.parquet")
MAP_PNG = os.path.join(os.path.dirname(__file__), "maps/Night.png")
OUT = os.path.join(os.path.dirname(__file__),
                   "queen_death_overlay_sharp.png")

X_MAX = 1920
Y_MAX = 1080
GRID_X = 960   # 2 px / cell
GRID_Y = 540
TARGET_SIGMA_PX = 24
SAMPLE_GAMES = 10_000


def hist2d(x, y):
    H, _, _ = np.histogram2d(
        x, y, bins=[GRID_X, GRID_Y], range=[[0, X_MAX], [0, Y_MAX]],
    )
    return H.T  # rows=y, cols=x


def naive_kde(H, sigma_cells):
    return gaussian_filter(H, sigma=sigma_cells, mode="constant")


def masked_kde(H, mask, sigma_cells):
    """Mask-renormalized Gaussian: density inside playable region only,
    not diluted by zeros outside."""
    num = gaussian_filter(H * mask, sigma=sigma_cells, mode="constant")
    den = gaussian_filter(mask.astype(np.float64), sigma=sigma_cells,
                          mode="constant")
    out = np.zeros_like(num)
    safe = den > 1e-6
    out[safe] = num[safe] / den[safe]
    out[~mask] = 0.0
    return out


def diffusion_kde(H, mask, target_sigma_cells):
    """Iterative Jacobi diffusion with no-flux wall boundaries.

    Each iteration: each playable cell becomes (1 - α) * self
    + α * mean(playable neighbors). Equivalent to wave-blocked Gaussian.
    """
    alpha = 0.2  # 4-neighbor stability bound is 0.25
    # In 2D, variance per axis after T iters ≈ alpha * T cells² (each step
    # contributes α to variance via 4-neighbor split — empirically tuned)
    iters = max(1, int(round(target_sigma_cells ** 2 / alpha)))

    # Pad with playable=False borders
    f = (H * mask).astype(np.float64)
    m = mask.astype(np.float64)
    # Precompute neighbor mask sums for normalization
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


def sample_df(df, n_games):
    rng = np.random.default_rng(0)
    games = df["game_id"].unique()
    keep = rng.choice(games, size=min(n_games, len(games)), replace=False)
    return df[df["game_id"].isin(keep)]


def main():
    df = sample_df(pd.read_parquet(DATA), SAMPLE_GAMES)
    deaths = df[df["killed_cat"] == "Queen"]
    print(f"Sampled: {len(df):,} kills, {len(deaths):,} queen deaths")

    x_all = df["x_canon"].to_numpy(np.float64)
    y_all = (Y_MAX - df["y"]).to_numpy(np.float64)
    x_q = deaths["x_canon"].to_numpy(np.float64)
    y_q = (Y_MAX - deaths["y"]).to_numpy(np.float64)

    # Build playable mask from ALL kills (more samples => better mask)
    H_all = hist2d(x_all, y_all)
    # Slight dilation: smooth and threshold so single-kill cells aren't
    # isolated; this avoids speckled mask edges.
    H_all_smooth = gaussian_filter(H_all, sigma=2.0)
    mask = H_all_smooth > 0.01

    # Queen-death histogram on the same grid
    H_q = hist2d(x_q, y_q)

    cell_px = X_MAX / GRID_X
    sigma_cells = TARGET_SIGMA_PX / cell_px

    naive = naive_kde(H_q, sigma_cells)
    masked = masked_kde(H_q, mask, sigma_cells)
    print(f"Running wall-respecting diffusion ({TARGET_SIGMA_PX}px target)...")
    diff = diffusion_kde(H_q, mask, sigma_cells)

    # Convert to per-px² density for display
    cell_area = cell_px * cell_px
    naive /= cell_area
    masked /= cell_area
    diff /= cell_area

    # Shared color scale derived from masked variant (most robust)
    nz = masked[mask & (masked > 0)]
    vmin = np.quantile(nz, 0.40)
    vmax = np.quantile(nz, 0.999)

    img = np.array(Image.open(MAP_PNG).convert("RGB")
                   .transpose(Image.FLIP_LEFT_RIGHT))

    fig, axes = plt.subplots(3, 1, figsize=(13, 19), constrained_layout=True)
    titles = [
        f"naive Gaussian KDE (σ={TARGET_SIGMA_PX}px) — leaks across walls",
        f"mask-renormalized Gaussian (σ={TARGET_SIGMA_PX}px)",
        f"wall-respecting diffusion (σ_eff≈{TARGET_SIGMA_PX}px) — sharpest",
    ]
    def colorize(density, cmap_name, norm, alpha_max=0.92, gamma=0.55):
        cmap = plt.get_cmap(cmap_name)
        rgba = cmap(norm(density))
        p = np.clip(norm(density), 0.0, 1.0)
        p = np.where(np.isfinite(p), p, 0.0)
        rgba[..., 3] = np.where(density > 0, alpha_max * np.power(p, gamma), 0.0)
        return rgba

    norm = LogNorm(vmin=vmin, vmax=vmax)
    last_im = None
    for ax, density, title in zip(axes, [naive, masked, diff], titles):
        ax.imshow(img, extent=[0, X_MAX, Y_MAX, 0], aspect="equal")
        rgba = colorize(density, "inferno", norm)
        ax.imshow(rgba, extent=[0, X_MAX, Y_MAX, 0], aspect="equal",
                  interpolation="nearest")
        last_im = plt.cm.ScalarMappable(cmap="inferno", norm=norm)
        ax.set_title(title)
        ax.set_xlim(0, X_MAX); ax.set_ylim(Y_MAX, 0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(20, 50, "GOLD", color="#ffd166", fontsize=12,
                fontweight="bold")
        ax.text(X_MAX - 110, 50, "BLUE", color="#7ce0ff", fontsize=12,
                fontweight="bold")

    fig.colorbar(last_im, ax=axes, shrink=0.5, label="density (kills / px²)")
    fig.suptitle(
        f"Geometry-aware queen-death heatmap (night, {len(deaths):,} kills "
        f"from {SAMPLE_GAMES:,} games)\n"
        "playable region inferred from where ANY kill ever happened",
        fontsize=12,
    )
    plt.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
