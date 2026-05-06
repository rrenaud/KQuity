"""Left-half-folded heatmaps with alpha-modulated overlay so the
underlying map geometry remains visible in low-signal regions.

Renders three views:
  1. queen-death density  (sequential, geometry-aware sharpening)
  2. D/K ratio centered on global mean (diverging)
  3. killer split: by-queen vs by-soldier, side-by-side (sequential)

Each game's kills are mirror-folded about x=960 so both halves stack on
the gold side, doubling effective n in the visible area.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, TwoSlopeNorm
from PIL import Image
from scipy.ndimage import gaussian_filter

DATA = os.path.join(os.path.dirname(__file__), "night_kills.parquet")
MAP_PNG = os.path.join(os.path.dirname(__file__), "maps/Night.png")
OUT_DIR = os.path.dirname(__file__)

X_MAX = 1920
Y_MAX = 1080
X_HALF = X_MAX // 2  # 960

GRID_X = 480     # 2 px / cell on the half-frame
GRID_Y = 540
TARGET_SIGMA_PX = 24
SAMPLE_GAMES = 10_000


# ----- data prep ------------------------------------------------------

def load_folded():
    df = pd.read_parquet(DATA)
    rng = np.random.default_rng(0)
    games = df["game_id"].unique()
    keep = rng.choice(games, size=min(SAMPLE_GAMES, len(games)),
                      replace=False)
    df = df[df["game_id"].isin(keep)].copy()
    # Fold blue side onto gold side
    xc = df["x_canon"].to_numpy(np.float64)
    df["x_fold"] = np.where(xc <= X_HALF, xc, X_MAX - xc)
    df["y_screen"] = (Y_MAX - df["y"]).astype(np.float64)
    return df


def hist2d(x, y):
    H, _, _ = np.histogram2d(
        x, y, bins=[GRID_X, GRID_Y],
        range=[[0, X_HALF], [0, Y_MAX]],
    )
    return H.T


def masked_kde(H, mask, sigma_cells):
    num = gaussian_filter(H * mask, sigma=sigma_cells, mode="constant")
    den = gaussian_filter(mask.astype(np.float64), sigma=sigma_cells,
                          mode="constant")
    out = np.zeros_like(num)
    safe = den > 1e-6
    out[safe] = num[safe] / den[safe]
    out[~mask] = 0.0
    return out


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


# ----- alpha-modulated overlay ----------------------------------------

def colorize(density, cmap, norm, alpha_max=0.92, gamma=0.55):
    """RGBA with alpha = alpha_max * progress^gamma where progress is
    norm(density) clipped to [0,1]. gamma<1 keeps mid-range visible while
    letting low-density cells fade so the geometry shows through. Zero
    density and NaN are fully transparent.
    """
    cmap = plt.get_cmap(cmap)
    rgba = cmap(norm(density))
    progress = np.clip(norm(density), 0.0, 1.0)
    progress = np.where(np.isfinite(progress), progress, 0.0)
    a = alpha_max * np.power(progress, gamma)
    rgba[..., 3] = np.where(density > 0, a, 0.0)
    return rgba


def colorize_diverging(value, cmap, vlim, alpha_max=0.85, mask=None):
    """Diverging cmap centered at 0: alpha rises with |value|/vlim.
    Cells at 0 deviation become fully transparent."""
    cmap = plt.get_cmap(cmap)
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)
    rgba = cmap(norm(value))
    a = np.clip(np.abs(value) / vlim, 0.0, 1.0)
    a = np.where(np.isfinite(a), a, 0.0)
    if mask is not None:
        a = a * mask
    rgba[..., 3] = a * alpha_max
    return rgba


# ----- map background -------------------------------------------------

def load_left_half_map():
    img = Image.open(MAP_PNG).convert("RGB").transpose(Image.FLIP_LEFT_RIGHT)
    arr = np.array(img)[:, :X_HALF]
    return arr


# ----- plots ----------------------------------------------------------

def plot_density_sharp(df, mask, cell_px, sigma_cells):
    deaths = df[df["killed_cat"] == "Queen"]
    print(f"  density-sharp: {len(deaths):,} queen deaths")
    H = hist2d(deaths["x_fold"].to_numpy(np.float64),
               deaths["y_screen"].to_numpy(np.float64))
    diff = diffusion_kde(H, mask, sigma_cells) / (cell_px ** 2)

    nz = diff[mask & (diff > 0)]
    vmin = np.quantile(nz, 0.40)
    vmax = np.quantile(nz, 0.999)
    norm = LogNorm(vmin=vmin, vmax=vmax)
    rgba = colorize(diff, "inferno", norm, alpha_max=0.85)

    img_bg = load_left_half_map()
    fig, ax = plt.subplots(figsize=(8, 9), constrained_layout=True)
    ax.imshow(img_bg, extent=[0, X_HALF, Y_MAX, 0], aspect="equal")
    im = ax.imshow(rgba, extent=[0, X_HALF, Y_MAX, 0], aspect="equal",
                   interpolation="nearest")
    # ScalarMappable for colorbar
    sm = plt.cm.ScalarMappable(cmap="inferno", norm=norm)
    fig.colorbar(sm, ax=ax, shrink=0.8, label="density (kills / px²)")
    ax.set_xlim(0, X_HALF); ax.set_ylim(Y_MAX, 0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(20, 50, "GOLD", color="#ffd166", fontsize=13, fontweight="bold")
    ax.set_title(
        f"Queen-death heatmap — gold half (mirror-folded)\n"
        f"wall-respecting diffusion, σ_eff≈{TARGET_SIGMA_PX}px, "
        f"{len(deaths):,} kills"
    )
    out = os.path.join(OUT_DIR, "queen_death_left_sharp.png")
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def plot_dk_ratio(df, mask, cell_px, sigma_cells):
    deaths = df[df["killed_cat"] == "Queen"]
    qkills = df[df["killer_pid"].isin([1, 2])]
    print(f"  D/K: D={len(deaths):,}, K={len(qkills):,}")

    Hd = hist2d(deaths["x_fold"].to_numpy(np.float64),
                deaths["y_screen"].to_numpy(np.float64))
    Hk = hist2d(qkills["x_fold"].to_numpy(np.float64),
                qkills["y_screen"].to_numpy(np.float64))

    D = diffusion_kde(Hd, mask, sigma_cells) / (cell_px ** 2)
    K = diffusion_kde(Hk, mask, sigma_cells) / (cell_px ** 2)

    eps = 1e-9
    log_ratio = np.log10((D + eps) / (K + eps))
    global_log = np.log10(len(deaths) / max(len(qkills), 1))
    centered = log_ratio - global_log

    # Suppress noise where total signal is weak
    total = D + K
    sig_mask = (total > np.quantile(total[mask & (total > 0)], 0.20)) & mask

    flat = centered[sig_mask]
    vlim = max(np.quantile(np.abs(flat), 0.99), 0.2)

    rgba = colorize_diverging(centered, "RdBu_r", vlim,
                              alpha_max=0.85,
                              mask=sig_mask.astype(np.float64))

    img_bg = load_left_half_map()
    fig, ax = plt.subplots(figsize=(8, 9), constrained_layout=True)
    ax.imshow(img_bg, extent=[0, X_HALF, Y_MAX, 0], aspect="equal")
    ax.imshow(rgba, extent=[0, X_HALF, Y_MAX, 0], aspect="equal",
              interpolation="nearest")
    sm = plt.cm.ScalarMappable(
        cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim))
    cb = fig.colorbar(sm, ax=ax, shrink=0.8,
                      label="log10(D/K) − global mean")
    ax.set_xlim(0, X_HALF); ax.set_ylim(Y_MAX, 0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(20, 50, "GOLD", color="#ffd166", fontsize=13, fontweight="bold")
    ax.set_title(
        f"Queen kill-zone vs. death-zone — gold half\n"
        f"global log10(D/K)={global_log:+.2f}    "
        "red = die more than expected, blue = deadlier than expected"
    )
    out = os.path.join(OUT_DIR, "queen_dk_ratio_left.png")
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def plot_queen_minus_soldier(df, mask, cell_px, sigma_cells):
    """Where do enemy queens kill queens vs. enemy soldiers kill queens?
    Shows log10(D_queen / D_soldier) centered on the global ratio so red
    = soldier-killer dominates relative to map average, blue = queen-
    killer dominates."""
    deaths = df[df["killed_cat"] == "Queen"]
    by_q = deaths[deaths["killer_pid"].isin([1, 2])]
    by_s = deaths[deaths["killer_pid"] >= 3]

    Hq = hist2d(by_q["x_fold"].to_numpy(np.float64),
                by_q["y_screen"].to_numpy(np.float64))
    Hs = hist2d(by_s["x_fold"].to_numpy(np.float64),
                by_s["y_screen"].to_numpy(np.float64))
    Dq = diffusion_kde(Hq, mask, sigma_cells) / (cell_px ** 2)
    Ds = diffusion_kde(Hs, mask, sigma_cells) / (cell_px ** 2)

    eps = 1e-9
    # Convention: positive = soldier dominates locally (relative to mean),
    # negative = queen-killer dominates. So we plot log10(Ds / Dq) - mean.
    log_ratio = np.log10((Ds + eps) / (Dq + eps))
    global_log = np.log10(len(by_s) / max(len(by_q), 1))
    centered = log_ratio - global_log

    total = Dq + Ds
    sig_mask = (total > np.quantile(total[mask & (total > 0)], 0.20)) & mask
    flat = centered[sig_mask]
    vlim = max(np.quantile(np.abs(flat), 0.99), 0.2)

    rgba = colorize_diverging(centered, "RdBu_r", vlim,
                              alpha_max=0.85,
                              mask=sig_mask.astype(np.float64))

    img_bg = load_left_half_map()
    fig, ax = plt.subplots(figsize=(8, 9), constrained_layout=True)
    ax.imshow(img_bg, extent=[0, X_HALF, Y_MAX, 0], aspect="equal")
    ax.imshow(rgba, extent=[0, X_HALF, Y_MAX, 0], aspect="equal",
              interpolation="nearest")
    sm = plt.cm.ScalarMappable(
        cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim))
    fig.colorbar(sm, ax=ax, shrink=0.8,
                 label="log10(D_soldier / D_queen) − global mean")
    ax.set_xlim(0, X_HALF); ax.set_ylim(Y_MAX, 0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(20, 50, "GOLD", color="#ffd166", fontsize=13, fontweight="bold")
    ax.set_title(
        f"Killer-type imbalance (queen-deaths) — gold half\n"
        f"global log10(soldier/queen)={global_log:+.2f}    "
        "red = soldier-killer dominates locally, blue = queen-killer dominates"
    )
    out = os.path.join(OUT_DIR, "queen_death_killer_diff_left.png")
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def plot_killer_split(df, mask, cell_px, sigma_cells):
    deaths = df[df["killed_cat"] == "Queen"]
    by_q = deaths[deaths["killer_pid"].isin([1, 2])]
    by_s = deaths[deaths["killer_pid"] >= 3]
    print(f"  killer-split: by-queen={len(by_q):,}, by-soldier={len(by_s):,}")

    def density(sub):
        H = hist2d(sub["x_fold"].to_numpy(np.float64),
                   sub["y_screen"].to_numpy(np.float64))
        return diffusion_kde(H, mask, sigma_cells) / (cell_px ** 2)

    Dq = density(by_q)
    Ds = density(by_s)

    # Shared color scale
    nzq = Dq[mask & (Dq > 0)]
    nzs = Ds[mask & (Ds > 0)]
    vmin = max(np.quantile(nzq, 0.40), np.quantile(nzs, 0.40))
    vmax = max(np.quantile(nzq, 0.999), np.quantile(nzs, 0.999))
    norm = LogNorm(vmin=vmin, vmax=vmax)

    img_bg = load_left_half_map()
    fig, axes = plt.subplots(1, 2, figsize=(14, 8.5),
                             constrained_layout=True)
    titles = [
        f"killed by enemy QUEEN ({len(by_q):,} events)",
        f"killed by enemy SOLDIER ({len(by_s):,} events)",
    ]
    for ax, dens, title in zip(axes, [Dq, Ds], titles):
        ax.imshow(img_bg, extent=[0, X_HALF, Y_MAX, 0], aspect="equal")
        rgba = colorize(dens, "inferno", norm, alpha_max=0.85)
        ax.imshow(rgba, extent=[0, X_HALF, Y_MAX, 0], aspect="equal",
                  interpolation="nearest")
        ax.set_xlim(0, X_HALF); ax.set_ylim(Y_MAX, 0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(20, 50, "GOLD", color="#ffd166", fontsize=12,
                fontweight="bold")
        ax.set_title(title)
    sm = plt.cm.ScalarMappable(cmap="inferno", norm=norm)
    fig.colorbar(sm, ax=axes, shrink=0.6, label="density (kills / px²)")
    fig.suptitle(
        "Queen-death heatmap by killer type — gold half (mirror-folded)",
        fontsize=12,
    )
    out = os.path.join(OUT_DIR, "queen_death_by_killer_left.png")
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def main():
    df = load_folded()
    print(f"Sample: {len(df):,} kills from {df['game_id'].nunique():,} games")

    # Build playable mask from all kills (any category)
    H_all = hist2d(df["x_fold"].to_numpy(np.float64),
                   df["y_screen"].to_numpy(np.float64))
    H_all_smooth = gaussian_filter(H_all, sigma=2.0)
    mask = H_all_smooth > 0.01

    cell_px = X_HALF / GRID_X  # = 2.0
    sigma_cells = TARGET_SIGMA_PX / cell_px

    plot_density_sharp(df, mask, cell_px, sigma_cells)
    plot_dk_ratio(df, mask, cell_px, sigma_cells)
    plot_killer_split(df, mask, cell_px, sigma_cells)
    plot_queen_minus_soldier(df, mask, cell_px, sigma_cells)


if __name__ == "__main__":
    main()
