"""Animate the four night-map heatmaps as a function of a per-game
bucket attribute (quality rank, or average pre-game queen skill).

Usage:
    python animate_by_bucket.py quality
    python animate_by_bucket.py skill

For each bucket the four densities are recomputed and rendered as one
frame; output is 4 mp4 files (sharp, dk, killer-split, killer-diff).
A static N-panel summary png is also written for each.

Buckets are quantiles of the per-game attribute, so each bucket holds
~1/N of the games. Default N = 12.
"""
import argparse
import os
import pickle
import sys

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, TwoSlopeNorm
from PIL import Image
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "night_kills.parquet")
RATINGS = os.path.join(HERE, "ratings_queen_drone.pkl")
MAP_PNG = os.path.join(HERE, "maps/Night.png")

X_MAX, Y_MAX, X_HALF = 1920, 1080, 960
GRID_X, GRID_Y = 480, 540
TARGET_SIGMA_PX = 24
NUM_BUCKETS = 12


# ---------- data ------------------------------------------------------

def load_folded() -> pd.DataFrame:
    df = pd.read_parquet(DATA).copy()
    xc = df["x_canon"].to_numpy(np.float64)
    df["x_fold"] = np.where(xc <= X_HALF, xc, X_MAX - xc).astype(np.float64)
    df["y_screen"] = (Y_MAX - df["y"]).astype(np.float64)
    return df


def attach_quality(df: pd.DataFrame) -> pd.DataFrame:
    """The bin file is sorted by quality_score DESC, so rank 0 = highest
    quality. We invert so larger quality_pct => higher quality."""
    games = df.groupby("game_id")["rank"].first()
    qpct = 1.0 - games.rank(method="first") / len(games)
    df = df.copy()
    df["quality_pct"] = df["game_id"].map(qpct)
    return df


def attach_queen_skill(df: pd.DataFrame) -> pd.DataFrame:
    """Average pre-game queen mu (positions 1 and 2) per game.
    Pickle layout: {game_id: ndarray of 10 floats indexed by position-1}."""
    print(f"  loading ratings from {RATINGS}", file=sys.stderr)
    with open(RATINGS, "rb") as f:
        ratings_by_game: dict = pickle.load(f)
    # Position 1 -> index 0 (Gold queen), position 2 -> index 1 (Blue queen)
    skill = {gid: 0.5 * (arr[0] + arr[1])
             for gid, arr in ratings_by_game.items()}
    print(f"  computed queen skill for {len(skill):,} games",
          file=sys.stderr)
    df = df.copy()
    df["queen_mu"] = df["game_id"].map(skill)
    df = df.dropna(subset=["queen_mu"])
    games = df.groupby("game_id")["queen_mu"].first()
    qpct = games.rank(method="first") / len(games)
    df["skill_pct"] = df["game_id"].map(qpct)
    return df


def make_buckets(df: pd.DataFrame, col: str, n: int) -> list[pd.DataFrame]:
    edges = np.linspace(0, 1, n + 1)
    return [df[(df[col] >= edges[i]) & (df[col] < edges[i + 1] + (1e-9 if i == n - 1 else 0))]
            for i in range(n)]


# ---------- KDE -------------------------------------------------------

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


def colorize(density, cmap_name, norm, alpha_max=0.92, gamma=0.55):
    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(norm(density))
    p = np.clip(norm(density), 0.0, 1.0)
    p = np.where(np.isfinite(p), p, 0.0)
    rgba[..., 3] = np.where(density > 0, alpha_max * np.power(p, gamma), 0.0)
    return rgba


def colorize_div(value, cmap_name, vlim, mask, alpha_max=0.85):
    cmap = plt.get_cmap(cmap_name)
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)
    rgba = cmap(norm(value))
    a = np.clip(np.abs(value) / vlim, 0.0, 1.0)
    a = np.where(np.isfinite(a), a, 0.0) * mask.astype(np.float64)
    rgba[..., 3] = a * alpha_max
    return rgba


# ---------- per-bucket density computation ----------------------------

def bucket_densities(buckets: list[pd.DataFrame], mask, sigma_cells,
                     cell_area: float):
    """Returns list of dicts with 'sharp', 'D', 'K', 'Dq', 'Ds' arrays."""
    out = []
    for b in buckets:
        deaths = b[b["killed_cat"] == "Queen"]
        qkills = b[b["killer_pid"].isin([1, 2])]
        by_q = deaths[deaths["killer_pid"].isin([1, 2])]
        by_s = deaths[deaths["killer_pid"] >= 3]

        Hd = hist2d(deaths["x_fold"].to_numpy(np.float64),
                    deaths["y_screen"].to_numpy(np.float64))
        Hk = hist2d(qkills["x_fold"].to_numpy(np.float64),
                    qkills["y_screen"].to_numpy(np.float64))
        Hq = hist2d(by_q["x_fold"].to_numpy(np.float64),
                    by_q["y_screen"].to_numpy(np.float64))
        Hs = hist2d(by_s["x_fold"].to_numpy(np.float64),
                    by_s["y_screen"].to_numpy(np.float64))

        D = diffusion_kde(Hd, mask, sigma_cells) / cell_area
        K = diffusion_kde(Hk, mask, sigma_cells) / cell_area
        Dq = diffusion_kde(Hq, mask, sigma_cells) / cell_area
        Ds = diffusion_kde(Hs, mask, sigma_cells) / cell_area

        out.append(dict(
            n_games=b["game_id"].nunique(),
            n_deaths=len(deaths), n_qkills=len(qkills),
            n_by_q=len(by_q), n_by_s=len(by_s),
            D=D, K=K, Dq=Dq, Ds=Ds,
        ))
    return out


# ---------- common figure helpers -------------------------------------

def left_half_map():
    img = Image.open(MAP_PNG).convert("RGB").transpose(Image.FLIP_LEFT_RIGHT)
    return np.array(img)[:, :X_HALF]


def setup_axes(ax, title=""):
    ax.set_xlim(0, X_HALF)
    ax.set_ylim(Y_MAX, 0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(20, 50, "GOLD", color="#ffd166", fontsize=12, fontweight="bold")
    if title:
        ax.set_title(title)


# ---------- four animations -------------------------------------------

def animate_sharp(densities, bucket_label_fn, bucket_attr, out_path):
    Ds = [d["D"] for d in densities]
    pooled = np.concatenate([d[d > 0].ravel() for d in Ds])
    vmin = float(np.quantile(pooled, 0.40))
    vmax = float(np.quantile(pooled, 0.999))
    norm = LogNorm(vmin=vmin, vmax=vmax)
    img = left_half_map()

    fig, ax = plt.subplots(figsize=(8, 9), constrained_layout=True)
    ax.imshow(img, extent=[0, X_HALF, Y_MAX, 0], aspect="equal")
    overlay = ax.imshow(
        colorize(Ds[0], "inferno", norm),
        extent=[0, X_HALF, Y_MAX, 0], aspect="equal",
        interpolation="nearest",
    )
    setup_axes(ax)
    ttl = ax.set_title("")
    sm = plt.cm.ScalarMappable(cmap="inferno", norm=norm)
    fig.colorbar(sm, ax=ax, shrink=0.8, label="density (kills / px²)")

    def update(i):
        overlay.set_data(colorize(Ds[i], "inferno", norm))
        d = densities[i]
        ttl.set_text(
            f"Queen-death density — {bucket_attr} bucket {bucket_label_fn(i)}\n"
            f"{d['n_games']:,} games · {d['n_deaths']:,} queen deaths"
        )
        return overlay, ttl

    anim = animation.FuncAnimation(fig, update, frames=len(Ds),
                                   interval=600, blit=False)
    anim.save(out_path, writer="ffmpeg", fps=2, dpi=110)
    plt.close(fig)


def animate_dk(densities, bucket_label_fn, bucket_attr, out_path, mask):
    eps = 1e-9
    centered_list = []
    sig_masks = []
    for d in densities:
        D, K = d["D"], d["K"]
        log_ratio = np.log10((D + eps) / (K + eps))
        global_log = np.log10(d["n_deaths"] / max(d["n_qkills"], 1))
        centered = log_ratio - global_log
        total = D + K
        sig = (total > np.quantile(total[mask & (total > 0)], 0.20)) & mask
        centered_list.append(centered)
        sig_masks.append(sig)

    pooled = np.concatenate([
        c[s].ravel() for c, s in zip(centered_list, sig_masks)
    ])
    vlim = max(np.quantile(np.abs(pooled), 0.99), 0.2)

    img = left_half_map()
    fig, ax = plt.subplots(figsize=(8, 9), constrained_layout=True)
    ax.imshow(img, extent=[0, X_HALF, Y_MAX, 0], aspect="equal")
    overlay = ax.imshow(
        colorize_div(centered_list[0], "RdBu_r", vlim, sig_masks[0]),
        extent=[0, X_HALF, Y_MAX, 0], aspect="equal",
        interpolation="nearest",
    )
    setup_axes(ax)
    ttl = ax.set_title("")
    sm = plt.cm.ScalarMappable(
        cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim))
    fig.colorbar(sm, ax=ax, shrink=0.8,
                 label="log10(D/K) − global mean")

    def update(i):
        overlay.set_data(colorize_div(
            centered_list[i], "RdBu_r", vlim, sig_masks[i]))
        d = densities[i]
        ttl.set_text(
            f"D/K deviation — {bucket_attr} bucket {bucket_label_fn(i)}\n"
            f"{d['n_deaths']:,} deaths · {d['n_qkills']:,} kills-by-queen"
        )
        return overlay, ttl

    anim = animation.FuncAnimation(fig, update, frames=len(centered_list),
                                   interval=600, blit=False)
    anim.save(out_path, writer="ffmpeg", fps=2, dpi=110)
    plt.close(fig)


def animate_killer_split(densities, bucket_label_fn, bucket_attr, out_path):
    Dqs = [d["Dq"] for d in densities]
    Dss = [d["Ds"] for d in densities]
    pooled_q = np.concatenate([d[d > 0].ravel() for d in Dqs])
    pooled_s = np.concatenate([d[d > 0].ravel() for d in Dss])
    vmin = max(np.quantile(pooled_q, 0.40), np.quantile(pooled_s, 0.40))
    vmax = max(np.quantile(pooled_q, 0.999), np.quantile(pooled_s, 0.999))
    norm = LogNorm(vmin=vmin, vmax=vmax)
    img = left_half_map()

    fig, axes = plt.subplots(1, 2, figsize=(15, 8.5), constrained_layout=True)
    overlays = []
    titles = []
    for ax in axes:
        ax.imshow(img, extent=[0, X_HALF, Y_MAX, 0], aspect="equal")
        ov = ax.imshow(
            colorize(Dqs[0], "inferno", norm),
            extent=[0, X_HALF, Y_MAX, 0], aspect="equal",
            interpolation="nearest",
        )
        setup_axes(ax)
        titles.append(ax.set_title(""))
        overlays.append(ov)
    sm = plt.cm.ScalarMappable(cmap="inferno", norm=norm)
    fig.colorbar(sm, ax=axes, shrink=0.6, label="density (kills / px²)")
    suptitle = fig.suptitle("", fontsize=12)

    def update(i):
        overlays[0].set_data(colorize(Dqs[i], "inferno", norm))
        overlays[1].set_data(colorize(Dss[i], "inferno", norm))
        d = densities[i]
        titles[0].set_text(f"killed by enemy QUEEN ({d['n_by_q']:,})")
        titles[1].set_text(f"killed by enemy SOLDIER ({d['n_by_s']:,})")
        suptitle.set_text(
            f"Queen-death by killer — {bucket_attr} bucket {bucket_label_fn(i)}"
        )
        return overlays + titles + [suptitle]

    anim = animation.FuncAnimation(fig, update, frames=len(Dqs),
                                   interval=600, blit=False)
    anim.save(out_path, writer="ffmpeg", fps=2, dpi=110)
    plt.close(fig)


def animate_killer_diff(densities, bucket_label_fn, bucket_attr, out_path,
                        mask):
    eps = 1e-9
    centered_list = []
    sig_masks = []
    for d in densities:
        Dq, Ds = d["Dq"], d["Ds"]
        log_ratio = np.log10((Ds + eps) / (Dq + eps))
        global_log = np.log10(d["n_by_s"] / max(d["n_by_q"], 1))
        centered = log_ratio - global_log
        total = Dq + Ds
        sig = (total > np.quantile(total[mask & (total > 0)], 0.20)) & mask
        centered_list.append(centered)
        sig_masks.append(sig)

    pooled = np.concatenate([
        c[s].ravel() for c, s in zip(centered_list, sig_masks)
    ])
    vlim = max(np.quantile(np.abs(pooled), 0.99), 0.2)
    img = left_half_map()

    fig, ax = plt.subplots(figsize=(8, 9), constrained_layout=True)
    ax.imshow(img, extent=[0, X_HALF, Y_MAX, 0], aspect="equal")
    overlay = ax.imshow(
        colorize_div(centered_list[0], "RdBu_r", vlim, sig_masks[0]),
        extent=[0, X_HALF, Y_MAX, 0], aspect="equal",
        interpolation="nearest",
    )
    setup_axes(ax)
    ttl = ax.set_title("")
    sm = plt.cm.ScalarMappable(
        cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim))
    fig.colorbar(sm, ax=ax, shrink=0.8,
                 label="log10(soldier/queen) − global")

    def update(i):
        overlay.set_data(colorize_div(
            centered_list[i], "RdBu_r", vlim, sig_masks[i]))
        d = densities[i]
        ttl.set_text(
            f"Killer-type imbalance — {bucket_attr} bucket {bucket_label_fn(i)}\n"
            f"{d['n_by_q']:,} by-queen · {d['n_by_s']:,} by-soldier"
        )
        return overlay, ttl

    anim = animation.FuncAnimation(fig, update, frames=len(centered_list),
                                   interval=600, blit=False)
    anim.save(out_path, writer="ffmpeg", fps=2, dpi=110)
    plt.close(fig)


# ---------- main ------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("attr", choices=["quality", "skill"])
    p.add_argument("--num-buckets", type=int, default=NUM_BUCKETS)
    args = p.parse_args()

    df = load_folded()
    print(f"Loaded {len(df):,} kills from {df['game_id'].nunique():,} "
          f"games", file=sys.stderr)

    if args.attr == "quality":
        df = attach_quality(df)
        col, label = "quality_pct", "quality"
        slug = "quality"
    else:
        df = attach_queen_skill(df)
        col, label = "skill_pct", "queen-skill"
        slug = "skill"

    n = args.num_buckets
    buckets = make_buckets(df, col, n)
    print(f"  built {n} {label} buckets; sizes: "
          f"{[b['game_id'].nunique() for b in buckets]}", file=sys.stderr)

    # Build playable mask once from the whole sample
    mask = gaussian_filter(
        hist2d(df["x_fold"].to_numpy(np.float64),
               df["y_screen"].to_numpy(np.float64)),
        sigma=2.0) > 0.01
    cell_px = X_HALF / GRID_X
    sigma_cells = TARGET_SIGMA_PX / cell_px
    cell_area = cell_px * cell_px

    print("Computing per-bucket densities...", file=sys.stderr)
    dens = bucket_densities(buckets, mask, sigma_cells, cell_area)

    def bucket_label(i):
        edges = np.linspace(0, 1, n + 1)
        return f"{int(edges[i] * 100)}–{int(edges[i + 1] * 100)}%"

    out = lambda name: os.path.join(HERE, f"anim_{slug}_{name}.mp4")

    print("Rendering sharp animation...", file=sys.stderr)
    animate_sharp(dens, bucket_label, label, out("sharp"))
    print(f"  -> {out('sharp')}", file=sys.stderr)

    print("Rendering D/K animation...", file=sys.stderr)
    animate_dk(dens, bucket_label, label, out("dk"), mask)
    print(f"  -> {out('dk')}", file=sys.stderr)

    print("Rendering killer-split animation...", file=sys.stderr)
    animate_killer_split(dens, bucket_label, label, out("killer_split"))
    print(f"  -> {out('killer_split')}", file=sys.stderr)

    print("Rendering killer-diff animation...", file=sys.stderr)
    animate_killer_diff(dens, bucket_label, label, out("killer_diff"), mask)
    print(f"  -> {out('killer_diff')}", file=sys.stderr)


if __name__ == "__main__":
    main()
