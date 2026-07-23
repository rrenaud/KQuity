#!/usr/bin/env python3
"""Model-free break-even from raw game outcomes.

Instead of scoring counterfactual states with the win-prob model, value each
state by its *empirical* win rate over real games (the worker_state_values
methodology), on a relative joint state oriented to the attacker:

    state = (defender queen lives, net warrior edge)   [marginalized over
             attacker queen lives and everything else]

    winrate[de, nw] = P(attacker's team eventually wins | that bucket),
                      accumulated from both attacking perspectives.

Combat outcomes are transitions between *observed* buckets:
    warrior -> queen : kill = winrate[de-1, nw] (or 1.0 at de=0, terminal win)
                       death = winrate[de, nw-1]   (attacker warrior dies)
    warrior -> warrior: kill = winrate[de, nw+1]   (defender warrior dies)
                        death = winrate[de, nw-1]

    p* = (V(S) - V_death) / (V_kill - V_death)

Limitations vs. the model: raw buckets can't resolve vanilla vs. speed warriors
(every winged piece is a "warrior"), and a transition-target bucket's win rate
is only a proxy for the true counterfactual.

Usage:
  python -m combat_value.empirical [--data ...] [--max-games N]
                                   [--html path] [--open]
"""
from __future__ import annotations

import argparse
import html as _html
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from combat_value import core, analysis
from combat_value.html_report import diverging_color, text_on

NW_MAX = 6                     # clip net warriors to [-6, 6]
NWN = 2 * NW_MAX + 1
EGG_MAX = 2
EMP_MIN = 500                  # min states per bucket to trust a win rate
MIN_SWING = 0.02               # empirical win rates are noisy: need a real gap

# Display grid (matches the model heatmap for comparison)
EGG_ROWS = [2, 1, 0]
NET_COLS = [-3, -2, -1, 0, 1, 2, 3, 4]


def new_winrate_accum():
    """Zeroed (wins, counts) arrays of shape (3, NWN)."""
    return (np.zeros((EGG_MAX + 1, NWN), np.float64),
            np.zeros((EGG_MAX + 1, NWN), np.float64))


def accumulate_winrate(X, y, wins, cnt):
    """Add one batch of states into empirical (wins, counts) accumulators.

    Buckets are (def_eggs, net_warriors), indexed [def_eggs, net+NW_MAX],
    filled from both attacking perspectives. y == 1 means blue won.
    """
    blue_eggs = np.clip(X[:, core.EGGS].astype(np.int64), 0, EGG_MAX)
    gold_eggs = np.clip(X[:, core.TEAM_OFFSET['gold'] + core.EGGS].astype(np.int64), 0, EGG_MAX)
    blue_war = X[:, core.N_VANILLA] + X[:, core.N_SPEED]
    gold_war = X[:, core.TEAM_OFFSET['gold'] + core.N_VANILLA] + \
        X[:, core.TEAM_OFFSET['gold'] + core.N_SPEED]
    yb = y.astype(np.float64)

    def add(def_eggs, net_war, win):
        nw = np.clip(np.rint(net_war).astype(np.int64), -NW_MAX, NW_MAX) + NW_MAX
        np.add.at(cnt, (def_eggs, nw), 1.0)
        np.add.at(wins, (def_eggs, nw), win)

    add(gold_eggs, blue_war - gold_war, yb)          # attacker = blue
    add(blue_eggs, gold_war - blue_war, 1.0 - yb)    # attacker = gold


def finalize_winrate(wins, cnt):
    """(wins, counts) -> winrate with NaN where counts < EMP_MIN."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(cnt >= EMP_MIN, wins / cnt, np.nan)


def _wr(winrate, de, nw):
    """winrate[def_eggs=de, net_warriors=nw] with bounds -> NaN."""
    if de < 0 or de > EGG_MAX:
        return np.nan
    idx = nw + NW_MAX
    if idx < 0 or idx >= NWN:
        return np.nan
    return winrate[de, idx]


def break_even_grid(winrate, matchup: str) -> dict:
    """p* per (def_eggs, net_warriors) for 'warrior_queen' or 'warrior_warrior'."""
    out = {}
    for de in range(EGG_MAX + 1):
        for nw in range(-NW_MAX, NW_MAX + 1):
            vS = _wr(winrate, de, nw)
            if matchup == 'warrior_queen':
                vK = 1.0 if de == 0 else _wr(winrate, de - 1, nw)
                vD = _wr(winrate, de, nw - 1)      # attacker warrior dies
            elif matchup == 'warrior_warrior':
                vK = _wr(winrate, de, nw + 1)       # defender warrior dies
                vD = _wr(winrate, de, nw - 1)
            else:
                raise ValueError(matchup)
            if not (np.isfinite(vS) and np.isfinite(vK) and np.isfinite(vD)):
                continue
            swing = vK - vD
            pstar = np.nan if abs(swing) < MIN_SWING else (vS - vD) / swing
            cnt_idx = nw + NW_MAX
            out[(de, nw)] = {
                'def_eggs': de, 'net_warriors': nw, 'pstar': float(pstar),
                'vS': float(vS), 'vK': float(vK), 'vD': float(vD),
            }
    return out


MODEL_MIN = 200  # min states per model bucket to display


def new_model_accum():
    """Zeroed per-bucket sums (sum_vS, sum_vK, sum_vD, count), shape (3, NWN)."""
    return [np.zeros((EGG_MAX + 1, NWN), np.float64) for _ in range(4)]


def accumulate_model(X, predict, atk_piece, dfn_piece, gids, ts, sums):
    """Add one batch's model V-sums into per-bucket accumulators (in place)."""
    res = core.evaluate_matchup_both_sides(X, predict, atk_piece, dfn_piece,
                                           game_ids=gids, timestamps=ts)
    m = analysis.valid_mask(res)
    de = np.clip(res.feats['def_eggs'][m], 0, EGG_MAX)
    nw = np.clip(res.feats['net_warriors'][m], -NW_MAX, NW_MAX) + NW_MAX
    svS, svK, svD, cn = sums
    np.add.at(svS, (de, nw), res.vS[m])
    np.add.at(svK, (de, nw), res.vK[m])
    np.add.at(svD, (de, nw), res.vD[m])
    np.add.at(cn, (de, nw), 1.0)


def finalize_model_grid(sums) -> dict:
    """Per-bucket mean V's and p* (from means), keyed (def_eggs, net_warriors)."""
    svS, svK, svD, cn = sums
    out = {}
    for de in range(EGG_MAX + 1):
        for j in range(NWN):
            n = cn[de, j]
            if n < MODEL_MIN:
                continue
            mS, mK, mD = svS[de, j] / n, svK[de, j] / n, svD[de, j] / n
            swing = mK - mD
            pstar = float('nan') if abs(swing) < MIN_SWING else (mS - mD) / swing
            out[(de, j - NW_MAX)] = {
                'pstar': pstar, 'mean_V_status_quo': mS,
                'mean_V_kill': mK, 'mean_V_death': mD, 'n': int(n),
            }
    return out


# ---------------------------------------------------------------- text output
def print_table(grid: dict, title: str):
    print(f"\n=== {title} ===")
    print("eggs  " + "  ".join(f"{nc:>+5d}" for nc in NET_COLS))
    for eg in EGG_ROWS:
        cells = []
        for nc in NET_COLS:
            c = grid.get((eg, nc))
            cells.append(f"{c['pstar']:>5.2f}" if c and np.isfinite(c['pstar']) else "    ·")
        print(f"  {eg}   " + "  ".join(cells))


# ---------------------------------------------------------------- html output
def _grid_html(grid: dict, min_n_key=None) -> str:
    out = ['<table class="grid"><thead><tr><th class="corner">'
           '<span class="ax-y">queen<br>lives</span>'
           '<span class="ax-x">warrior edge →</span></th>']
    for nc in NET_COLS:
        out.append(f'<th>{"+" if nc > 0 else ""}{nc}</th>')
    out.append('</tr></thead><tbody>')
    for eg in EGG_ROWS:
        out.append(f'<tr><th class="rowh">{eg}</th>')
        for nc in NET_COLS:
            c = grid.get((eg, nc))
            p = None if c is None else c['pstar']
            if c is None or p is None or not np.isfinite(p) or abs(p) > 1.6:
                out.append('<td class="empty"></td>')
                continue
            rgb = diverging_color(p)
            tip = _html.escape(json.dumps({
                'def_eggs': eg, 'net_warriors': nc, 'pstar': round(float(p), 3),
                'vS': round(c['vS'], 3), 'vK': round(c['vK'], 3), 'vD': round(c['vD'], 3),
            }))
            out.append(
                f'<td class="cell" style="background:rgb({rgb[0]},{rgb[1]},{rgb[2]});'
                f'color:{text_on(rgb)}" data-tip="{tip}">'
                f'<span class="p">{p:.2f}</span></td>')
        out.append('</tr>')
    out.append('</tbody></table>')
    return ''.join(out)


def _model_cell(cell) -> dict | None:
    if cell is None or not np.isfinite(cell.get('pstar', np.nan)):
        return None
    return {'pstar': cell['pstar'], 'vS': cell['mean_V_status_quo'],
            'vK': cell['mean_V_kill'], 'vD': cell['mean_V_death']}


SECTIONS_TMPL = '''<section class="mrow">
  <h2>{title}</h2>
  <div class="cap">{cap}</div>
  <div class="pair">
    <figure><figcaption>Empirical &mdash; raw game outcomes</figcaption>{emp}</figure>
    <figure><figcaption>Model &mdash; {model_lbl}</figcaption>{mdl}</figure>
  </div>
</section>'''


def build_html(emp_grids, mdl_grids, meta) -> str:
    sections = []
    for key, title, cap, model_lbl in [
        ('warrior_queen', 'Warrior → queen',
         'Empirical: attacker warrior dies (net−1) vs. queen dies (def eggs−1, or a win at last life). '
         'Model reference uses a vanilla warrior.', 'vanilla warrior → queen'),
        ('warrior_warrior', 'Warrior → warrior',
         'A symmetric trade: kill (net+1) vs. die (net−1). Model reference is vanilla vs. vanilla.',
         'vanilla → vanilla'),
    ]:
        emp = _grid_html(emp_grids[key])
        mdl = _grid_html({k: _model_cell(v) for k, v in mdl_grids[key].items()})
        sections.append(SECTIONS_TMPL.format(title=title, cap=cap, emp=emp, mdl=mdl,
                                             model_lbl=model_lbl))
    return PAGE.format(meta=meta, sections='\n'.join(sections))


PAGE = '''<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Break-even: empirical vs model — KQuity</title>
<style>
:root {{ color-scheme:light dark; --plane:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b;
  --ink2:#52514e; --mut:#898781; --line:#e1e0d9; --ring:rgba(11,11,11,.10); }}
@media (prefers-color-scheme: dark) {{ :root {{ --plane:#0d0d0d; --surface:#1a1a19;
  --ink:#fff; --ink2:#c3c2b7; --mut:#898781; --line:#2c2c2a; --ring:rgba(255,255,255,.10); }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--plane); color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif; line-height:1.5; }}
.wrap {{ max-width:1080px; margin:0 auto; padding:40px 24px 72px; }}
h1 {{ font-size:26px; margin:0 0 6px; letter-spacing:-.01em; }}
.sub {{ color:var(--ink2); max-width:74ch; margin:0 0 4px; }}
.meta {{ color:var(--mut); font-size:13px; margin-top:10px; }}
.legend {{ display:flex; align-items:center; gap:14px; margin:18px 0 8px; flex-wrap:wrap; }}
.bar {{ height:14px; width:260px; border-radius:7px; border:1px solid var(--ring);
  background:linear-gradient(90deg, rgb(16,66,129), rgb(42,120,214) 25%,
    rgb(240,239,236) 50%, rgb(227,73,72) 75%, rgb(150,28,27)); }}
.legend .lab {{ font-size:12px; color:var(--ink2); }}
.mrow {{ border-top:1px solid var(--line); padding-top:20px; margin-top:26px; }}
.mrow h2 {{ font-size:18px; margin:0; }}
.mrow .cap {{ color:var(--ink2); font-size:12.5px; margin:2px 0 14px; max-width:80ch; }}
.pair {{ display:grid; grid-template-columns:1fr 1fr; gap:22px; }}
@media (max-width:720px) {{ .pair {{ grid-template-columns:1fr; }} }}
figure {{ margin:0; background:var(--surface); border:1px solid var(--ring);
  border-radius:12px; padding:14px 16px 16px; }}
figcaption {{ font-size:13px; color:var(--ink2); margin-bottom:10px; font-weight:600; }}
table.grid {{ border-collapse:separate; border-spacing:3px; width:100%; }}
table.grid th {{ font-weight:500; color:var(--mut); font-size:12px; font-variant-numeric:tabular-nums; }}
th.rowh {{ text-align:right; padding-right:6px; color:var(--ink2); }}
.ax-y {{ display:block; font-size:10px; line-height:1.05; color:var(--mut); text-align:right; padding-right:6px; }}
.ax-x {{ display:block; font-size:10px; color:var(--mut); text-align:right; padding-right:6px; margin-top:2px; }}
td.cell {{ height:40px; border-radius:5px; text-align:center; vertical-align:middle;
  cursor:default; font-variant-numeric:tabular-nums; }}
td.cell:hover {{ outline:2px solid var(--ink); outline-offset:-2px; }}
td.cell .p {{ font-size:14px; font-weight:600; }}
td.empty {{ height:40px; border-radius:5px; opacity:.5;
  background:repeating-linear-gradient(45deg,transparent,transparent 4px,var(--line) 4px,var(--line) 5px); }}
.foot {{ margin-top:34px; color:var(--ink2); font-size:13px; max-width:80ch; }}
.foot b {{ color:var(--ink); font-weight:600; }}
#tip {{ position:fixed; pointer-events:none; z-index:10; background:var(--ink); color:var(--plane);
  padding:9px 11px; border-radius:8px; font-size:12px; line-height:1.5; opacity:0;
  transition:opacity .08s; box-shadow:0 6px 24px rgba(0,0,0,.28); font-variant-numeric:tabular-nums; }}
#tip .k {{ color:var(--mut); }}
</style></head><body>
<div class="wrap">
<h1>Break-even: empirical vs. model</h1>
<p class="sub">The same break-even p*, computed two ways. <b>Left</b>: model-free —
each state's value is its raw empirical win rate over real games, and combat
outcomes are transitions between observed buckets. <b>Right</b>: the LightGBM
win-probability model scoring counterfactual state edits. Rows are the defender's
queen lives; columns are the attacker's warrior edge.</p>
<div class="meta">{meta}</div>
<div class="legend">
  <div><div class="bar"></div></div>
  <div class="lab"><b>blue</b> = worth it at long odds · <b>gray</b> = even-odds · <b>red</b> = only when heavily favored</div>
</div>
{sections}
<div class="foot">
  <p><b>Why they can differ.</b> The empirical grid marginalizes over everything
  the bucket doesn't name (attacker queen lives, snail, berries, absolute counts)
  and can't tell a vanilla from a speed warrior. It also inherits selection: the
  states you <i>reach</i> at a given eggs/edge aren't a controlled edit of the
  current one. The model holds all else fixed and edits exactly the one piece —
  cleaner counterfactually, but only as trustworthy as the model. Agreement
  between the two is the reassuring case. Hover any cell for the three V's.</p>
</div>
</div>
<div id="tip"></div>
<script>
const tip = document.getElementById('tip');
document.querySelectorAll('td.cell').forEach(td => {{
  td.addEventListener('mousemove', e => {{
    const d = JSON.parse(td.dataset.tip);
    tip.innerHTML = `<b>p* = ${{d.pstar}}</b> &nbsp; <span class="k">eggs</span> ${{d.def_eggs}} `
      + `<span class="k">edge</span> ${{d.net_warriors > 0 ? '+' + d.net_warriors : d.net_warriors}}<br>`
      + `<span class="k">V(S)</span> ${{d.vS}} &nbsp; <span class="k">V<sub>kill</sub></span> ${{d.vK}} `
      + `&nbsp; <span class="k">V<sub>death</sub></span> ${{d.vD}}`;
    tip.style.opacity = 1;
    let x = e.clientX + 14, y = e.clientY + 14;
    const r = tip.getBoundingClientRect();
    if (x + r.width > innerWidth) x = e.clientX - r.width - 14;
    if (y + r.height > innerHeight) y = e.clientY - r.height - 14;
    tip.style.left = x + 'px'; tip.style.top = y + 'px';
  }});
  td.addEventListener('mouseleave', () => tip.style.opacity = 0);
}});
</script>
</body></html>'''


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='current_preferred_model.mdl')
    ap.add_argument('--data', default='quality_filtered/encoded/all_games.bin')
    ap.add_argument('--max-games', type=int, default=None,
                    help='limit games (default: all)')
    ap.add_argument('--batch-games', type=int, default=4000,
                    help='games materialized per streaming batch')
    ap.add_argument('--html', default='combat_value/break_even_empirical.html')
    ap.add_argument('--open', action='store_true')
    args = ap.parse_args()

    import lightgbm as lgb
    from event_codec import read_packed_games, materialize_entries
    print(f"Loading model {args.model}")
    model = lgb.Booster(model_file=args.model)

    # Streaming accumulators (never hold all states at once).
    wins, cnt = new_winrate_accum()
    model_matchups = {'warrior_queen': ('vanilla_warrior', 'queen'),
                      'warrior_warrior': ('vanilla_warrior', 'vanilla_warrior')}
    model_sums = {k: new_model_accum() for k in model_matchups}

    print(f"Streaming games from {args.data} "
          f"(batch {args.batch_games}, limit {args.max_games or 'all'})")
    batch, n_games, n_states = [], 0, 0

    def flush(entries):
        nonlocal n_states
        if not entries:
            return
        X, y, gids, ts = materialize_entries(entries)
        n_states += len(X)
        accumulate_winrate(X, y, wins, cnt)
        for key, (atk, dfn) in model_matchups.items():
            accumulate_model(X, model.predict, atk, dfn, gids, ts, model_sums[key])

    for game_id, encoded in read_packed_games(args.data):
        if args.max_games is not None and n_games >= args.max_games:
            break
        batch.append((game_id, encoded))
        n_games += 1
        if len(batch) >= args.batch_games:
            flush(batch)
            batch = []
            print(f"\r  {n_games:,} games · {n_states:,} states", end='', flush=True)
    flush(batch)
    print(f"\r  {n_games:,} games · {n_states:,} states")

    winrate = finalize_winrate(wins, cnt)
    emp_grids = {
        'warrior_queen': break_even_grid(winrate, 'warrior_queen'),
        'warrior_warrior': break_even_grid(winrate, 'warrior_warrior'),
    }
    print_table(emp_grids['warrior_queen'], 'EMPIRICAL  warrior → queen  (p*)')
    print_table(emp_grids['warrior_warrior'], 'EMPIRICAL  warrior → warrior  (p*)')

    mdl_grids = {k: finalize_model_grid(model_sums[k]) for k in model_matchups}

    # Agreement on shared, stable buckets (warrior->queen).
    for key in ('warrior_queen', 'warrior_warrior'):
        e, m = [], []
        for k, ec in emp_grids[key].items():
            mc = mdl_grids[key].get(k)
            if mc and np.isfinite(ec['pstar']) and np.isfinite(mc['pstar']) \
                    and abs(ec['pstar']) <= 1.5 and abs(mc['pstar']) <= 1.5:
                e.append(ec['pstar'])
                m.append(mc['pstar'])
        if len(e) >= 3:
            e, m = np.array(e), np.array(m)
            r = np.corrcoef(e, m)[0, 1]
            print(f"{key}: {len(e)} stable shared buckets · "
                  f"median |emp−model| p* = {np.median(np.abs(e - m)):.3f} · "
                  f"corr = {r:.2f}")

    meta = (f"Data: {args.data} · {n_games:,} games, {n_states:,} states "
            f"(both attacking sides) · empirical min {EMP_MIN}/bucket")
    page = build_html(emp_grids, mdl_grids, meta)
    out = os.path.abspath(args.html)
    with open(out, 'w') as f:
        f.write(page)
    print(f"\nWrote {out}")

    if args.open:
        for cmd in ('google-chrome', 'google-chrome-stable', 'chromium', 'chromium-browser'):
            try:
                subprocess.Popen([cmd, out], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"Opening in {cmd}")
                return
            except FileNotFoundError:
                continue
        print("Chrome not found; open the file manually.")


if __name__ == '__main__':
    main()
