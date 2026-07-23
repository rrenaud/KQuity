#!/usr/bin/env python3
"""Generate a clean HTML dashboard of coarse break-even p* heatmaps.

For a curated set of matchups, renders a def_eggs x net_warriors heatmap of the
break-even strike-success probability p*. Color is a blue<->red diverging scale
centered at p*=0.5 (the coin-flip pivot): blue = worth it at long odds (low p*),
red = only worth it when heavily favored (high p*). Every cell also prints the
number, so color is secondary encoding.

Usage:
  python -m combat_value.html_report [--data ...] [--model ...]
                                     [--max-games N] [--out path] [--open]
"""
from __future__ import annotations

import argparse
import html
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import lightgbm as lgb

from combat_value import core, analysis

# Curated matchups: (attacker_piece, defender_piece, short caption)
MATCHUPS = [
    ('vanilla_warrior', 'queen',           'Vanilla warrior assassinating the queen'),
    ('speed_warrior',   'queen',           'Speed warrior assassinating the queen'),
    ('vanilla_warrior', 'vanilla_warrior', 'Vanilla vs vanilla duel'),
    ('speed_warrior',   'speed_warrior',   'Speed vs speed duel'),
    ('speed_warrior',   'vanilla_warrior', 'Speed warrior hunting a vanilla'),
    ('vanilla_warrior', 'speed_warrior',   'Vanilla warrior into a speed warrior'),
    ('vanilla_warrior', 'speed_drone',     'Vanilla warrior hunting a speed drone'),
    ('speed_warrior',   'speed_drone',     'Speed warrior hunting a speed drone'),
]

# Fixed grid so every matchup aligns visually.
EGG_ROWS = [2, 1, 0]                     # top = full lives, bottom = last life
NET_COLS = [-3, -2, -1, 0, 1, 2, 3, 4]   # attacker warriors minus defender's
MIN_N = 120                              # hide buckets thinner than this

PIECE_LABEL = {
    'vanilla_warrior': 'vanilla warrior',
    'speed_warrior': 'speed warrior',
    'drone': 'drone',
    'speed_drone': 'speed drone',
    'queen': 'queen',
}

# Full attacker x defender matchup matrix over the four "combatant" pieces, in
# axis order. Speed drones don't strike directly, but a hunted speed drone can
# bump/bait an opponent into a teammate's kill, so drone->X is still a real
# value swing. Superset of MATCHUPS; used for the vdecomp 4x4 grid and to fill
# the shared bucket cache.
MATRIX_PIECES = ['queen', 'speed_warrior', 'vanilla_warrior', 'speed_drone']


def _matrix_caption(atk: str, dfn: str) -> str:
    a, d = PIECE_LABEL[atk], PIECE_LABEL[dfn]
    cap = f'{a} mirror' if atk == dfn else f'{a} vs {d}'
    return cap[0].upper() + cap[1:]


MATRIX_MATCHUPS = [(a, d, _matrix_caption(a, d))
                   for a in MATRIX_PIECES for d in MATRIX_PIECES]

# Diverging blue<->red anchors (dataviz reference palette), gray midpoint.
_ANCHORS = [
    (0.00, (16, 66, 129)),    # #104281 deep blue
    (0.25, (42, 120, 214)),   # #2a78d6
    (0.50, (240, 239, 236)),  # #f0efec neutral gray
    (0.75, (227, 73, 72)),    # #e34948
    (1.00, (150, 28, 27)),    # #961c1b deep red
]


def diverging_color(p: float) -> tuple[int, int, int]:
    """Map p* in [0,1] to an RGB triple; clamp outside, blue<->gray<->red."""
    t = max(0.0, min(1.0, p))
    for (t0, c0), (t1, c1) in zip(_ANCHORS, _ANCHORS[1:]):
        if t <= t1:
            f = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            return tuple(round(a + (b - a) * f) for a, b in zip(c0, c1))
    return _ANCHORS[-1][1]


def text_on(rgb: tuple[int, int, int]) -> str:
    """Pick readable ink for a cell background via relative luminance."""
    r, g, b = (c / 255 for c in rgb)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return '#0b0b0b' if lum > 0.55 else '#ffffff'


def compute_matchup(X, predict, atk, dfn, gids, ts):
    res = core.evaluate_matchup_both_sides(X, predict, atk, dfn,
                                           game_ids=gids, timestamps=ts)
    summ = analysis.summarize(res)
    rows = analysis.bucket_table(res, ['def_eggs', 'net_warriors'])
    grid = {(r['def_eggs'], r['net_warriors']): r for r in rows}
    return summ, grid


def render_grid(grid: dict) -> str:
    """Render one def_eggs x net_warriors heatmap table."""
    out = ['<table class="grid"><thead><tr>',
           '<th class="corner"><span class="ax-y">queen<br>lives</span>'
           '<span class="ax-x">warrior edge →</span></th>']
    for nc in NET_COLS:
        lab = f'+{nc}' if nc > 0 else str(nc)
        out.append(f'<th>{lab}</th>')
    out.append('</tr></thead><tbody>')
    for eg in EGG_ROWS:
        out.append(f'<tr><th class="rowh">{eg}</th>')
        for nc in NET_COLS:
            cell = grid.get((eg, nc))
            if cell is None or cell['n'] < MIN_N or not np.isfinite(cell['pstar']):
                out.append('<td class="empty"></td>')
                continue
            p = cell['pstar']
            rgb = diverging_color(p)
            bg = f'rgb({rgb[0]},{rgb[1]},{rgb[2]})'
            fg = text_on(rgb)
            tip = html.escape(json.dumps({
                'def_eggs': eg, 'net_warriors': nc, 'n': cell['n'],
                'pstar': round(p, 3), 'median': round(cell['pstar_median'], 3),
                'iqr': [round(cell['pstar_p25'], 2), round(cell['pstar_p75'], 2)],
                'vS': round(cell['mean_V_status_quo'], 3),
                'vK': round(cell['mean_V_kill'], 3),
                'vD': round(cell['mean_V_death'], 3),
            }))
            out.append(
                f'<td class="cell" style="background:{bg};color:{fg}" '
                f'data-tip="{tip}">'
                f'<span class="p">{p:.2f}</span>'
                f'<span class="n">{cell["n"]:,}</span></td>')
        out.append('</tr>')
    out.append('</tbody></table>')
    return ''.join(out)


def render_card(atk, dfn, caption, summ, grid) -> str:
    med = summ.get('pstar_median')
    med_s = f'{med:.2f}' if med is not None else '—'
    n = summ.get('n', 0)
    title = f'{PIECE_LABEL[atk]} <span class="vs">vs</span> {PIECE_LABEL[dfn]}'
    return f'''<section class="card">
  <div class="card-head">
    <h2>{title}</h2>
    <div class="cap">{html.escape(caption)}</div>
    <div class="hl"><span class="hl-n">{med_s}</span>
      <span class="hl-l">median p*<br><span class="mut">{n:,} states</span></span></div>
  </div>
  {render_grid(grid)}
</section>'''


PAGE = '''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Break-even combat value — KQuity</title>
<style>
:root {{
  color-scheme: light dark;
  --plane:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e;
  --mut:#898781; --line:#e1e0d9; --ring:rgba(11,11,11,.10);
}}
@media (prefers-color-scheme: dark) {{
  :root {{ --plane:#0d0d0d; --surface:#1a1a19; --ink:#fff; --ink2:#c3c2b7;
    --mut:#898781; --line:#2c2c2a; --ring:rgba(255,255,255,.10); }}
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--plane); color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif; line-height:1.5; }}
.wrap {{ max-width:1200px; margin:0 auto; padding:40px 24px 72px; }}
header.top {{ margin-bottom:8px; }}
h1 {{ font-size:26px; margin:0 0 6px; letter-spacing:-.01em; }}
.sub {{ color:var(--ink2); max-width:64ch; margin:0 0 4px; }}
.meta {{ color:var(--mut); font-size:13px; }}
.formula {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  background:var(--surface); border:1px solid var(--ring); border-radius:8px;
  padding:10px 14px; margin:16px 0 8px; display:inline-block; font-size:13px;
  color:var(--ink2); }}
.legend {{ display:flex; align-items:center; gap:14px; margin:18px 0 28px;
  flex-wrap:wrap; }}
.bar {{ height:14px; width:280px; border-radius:7px; border:1px solid var(--ring);
  background:linear-gradient(90deg,
    rgb(16,66,129) 0%, rgb(42,120,214) 25%, rgb(240,239,236) 50%,
    rgb(227,73,72) 75%, rgb(150,28,27) 100%); }}
.legend .lab {{ font-size:12px; color:var(--ink2); }}
.legend .ticks {{ display:flex; justify-content:space-between; width:280px;
  font-size:11px; color:var(--mut); font-variant-numeric:tabular-nums; }}
.grid-wrap {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(360px,1fr));
  gap:22px; }}
.card {{ background:var(--surface); border:1px solid var(--ring); border-radius:12px;
  padding:18px 18px 20px; }}
.card-head {{ display:grid; grid-template-columns:1fr auto; row-gap:2px;
  align-items:start; margin-bottom:14px; }}
.card h2 {{ font-size:16px; margin:0; grid-column:1; }}
.card h2 .vs {{ color:var(--mut); font-weight:400; font-size:13px; }}
.cap {{ grid-column:1; color:var(--ink2); font-size:12.5px; }}
.hl {{ grid-column:2; grid-row:1 / span 2; display:flex; align-items:baseline;
  gap:7px; justify-self:end; }}
.hl-n {{ font-size:30px; font-weight:650; letter-spacing:-.02em;
  font-variant-numeric:tabular-nums; }}
.hl-l {{ font-size:11px; color:var(--mut); text-align:left; line-height:1.25; }}
.mut {{ color:var(--mut); }}
table.grid {{ border-collapse:separate; border-spacing:3px; width:100%; }}
table.grid th {{ font-weight:500; color:var(--mut); font-size:12px;
  font-variant-numeric:tabular-nums; }}
table.grid thead th {{ padding-bottom:2px; }}
th.rowh {{ text-align:right; padding-right:6px; color:var(--ink2);
  font-variant-numeric:tabular-nums; }}
th.corner {{ position:relative; }}
.ax-y {{ display:block; font-size:10.5px; line-height:1.05; color:var(--mut);
  text-align:right; padding-right:6px; }}
.ax-x {{ display:block; font-size:10.5px; color:var(--mut); text-align:right;
  padding-right:6px; margin-top:2px; }}
td.cell {{ width:11.5%; height:44px; border-radius:5px; text-align:center;
  vertical-align:middle; cursor:default; transition:outline .08s;
  font-variant-numeric:tabular-nums; }}
td.cell:hover {{ outline:2px solid var(--ink); outline-offset:-2px; }}
td.cell .p {{ display:block; font-size:15px; font-weight:600; line-height:1.05; }}
td.cell .n {{ display:block; font-size:9.5px; opacity:.75; }}
td.empty {{ height:44px; border-radius:5px;
  background:repeating-linear-gradient(45deg,transparent,transparent 4px,
    var(--line) 4px,var(--line) 5px); opacity:.5; }}
.foot {{ margin-top:36px; color:var(--ink2); font-size:13px; max-width:78ch; }}
.foot b {{ color:var(--ink); font-weight:600; }}
.foot ul {{ padding-left:18px; }}
#tip {{ position:fixed; pointer-events:none; z-index:10; background:var(--ink);
  color:var(--plane); padding:9px 11px; border-radius:8px; font-size:12px;
  line-height:1.5; opacity:0; transition:opacity .08s; max-width:250px;
  box-shadow:0 6px 24px rgba(0,0,0,.28); }}
#tip .tt {{ font-variant-numeric:tabular-nums; }}
#tip .k {{ color:var(--mut); }}
</style>
</head>
<body>
<div class="wrap">
<header class="top">
  <h1>Break-even combat value</h1>
  <p class="sub">The minimum strike-success probability <b>p*</b> at which a
  combat pays off versus declining it. Fight iff your true survival odds exceed
  the cell value. Rows are the defender's remaining queen lives; columns are the
  attacker's warrior-count edge.</p>
  <div class="formula">p* = ( V(S) &minus; V<sub>death</sub> ) / ( V<sub>kill</sub> &minus; V<sub>death</sub> )</div>
  <div class="meta">{meta}</div>
</header>

<div class="legend">
  <div>
    <div class="bar"></div>
    <div class="ticks"><span>0.0</span><span>0.25</span><span>0.5</span><span>0.75</span><span>1.0</span></div>
  </div>
  <div class="lab"><b>blue</b> = worth it at long odds &nbsp;·&nbsp;
    <b>gray</b> = even-odds pivot &nbsp;·&nbsp;
    <b>red</b> = only when heavily favored</div>
</div>

<div class="grid-wrap">
{cards}
</div>

<div class="foot">
  <p><b>How to read it.</b> Each cell is the break-even survival probability for
  that situation, computed from the win-probability model: it values the state
  after the defender dies (V<sub>kill</sub>), after the attacker dies
  (V<sub>death</sub>), and the status quo (V(S)), all from the attacker's
  perspective. A queen kill on the last life (queen&nbsp;lives&nbsp;=&nbsp;0) wins
  the game, so V<sub>kill</sub>=1 and p* collapses. Cell number = p* from
  bucket-mean values; hover for the per-state median, IQR, and the three V's.
  Hatched cells have too few states ({min_n} min).</p>
  <ul>
    <li>p* is a threshold on the <b>conditional</b> survival probability (given
      the engagement resolves in a death), not "P(strike lands)".</li>
    <li>The state has no positional information beyond the snail; p* values
      composition, eggs, snail, and berries — nothing spatial.</li>
    <li>Baseline is the status quo, which ignores opportunity cost and so tends
      to understate p*.</li>
  </ul>
</div>
</div>

<div id="tip"></div>
<script>
const tip = document.getElementById('tip');
function fmt(d) {{
  const iqr = `[${{d.iqr[0]}}, ${{d.iqr[1]}}]`;
  return `<div class="tt"><b>p* = ${{d.pstar}}</b> &nbsp; <span class="k">from bucket means</span><br>`
    + `<span class="k">median</span> ${{d.median}} &nbsp; <span class="k">IQR</span> ${{iqr}}<br>`
    + `<span class="k">states</span> ${{d.n.toLocaleString()}}<br>`
    + `<span class="k">V(S)</span> ${{d.vS}} &nbsp; <span class="k">V<sub>kill</sub></span> ${{d.vK}} &nbsp; <span class="k">V<sub>death</sub></span> ${{d.vD}}</div>`;
}}
document.querySelectorAll('td.cell').forEach(td => {{
  td.addEventListener('mousemove', e => {{
    const d = JSON.parse(td.dataset.tip);
    tip.innerHTML = fmt(d);
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
</body>
</html>'''


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='current_preferred_model.mdl')
    ap.add_argument('--data', default='quality_filtered/encoded/all_games.bin')
    ap.add_argument('--max-games', type=int, default=3000)
    ap.add_argument('--out', default='combat_value/break_even.html')
    ap.add_argument('--from-cache', default=None,
                    help='build from a stream.py bucket cache (full-scale data)')
    ap.add_argument('--open', action='store_true', help='open in Chrome when done')
    args = ap.parse_args()

    if args.from_cache:
        from combat_value import stream
        cache = stream.load_cache(args.from_cache)
        def get(atk, dfn):
            return stream.cache_summary(cache, atk, dfn), stream.cache_grid(cache, atk, dfn)
        n_games, n_states = cache['meta']['n_games'], cache['meta']['n_states']
        data_src = cache['meta']['data']
    else:
        print(f"Loading model {args.model}")
        model = lgb.Booster(model_file=args.model)
        print(f"Materializing up to {args.max_games} games from {args.data}")
        from event_codec import fast_materialize_from_codec
        X, y, gids, ts = fast_materialize_from_codec(args.data, max_games=args.max_games)
        print(f"  {len(X):,} states")
        def get(atk, dfn):
            return compute_matchup(X, model.predict, atk, dfn, gids, ts)
        n_games, n_states, data_src = args.max_games, len(X), args.data

    cards = []
    for atk, dfn, cap in MATCHUPS:
        summ, grid = get(atk, dfn)
        med = summ.get('pstar_median')
        print(f"  {atk:16s} vs {dfn:16s}  median p*="
              f"{('%.3f' % med) if med is not None else 'n/a':>6}  n={summ.get('n',0):,}")
        cards.append(render_card(atk, dfn, cap, summ, grid))

    meta = (f"Model: {os.path.basename(os.path.realpath(args.model))} &nbsp;·&nbsp; "
            f"Data: {data_src} &nbsp;·&nbsp; {n_games:,} games, "
            f"{n_states:,} states (both attacking sides)")
    page = PAGE.format(meta=meta, cards='\n'.join(cards), min_n=MIN_N)

    out = os.path.abspath(args.out)
    with open(out, 'w') as f:
        f.write(page)
    print(f"\nWrote {out}")

    if args.open:
        for cmd in ('google-chrome', 'google-chrome-stable', 'chromium',
                    'chromium-browser'):
            try:
                subprocess.Popen([cmd, out],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"Opening in {cmd}")
                return
            except FileNotFoundError:
                continue
        print("Chrome not found; open the file manually.")


if __name__ == '__main__':
    main()
