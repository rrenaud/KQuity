#!/usr/bin/env python3
"""Small-multiple line charts of the break-even tradeoff curves.

Two faceted views of p* vs the attacker's warrior edge:
  1. By matchup   — 8 panels, one line per defender queen-lives level.
  2. By queen lives — 3 panels, one line per matchup.
Each line is the per-state median p* with a translucent 25-75th percentile
(IQR) band. A dashed reference marks the p*=0.5 even-odds pivot.

Usage:
  python -m combat_value.line_report [--data ...] [--model ...]
                                     [--max-games N] [--out path] [--open]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import lightgbm as lgb

from combat_value import core, analysis
from combat_value.html_report import MATCHUPS, PIECE_LABEL, MIN_N

EDGES = [-3, -2, -1, 0, 1, 2, 3, 4]
LIVES = [2, 1, 0]

# Sequential blue ramp for queen lives (light = full lives, dark = last life).
LIVES_COLOR = {2: '#86b6ef', 1: '#3987e5', 0: '#104281'}
LIVES_NAME = {2: '2 (full)', 1: '1', 0: '0 (last life)'}

# Categorical slots (dataviz reference order) for the by-lives view.
V2_MATCHUPS = [
    ('vanilla_warrior', 'queen',           'vanilla → queen', '#2a78d6'),
    ('speed_warrior',   'queen',           'speed → queen',   '#008300'),
    ('vanilla_warrior', 'vanilla_warrior', 'vanilla mirror',  '#e87ba4'),
    ('speed_warrior',   'vanilla_warrior', 'speed → vanilla', '#eda100'),
    ('vanilla_warrior', 'speed_warrior',   'vanilla → speed', '#1baf7a'),
]


def point(cell) -> dict | None:
    if cell is None or cell['n'] < MIN_N:
        return None
    return {
        'x': int(cell['net_warriors']),
        'med': round(float(cell['pstar_median']), 4),
        'p25': round(float(cell['pstar_p25']), 4),
        'p75': round(float(cell['pstar_p75']), 4),
        'mean': (None if not np.isfinite(cell['pstar']) else round(float(cell['pstar']), 4)),
        'n': int(cell['n']),
        'vS': round(float(cell['mean_V_status_quo']), 3),
        'vK': round(float(cell['mean_V_kill']), 3),
        'vD': round(float(cell['mean_V_death']), 3),
    }


def compute_grids(model, X, gids, ts):
    """{(atk, dfn): {(def_eggs, net_warriors): row}} from in-memory states."""
    grids = {}
    for atk, dfn, _cap in MATCHUPS:
        res = core.evaluate_matchup_both_sides(X, model.predict, atk, dfn,
                                               game_ids=gids, timestamps=ts)
        rows = analysis.bucket_table(res, ['def_eggs', 'net_warriors'])
        grids[(atk, dfn)] = {(r['def_eggs'], r['net_warriors']): r for r in rows}
        med = float(np.median(res.pstar[analysis.valid_mask(res)]))
        print(f"  {atk:16s} vs {dfn:16s} median p*={med:.3f}")
    return grids


def build(grids):
    # View 1: panel per matchup, series per queen-lives level.
    view1 = []
    for atk, dfn, cap in MATCHUPS:
        g = grids[(atk, dfn)]
        series = []
        for lv in LIVES:
            pts = [p for e in EDGES if (p := point(g.get((lv, e)))) is not None]
            if pts:
                series.append({'name': LIVES_NAME[lv], 'color': LIVES_COLOR[lv],
                               'pts': pts})
        view1.append({
            'title': f'{PIECE_LABEL[atk]} → {PIECE_LABEL[dfn]}',
            'subtitle': cap, 'series': series})

    # View 2: panel per queen-lives level, series per matchup.
    view2 = []
    for lv in LIVES:
        series = []
        for atk, dfn, name, color in V2_MATCHUPS:
            g = grids[(atk, dfn)]
            pts = [p for e in EDGES if (p := point(g.get((lv, e)))) is not None]
            if pts:
                series.append({'name': name, 'color': color, 'pts': pts})
        view2.append({'title': f'Defender queen lives = {lv}',
                      'subtitle': LIVES_NAME[lv], 'series': series})

    legends = {
        'lives': [{'name': LIVES_NAME[lv], 'color': LIVES_COLOR[lv]} for lv in LIVES],
        'matchups': [{'name': n, 'color': c} for _a, _d, n, c in V2_MATCHUPS],
    }
    return {'view1': view1, 'view2': view2, 'legends': legends,
            'edges': EDGES}


PAGE = '''<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Break-even tradeoff curves — KQuity</title>
<style>
:root {{ color-scheme:light dark;
  --plane:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --mut:#898781;
  --line:#e1e0d9; --axis:#c3c2b7; --ring:rgba(11,11,11,.10);
  --fav:rgba(42,120,214,.055); --unf:rgba(227,73,72,.06); }}
@media (prefers-color-scheme: dark) {{ :root {{
  --plane:#0d0d0d; --surface:#1a1a19; --ink:#fff; --ink2:#c3c2b7; --mut:#898781;
  --line:#2c2c2a; --axis:#383835; --ring:rgba(255,255,255,.10);
  --fav:rgba(57,135,229,.10); --unf:rgba(230,103,103,.10); }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--plane); color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif; line-height:1.5; }}
.wrap {{ max-width:1200px; margin:0 auto; padding:40px 24px 72px; }}
h1 {{ font-size:26px; margin:0 0 6px; letter-spacing:-.01em; }}
.sub {{ color:var(--ink2); max-width:70ch; margin:0 0 4px; }}
.meta {{ color:var(--mut); font-size:13px; margin-top:10px; }}
.sec-head {{ display:flex; align-items:baseline; gap:16px; flex-wrap:wrap;
  margin:34px 0 6px; border-top:1px solid var(--line); padding-top:22px; }}
.sec-head h2 {{ font-size:18px; margin:0; }}
.sec-head .d {{ color:var(--ink2); font-size:13px; }}
.legend {{ display:flex; gap:16px; flex-wrap:wrap; margin:10px 0 16px; }}
.legend .it {{ display:flex; align-items:center; gap:7px; font-size:12.5px;
  color:var(--ink2); }}
.legend .sw {{ width:22px; height:0; border-top:3px solid; border-radius:2px; }}
.grid1 {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(270px,1fr)); gap:20px; }}
.grid2 {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:20px; }}
.panel {{ background:var(--surface); border:1px solid var(--ring); border-radius:12px;
  padding:14px 14px 8px; }}
.panel h3 {{ font-size:14px; margin:0; }}
.panel .cap {{ color:var(--mut); font-size:11.5px; margin:1px 0 6px; }}
.panel svg {{ width:100%; height:auto; display:block; overflow:visible; }}
.ax {{ fill:var(--mut); font-size:10px; }}
.axtitle {{ fill:var(--mut); font-size:10px; }}
.foot {{ margin-top:36px; color:var(--ink2); font-size:13px; max-width:80ch; }}
.foot b {{ color:var(--ink); font-weight:600; }}
#tip {{ position:fixed; pointer-events:none; z-index:10; background:var(--ink);
  color:var(--plane); padding:9px 11px; border-radius:8px; font-size:12px;
  line-height:1.5; opacity:0; transition:opacity .08s; box-shadow:0 6px 24px rgba(0,0,0,.28);
  font-variant-numeric:tabular-nums; }}
#tip .k {{ color:var(--mut); }}
</style></head><body>
<div class="wrap">
<h1>Break-even tradeoff curves</h1>
<p class="sub">The minimum survival probability <b>p*</b> at which taking a combat
beats declining it, plotted against the attacker's <b>warrior edge</b> (own winged
count minus the defender's). Lines are per-state medians; shaded bands are the
25–75th percentile. Fight when your true odds sit above the curve; the dashed line
is the even-odds pivot.</p>
<div class="meta">{meta}</div>

<div class="sec-head"><h2>By matchup</h2>
  <span class="d">one panel per matchup · line color = defender queen lives</span></div>
<div class="legend" id="leg-lives"></div>
<div class="grid1" id="view1"></div>

<div class="sec-head"><h2>By queen lives</h2>
  <span class="d">one panel per defender queen-lives level · line color = matchup</span></div>
<div class="legend" id="leg-match"></div>
<div class="grid2" id="view2"></div>

<div class="foot">
  <p><b>Reading the slope.</b> A steep line means your warrior advantage changes the
  break-even fast; a flat line means the trade is priced the same regardless of edge.
  The vertical gap between the queen-lives curves is the assassination leverage — the
  <b>last-life</b> curve sits far lower because that kill wins the game
  (V<sub>kill</sub>=1). Points above p*=1 (e.g. hunting drones) mean "never worth it".
  Hover any point for its median, IQR, state count, and the three V's.</p>
</div>
</div>
<div id="tip"></div>
<script>
const DATA = {data};
const tip = document.getElementById('tip');
const W = 300, H = 196, M = {{t: 10, r: 12, b: 30, l: 34}};
const PW = W - M.l - M.r, PH = H - M.t - M.b;
const XS = [-3, 4], YS = [0, 1.05];
const sx = x => M.l + (x - XS[0]) / (XS[1] - XS[0]) * PW;
const sy = y => M.t + (1 - (Math.min(y, YS[1]) - YS[0]) / (YS[1] - YS[0])) * PH;

function cssvar(n) {{ return getComputedStyle(document.body).getPropertyValue(n); }}

function panelSVG(panel) {{
  const svgns = 'http://www.w3.org/2000/svg';
  const s = document.createElementNS(svgns, 'svg');
  s.setAttribute('viewBox', `0 0 ${{W}} ${{H}}`);
  const mk = (t, a, txt) => {{
    const e = document.createElementNS(svgns, t);
    for (const k in a) e.setAttribute(k, a[k]);
    if (txt != null) e.textContent = txt;
    return e;
  }};
  // favorable / unfavorable washes around p*=0.5
  s.appendChild(mk('rect', {{x: M.l, y: sy(1.05), width: PW, height: sy(0.5) - sy(1.05),
    fill: 'var(--unf)'}}));
  s.appendChild(mk('rect', {{x: M.l, y: sy(0.5), width: PW, height: sy(0) - sy(0.5),
    fill: 'var(--fav)'}}));
  // y gridlines + labels
  [0, 0.25, 0.5, 0.75, 1.0].forEach(v => {{
    const y = sy(v);
    s.appendChild(mk('line', {{x1: M.l, y1: y, x2: M.l + PW, y2: y,
      stroke: v === 0.5 ? 'var(--axis)' : 'var(--line)',
      'stroke-dasharray': v === 0.5 ? '4 3' : '0', 'stroke-width': 1}}));
    s.appendChild(mk('text', {{x: M.l - 5, y: y + 3, 'text-anchor': 'end',
      class: 'ax'}}, v.toFixed(2)));
  }});
  // x ticks
  for (let e = -3; e <= 4; e++) {{
    s.appendChild(mk('text', {{x: sx(e), y: H - M.b + 13, 'text-anchor': 'middle',
      class: 'ax'}}, e > 0 ? '+' + e : '' + e));
  }}
  s.appendChild(mk('text', {{x: M.l + PW / 2, y: H - 3, 'text-anchor': 'middle',
    class: 'axtitle'}}, 'warrior edge'));
  s.appendChild(mk('text', {{x: -(M.t + PH / 2), y: 10, 'text-anchor': 'middle',
    class: 'axtitle', transform: 'rotate(-90)'}}, 'break-even p*'));

  // series: band then line then markers
  panel.series.forEach(se => {{
    if (se.pts.length > 1) {{
      let d = 'M';
      se.pts.forEach(p => {{ d += ` ${{sx(p.x)}} ${{sy(p.p75)}}`; }});
      for (let i = se.pts.length - 1; i >= 0; i--) {{
        const p = se.pts[i]; d += ` L ${{sx(p.x)}} ${{sy(p.p25)}}`;
      }}
      d += ' Z';
      s.appendChild(mk('path', {{d, fill: se.color, 'fill-opacity': 0.13, stroke: 'none'}}));
    }}
    const line = se.pts.map(p => `${{sx(p.x)}},${{sy(p.med)}}`).join(' ');
    s.appendChild(mk('polyline', {{points: line, fill: 'none', stroke: se.color,
      'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round'}}));
    se.pts.forEach(p => {{
      s.appendChild(mk('circle', {{cx: sx(p.x), cy: sy(p.med), r: 2.6,
        fill: se.color, stroke: 'var(--surface)', 'stroke-width': 1}}));
      const hit = mk('circle', {{cx: sx(p.x), cy: sy(p.med), r: 9,
        fill: 'transparent', style: 'cursor:default'}});
      hit.addEventListener('mousemove', ev => showTip(ev, panel, se, p));
      hit.addEventListener('mouseleave', () => tip.style.opacity = 0);
      s.appendChild(hit);
    }});
  }});
  return s;
}}

function showTip(ev, panel, se, p) {{
  const mean = p.mean == null ? '—' : p.mean.toFixed(3);
  tip.innerHTML =
    `<b>${{se.name}}</b> &nbsp; <span class="k">edge</span> ${{p.x > 0 ? '+' + p.x : p.x}}<br>`
    + `<b>p* median ${{p.med.toFixed(3)}}</b> &nbsp; <span class="k">IQR</span> [${{p.p25.toFixed(2)}}, ${{p.p75.toFixed(2)}}]<br>`
    + `<span class="k">p* from means</span> ${{mean}} &nbsp; <span class="k">n</span> ${{p.n.toLocaleString()}}<br>`
    + `<span class="k">V(S)</span> ${{p.vS}} &nbsp; <span class="k">V<sub>kill</sub></span> ${{p.vK}} &nbsp; <span class="k">V<sub>death</sub></span> ${{p.vD}}`;
  tip.style.opacity = 1;
  let x = ev.clientX + 14, y = ev.clientY + 14;
  const r = tip.getBoundingClientRect();
  if (x + r.width > innerWidth) x = ev.clientX - r.width - 14;
  if (y + r.height > innerHeight) y = ev.clientY - r.height - 14;
  tip.style.left = x + 'px'; tip.style.top = y + 'px';
}}

function renderLegend(id, items) {{
  const el = document.getElementById(id);
  items.forEach(it => {{
    const d = document.createElement('div'); d.className = 'it';
    d.innerHTML = `<span class="sw" style="border-color:${{it.color}}"></span>${{it.name}}`;
    el.appendChild(d);
  }});
}}

function renderView(id, panels) {{
  const root = document.getElementById(id);
  panels.forEach(panel => {{
    const card = document.createElement('div'); card.className = 'panel';
    const h = document.createElement('h3'); h.textContent = panel.title;
    const c = document.createElement('div'); c.className = 'cap'; c.textContent = panel.subtitle;
    card.appendChild(h); card.appendChild(c); card.appendChild(panelSVG(panel));
    root.appendChild(card);
  }});
}}

renderLegend('leg-lives', DATA.legends.lives);
renderLegend('leg-match', DATA.legends.matchups);
renderView('view1', DATA.view1);
renderView('view2', DATA.view2);
</script>
</body></html>'''


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='current_preferred_model.mdl')
    ap.add_argument('--data', default='quality_filtered/encoded/all_games.bin')
    ap.add_argument('--max-games', type=int, default=3000)
    ap.add_argument('--out', default='combat_value/break_even_lines.html')
    ap.add_argument('--from-cache', default=None,
                    help='build from a stream.py bucket cache (full-scale data)')
    ap.add_argument('--open', action='store_true')
    args = ap.parse_args()

    if args.from_cache:
        from combat_value import stream
        cache = stream.load_cache(args.from_cache)
        grids = {(a, d): stream.cache_grid(cache, a, d) for a, d, *_ in MATCHUPS}
        n_games, n_states, data_src = (cache['meta']['n_games'],
                                       cache['meta']['n_states'], cache['meta']['data'])
    else:
        print(f"Loading model {args.model}")
        model = lgb.Booster(model_file=args.model)
        print(f"Materializing up to {args.max_games} games from {args.data}")
        from event_codec import fast_materialize_from_codec
        X, y, gids, ts = fast_materialize_from_codec(args.data, max_games=args.max_games)
        print(f"  {len(X):,} states")
        grids = compute_grids(model, X, gids, ts)
        n_games, n_states, data_src = args.max_games, len(X), args.data

    data = build(grids)
    meta = (f"Model: {os.path.basename(os.path.realpath(args.model))} · "
            f"Data: {data_src} · {n_games:,} games, "
            f"{n_states:,} states (both attacking sides)")
    page = PAGE.format(meta=meta, data=json.dumps(data))

    out = os.path.abspath(args.out)
    with open(out, 'w') as f:
        f.write(page)
    print(f"\nWrote {out}")

    if args.open:
        for cmd in ('google-chrome', 'google-chrome-stable', 'chromium',
                    'chromium-browser'):
            try:
                subprocess.Popen([cmd, out], stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
                print(f"Opening in {cmd}")
                return
            except FileNotFoundError:
                continue
        print("Chrome not found; open the file manually.")


if __name__ == '__main__':
    main()
