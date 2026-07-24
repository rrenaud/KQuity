#!/usr/bin/env python3
"""Win-probability decomposition of the break-even, y-axis = P(attacker wins).

For each matchup, at a selectable defender queen-lives level, plots three model
outputs vs the attacker's warrior edge:
  V_kill   — win prob if the defender dies   (upper)
  V(S)     — win prob at the status quo       (middle)
  V_death  — win prob if the attacker dies    (lower)
The gap V(S)->V_kill (green) is the upside gained; V_death->V(S) (red) is the
downside risked. The break-even is the geometric reading
  p* = downside / (downside + upside) = (V(S)-V_death)/(V_kill-V_death).

Usage:
  python -m combat_value.vdecomp_report [--data ...] [--model ...]
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

from combat_value import core, analysis, kills
from combat_value.html_report import (
    MATRIX_MATCHUPS, MATRIX_PIECES, PIECE_LABEL, MIN_N)

EDGES = [-3, -2, -1, 0, 1, 2, 3, 4]
LIVES = [2, 1, 0]
LIVES_NAME = {2: '2 (full)', 1: '1', 0: '0 (last life)'}
KILL_MATRIX = 'combat_value/_kill_matrix.json'


def point(cell) -> dict | None:
    if cell is None or cell['n'] < MIN_N:
        return None
    p = cell['pstar']
    return {
        'x': int(cell['net_warriors']),
        'vS': round(float(cell['mean_V_status_quo']), 4),
        'vK': round(float(cell['mean_V_kill']), 4),
        'vD': round(float(cell['mean_V_death']), 4),
        'pstar': (None if not np.isfinite(p) else round(float(p), 4)),
        'med': round(float(cell['pstar_median']), 4),
        'p25': round(float(cell['pstar_p25']), 4),
        'p75': round(float(cell['pstar_p75']), 4),
        'n': int(cell['n']),
    }


def compute_grids(model, X, gids, ts):
    results = core.evaluate_matchups_shared(X, model.predict, MATRIX_MATCHUPS,
                                            game_ids=gids, timestamps=ts)
    grids, summaries = {}, {}
    for atk, dfn, _cap in MATRIX_MATCHUPS:
        res = results[(atk, dfn)]
        rows = analysis.bucket_table(res, ['def_eggs', 'net_warriors'])
        grids[(atk, dfn)] = {(r['def_eggs'], r['net_warriors']): r for r in rows}
        summaries[(atk, dfn)] = analysis.summarize(res)
        print(f"  {atk:16s} vs {dfn:16s}")
    return grids, summaries


def _global_pstar(summ) -> float | None:
    """Break-even p* from a matchup's globally-averaged V's."""
    if not summ or summ.get('n', 0) < 1:
        return None
    swing = summ['mean_V_kill'] - summ['mean_V_death']
    if abs(swing) < 1e-9:
        return None
    return (summ['mean_V_status_quo'] - summ['mean_V_death']) / swing


def build_summary_rows(summaries, kill_counts):
    """One row per matchup: global model p* vs empirical kill/death outcome."""
    rows = []
    for atk, dfn, _cap in MATRIX_MATCHUPS:
        summ = summaries.get((atk, dfn), {})
        pstar = _global_pstar(summ)
        emp = kills.matchup_stats(kill_counts, atk, dfn) if kill_counts else None
        wr = emp['win_rate'] if emp else None
        rows.append({
            'atk': atk, 'dfn': dfn,
            'pstar': pstar,
            'n_states': int(summ.get('n', 0)),
            'win_rate': wr,
            'kd': emp['kd'] if emp else None,
            'kills_for': emp['kills_for'] if emp else None,
            'kills_ag': emp['kills_ag'] if emp else None,
            'margin': (wr - pstar) if (wr is not None and pstar is not None) else None,
        })
    return rows


def _fmt_kd(kd, kills_for, kills_ag) -> str:
    if kills_ag == 0 and kills_for:
        return '∞'
    if kd is None:
        return '—'
    return f'{kd:,.0f}' if kd >= 100 else f'{kd:.2f}'


def _margin_class(m) -> str:
    if m is None:
        return 'mzero'
    return 'mpos' if m > 0.02 else ('mneg' if m < -0.02 else 'mzero')


def render_summary_table(rows) -> str:
    out = ['<table class="summary"><thead><tr>',
           '<th class="l">matchup (attacker → defender)</th>',
           '<th>model p*</th><th>empirical win%</th><th>margin</th>',
           '<th>K/D</th><th>kills (A→B / B→A)</th></tr></thead><tbody>']
    prev_atk = None
    for r in rows:
        sep = 'rowsep' if (prev_atk is not None and r['atk'] != prev_atk) else ''
        prev_atk = r['atk']
        label = (f'{PIECE_LABEL[r["atk"]]} <span class="arr">→</span> '
                 f'{PIECE_LABEL[r["dfn"]]}')
        ps = '—' if r['pstar'] is None else f'{r["pstar"]:.2f}'
        wr = '—' if r['win_rate'] is None else f'{r["win_rate"]:.2f}'
        m = r['margin']
        ms = '—' if m is None else f'{m:+.2f}'
        mcls = _margin_class(m)
        kd = _fmt_kd(r['kd'], r['kills_for'], r['kills_ag'])
        kills = ('—' if r['kills_for'] is None
                 else f'{r["kills_for"]:,} / {r["kills_ag"]:,}')
        out.append(
            f'<tr class="{sep}"><td class="l">{label}</td>'
            f'<td>{ps}</td><td>{wr}</td>'
            f'<td class="margin {mcls}">{ms}</td>'
            f'<td>{kd}</td><td class="k">{kills}</td></tr>')
    out.append('</tbody></table>')
    return ''.join(out)


def build(grids):
    matchups = []
    for atk, dfn, cap in MATRIX_MATCHUPS:
        grid = grids[(atk, dfn)]
        lives = {}
        for lv in LIVES:
            pts = [p for e in EDGES if (p := point(grid.get((lv, e)))) is not None]
            lives[str(lv)] = pts
        matchups.append({'atk': atk, 'dfn': dfn,
                         'title': f'{PIECE_LABEL[atk]} → {PIECE_LABEL[dfn]}',
                         'subtitle': cap, 'lives': lives})
        print(f"  {atk:16s} vs {dfn:16s}")
    used = {m['atk'] for m in matchups} | {m['dfn'] for m in matchups}
    pieces = [p for p in MATRIX_PIECES if p in used]
    return {'matchups': matchups, 'edges': EDGES, 'pieces': pieces,
            'piece_label': {p: PIECE_LABEL[p] for p in pieces},
            'lives_name': {str(k): v for k, v in LIVES_NAME.items()}}


PAGE = '''<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Break-even decomposition — KQuity</title>
<style>
:root {{ color-scheme:light dark;
  --plane:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --mut:#898781;
  --line:#e1e0d9; --axis:#c3c2b7; --ring:rgba(11,11,11,.10);
  --up:#0ca30c; --upline:#006300; --down:#d03b3b; --sq:#52514e; --sqline:#0b0b0b;
  --pstar:#7a5cc6; --pstarline:#5b3fa0; }}
@media (prefers-color-scheme: dark) {{ :root {{
  --plane:#0d0d0d; --surface:#1a1a19; --ink:#fff; --ink2:#c3c2b7; --mut:#898781;
  --line:#2c2c2a; --axis:#383835; --ring:rgba(255,255,255,.10);
  --up:#0ca30c; --upline:#0ca30c; --down:#e06767; --sq:#c3c2b7; --sqline:#ffffff;
  --pstar:#a98cff; --pstarline:#c4b0ff; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--plane); color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif; line-height:1.5; }}
.wrap {{ max-width:1200px; margin:0 auto; padding:40px 24px 72px; }}
h1 {{ font-size:26px; margin:0 0 6px; letter-spacing:-.01em; }}
h2.sec {{ font-size:19px; margin:44px 0 2px; letter-spacing:-.01em; }}
.sub {{ color:var(--ink2); max-width:72ch; margin:0 0 4px; }}
.meta {{ color:var(--mut); font-size:13px; margin-top:10px; }}
.formula {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  background:var(--surface); border:1px solid var(--ring); border-radius:8px;
  padding:11px 15px; margin:14px 0 4px; display:inline-block; font-size:13px;
  color:var(--ink2); line-height:1.7; }}
.formula b {{ color:var(--ink); font-weight:600; }}
.controls {{ display:flex; align-items:center; gap:16px; flex-wrap:wrap;
  margin:22px 0 6px; }}
.seg {{ display:inline-flex; border:1px solid var(--ring); border-radius:9px;
  overflow:hidden; }}
.seg button {{ appearance:none; border:0; background:var(--surface); color:var(--ink2);
  font:inherit; font-size:13px; padding:7px 15px; cursor:pointer; border-right:1px solid var(--ring); }}
.seg button:last-child {{ border-right:0; }}
.seg button[aria-pressed="true"] {{ background:var(--ink); color:var(--plane); }}
.seg-label {{ font-size:13px; color:var(--ink2); }}
.legend {{ display:flex; gap:16px; flex-wrap:wrap; margin:8px 0 18px; }}
.legend .it {{ display:flex; align-items:center; gap:7px; font-size:12.5px; color:var(--ink2); }}
.legend .sw {{ width:22px; height:0; border-top:3px solid; border-radius:2px; }}
.legend .bx {{ width:14px; height:12px; border-radius:3px; }}
.matrix-wrap {{ overflow-x:auto; padding-bottom:4px; }}
.matrix {{ display:grid; gap:14px; align-items:stretch; min-width:820px; }}
.mcorner {{ display:flex; align-items:flex-end; justify-content:flex-end;
  font-size:10.5px; color:var(--mut); text-align:right; padding-bottom:4px; line-height:1.25; }}
.mhead {{ font-size:13px; color:var(--ink); font-weight:650; text-align:center;
  align-self:end; padding-bottom:2px; }}
.mrow-head {{ font-size:13px; color:var(--ink); font-weight:650; padding-right:6px;
  display:flex; align-items:center; justify-content:flex-end; text-align:right; }}
.mempty {{ border:1px dashed var(--ring); border-radius:11px; opacity:.45; min-height:38px; }}
.panel {{ background:var(--surface); border:1px solid var(--ring); border-radius:12px;
  padding:12px 12px 8px; }}
.panel svg {{ width:100%; height:auto; display:block; overflow:visible; }}
.ax {{ fill:var(--mut); font-size:10px; }}
.axtitle {{ fill:var(--mut); font-size:10px; }}
.summary {{ border-collapse:collapse; width:100%; max-width:780px;
  font-size:13px; font-variant-numeric:tabular-nums; margin:6px 0 2px; }}
.summary th {{ text-align:right; color:var(--mut); font-weight:500; font-size:11.5px;
  padding:6px 12px; border-bottom:1px solid var(--ring); white-space:nowrap; }}
.summary th.l {{ text-align:left; }}
.summary td {{ text-align:right; padding:5px 12px; border-bottom:1px solid var(--line);
  white-space:nowrap; }}
.summary td.l {{ text-align:left; color:var(--ink); }}
.summary td.k {{ color:var(--mut); font-size:12px; }}
.summary td.margin {{ font-weight:600; }}
.summary .arr {{ color:var(--mut); }}
.summary tr.rowsep td {{ border-top:2px solid var(--ring); }}
.summary td.mpos {{ color:var(--up); }}
.summary td.mneg {{ color:var(--down); }}
.summary td.mzero {{ color:var(--mut); }}
.foot {{ margin-top:40px; color:var(--ink2); font-size:13px; max-width:82ch; }}
.foot b {{ color:var(--ink); font-weight:600; }}
.foot h3 {{ color:var(--ink); font-size:14px; margin:20px 0 4px; }}
.foot p {{ margin:6px 0; }}
.foot ul {{ padding-left:18px; margin:6px 0; }}
.foot li {{ margin:3px 0; }}
#tip {{ position:fixed; pointer-events:none; z-index:10; background:var(--ink);
  color:var(--plane); padding:9px 11px; border-radius:8px; font-size:12px;
  line-height:1.5; opacity:0; transition:opacity .08s; box-shadow:0 6px 24px rgba(0,0,0,.28);
  font-variant-numeric:tabular-nums; }}
#tip .k {{ color:var(--mut); }}
</style></head><body>
<div class="wrap">
<h1>Break-even, decomposed into win probabilities</h1>
<p class="sub"><b>When is it worth taking a shot at an enemy piece?</b> Going for
a kill in Killer Queen means committing to an exchange you might not win. To make
the question precise we adopt a <b>simplifying assumption</b>: the engagement
resolves as exactly one death &mdash; your target dies, or you do &mdash; with no
whiffs, mutual trades, or escapes. Under that model the shot is a gamble: land it
and the game swings your way, lose it and it swings away. This page reads that
gamble straight off the win-probability model, matchup by matchup and board state
by board state.</p>
<p class="sub">Each panel takes the attacker's view. We evaluate the model at
three states &mdash; the <b>status quo</b> <b>V(S)</b> (decline the fight), the
board <b>after the defender dies</b> <b style="color:var(--up)">V<sub>kill</sub></b>,
and the board <b>after the attacker dies</b> <b style="color:var(--down)">V<sub>death</sub></b>
&mdash; and plot all three as win&nbsp;probabilities (y) against the attacker's
<b>warrior edge</b> (x, their winged-unit count minus the defender's). The
<b style="color:var(--up)">green</b> band is the <b>upside</b> you gain by
winning; the <b style="color:var(--down)">red</b> band is the <b>downside</b> you
risk by losing.</p>
<p class="sub">Let <b>p</b> be your true probability of winning that exchange
(landing the kill rather than dying). Taking the shot beats declining it exactly
when its expected value clears the status quo, which pins the <b>break-even</b>
success rate <b>p*</b>: take the shot iff your odds beat&nbsp;p*.</p>
<div class="formula">
  fight&nbsp;iff&ensp; <b>p</b>&middot;V<sub>kill</sub> + (1&minus;<b>p</b>)&middot;V<sub>death</sub> &nbsp;&ge;&nbsp; V(S)
  &emsp;&hArr;&emsp; <b>p &ge; p*</b> =
  ( V(S) &minus; V<sub>death</sub> ) / ( V<sub>kill</sub> &minus; V<sub>death</sub> )
  &nbsp;=&nbsp; <span style="color:var(--down)">red</span> / (<span style="color:var(--down)">red</span> + <span style="color:var(--up)">green</span>)
</div>
<p class="sub">So p* is just where V(S) sits on the segment from V<sub>death</sub>
to V<sub>kill</sub>: a <b>low p*</b> means the fight pays off even at long odds
(mostly upside, little to lose); a <b>high p*</b> means only take it when you're
already favored. Panels are a matchup <b>matrix</b> &mdash; rows are the
attacking piece, columns the defending piece &mdash; so any matchup sits
transposed from its reverse across the diagonal. A speed drone can't strike
directly, but a hunted one can bait an opponent into a teammate's kill, so its
row is a real value swing too.</p>
<div class="meta">{meta}</div>

<div class="controls">
  <span class="seg-label">Defender queen lives:</span>
  <div class="seg" id="seg">
    <button data-lv="2">2 (full)</button>
    <button data-lv="1" aria-pressed="true">1</button>
    <button data-lv="0">0 (last life)</button>
  </div>
</div>
<div class="legend">
  <div class="it"><span class="sw" style="border-color:var(--up)"></span>V<sub>kill</sub> — defender dies</div>
  <div class="it"><span class="sw" style="border-color:var(--sqline)"></span>V(S) — status quo</div>
  <div class="it"><span class="sw" style="border-color:var(--down)"></span>V<sub>death</sub> — attacker dies</div>
  <div class="it"><span class="bx" style="background:var(--up);opacity:.16"></span>upside gained</div>
  <div class="it"><span class="bx" style="background:var(--down);opacity:.16"></span>downside risked</div>
</div>
<div class="matrix-wrap"><div class="matrix" id="grid"></div></div>

<h2 class="sec">The break-even p*, plotted directly</h2>
<p class="sub">The same p*, now read straight off the y-axis instead of inferred
from the gap between the curves. The
<b style="color:var(--pstarline)">solid line</b> is the bucket estimate (p*
computed from the bucket's mean V's, the number quoted above and in the heatmap).
Because every bucket pools thousands of distinct board states, p* also has a
<b>distribution</b>: the <b style="color:var(--pstar)">shaded band</b> is its
interquartile range (p25&ndash;p75) and the dashed line its median &mdash; wide
bands mean the same nominal matchup demands very different odds depending on the
rest of the board. The dashed <b>0.5</b> line is the coin-flip pivot: below it
the fight pays off at even-or-worse odds, above it you must be favored. p*
generally <b>climbs with your warrior edge</b> &mdash; the further ahead you are,
the less a fight can add and the more it can cost, so the bar to take it rises.</p>
<div class="legend">
  <div class="it"><span class="sw" style="border-color:var(--pstarline)"></span>p* (bucket mean)</div>
  <div class="it"><span class="sw" style="border-color:var(--pstar);border-top-style:dashed"></span>p* median</div>
  <div class="it"><span class="bx" style="background:var(--pstar);opacity:.16"></span>IQR (p25–p75)</div>
</div>
<div class="matrix-wrap"><div class="matrix" id="pgrid"></div></div>

<h2 class="sec">Global aggregates: model break-even vs empirical outcome</h2>
<p class="sub">Two single numbers per matchup, pooled over the whole dataset. The
<b>model p*</b> is the globally-averaged break-even (from each matchup's mean
V<sub>kill</sub> / V(S) / V<sub>death</sub>) — the odds a shot needs to be worth
it. The <b>empirical win%</b> and <b>K/D</b> come model-free from real kills: how
often an attacker piece actually killed the defender piece versus the reverse
(K/D = kills&nbsp;for ÷ against). The <b>margin</b> = win% − p*: <b style="color:var(--up)">positive</b>
means players win these exchanges more often than the break-even demands, so the
shot is <b>+EV on average</b>; <b style="color:var(--down)">negative</b> means it
is a losing proposition even before opportunity cost.</p>
<div class="matrix-wrap">{summary_table}</div>

<div class="foot">
  <h3>What the shapes mean in-game</h3>
  <p>The width of the two bands is the whole story. When the
  <b style="color:var(--down)">red</b> downside is thin next to the
  <b style="color:var(--up)">green</b> upside &mdash; the status-quo line hugging
  the death line &mdash; the fight is nearly free and p* is low: take it even as
  a heavy underdog. A few patterns to look for:</p>
  <ul>
    <li><b>Killing the queen is the biggest swing.</b> The <b>queen</b> column
      sits low (small p*): removing an enemy life is worth a risky fight. Flip
      the toggle to <b>0 (last life)</b> and V<sub>kill</sub> snaps to 1.0
      &mdash; that kill ends the game, so p* collapses toward 0. Killing a
      warrior only demotes it to a wingless drone, a milder swing, so warrior
      columns sit higher.</li>
    <li><b>Mirrors are coin flips.</b> On the diagonal (e.g. warrior vs the same
      warrior) p* &asymp; 0.5 at an even edge &mdash; a symmetric trade is worth
      it only at even-or-better odds.</li>
    <li><b>Being ahead raises the bar.</b> As your <b>warrior edge</b> grows the
      status quo is already winning, so there's little left to gain and more to
      lose: p* climbs to the right. Reverse matchups are transposes &mdash;
      speed&nbsp;warrior&nbsp;&rarr;&nbsp;vanilla and its mirror read as
      near-complements (p* and 1&minus;p*).</li>
  </ul>

  <h3>How the numbers are built</h3>
  <p>The <b>win-probability model</b> (gradient-boosted trees) maps a 52-feature
  game state to P(that team wins). For every real state in the data we form the
  three outcomes by applying the <em>exact</em> edits the model already tracks:
  a <b>queen kill</b> decrements that team's remaining lives (and ends the game
  on the last one); a <b>warrior kill</b> strips the unit's wings/speed back to a
  drone. Re-scoring the edited states gives V<sub>kill</sub> and V<sub>death</sub>;
  the unedited state gives V(S). Orienting each to the attacker's team yields the
  break-even p* above.</p>
  <p>Each cell pools <b>~{n_states} states</b> from {n_games} games, with both
  teams taking the attacker role, then <b>conditions</b> on two axes: the
  defender's remaining queen lives (the toggle) and the warrior edge (x). The
  line is p* from the bucket's mean V's (a stable point estimate); the band and
  dashed median describe the spread of the per-state p* within the bucket.</p>
  <p>The <b>empirical</b> win% and K/D are entirely model-free. We replay every
  game, and at each kill classify both killer and victim by piece type from the
  live game state (wings + speed; the queen is its own type), tallying a
  killer&times;victim count matrix. For a matchup A&nbsp;vs&nbsp;B the win% is
  A's share of the A&harr;B kills; K/D is kills-for over kills-against.</p>

  <h3>Caveats</h3>
  <ul>
    <li>p* is a threshold on the <b>conditional</b> probability that you win the
      duel given it resolves in a death &mdash; not "P(your strike lands)" and
      not a claim about how often such fights occur.</li>
    <li>The state carries no spatial information beyond the snail, so p* values
      composition, queen lives, berries and snail &mdash; not who is standing
      where. Positioning is exactly what real survival probability p depends on.</li>
    <li>The baseline is the pure status quo, which ignores opportunity cost (the
      berry or snail progress you forgo while fighting), so p* is, if anything,
      an <b>under</b>-estimate of the true bar.</li>
    <li>The <b>empirical</b> win% pools only fights that actually happened &mdash;
      a self-selected subset (players pick their spots), so margin&nbsp;= win%&minus;p*
      is suggestive, not a controlled test of the model.</li>
    <li>A speed drone earns no kill credit &mdash; a <b>baited</b> kill is scored
      to the teammate warrior who lands it &mdash; so the speed-drone attacker
      rows show only direct/snail kills and understate the bump mechanic the
      model's p* assumes. Likewise some drone-as-killer kills are snail crushes
      (credited to the rider), not piece duels.</li>
    <li>Hover any point for the exact V's, p*, median, IQR and state count.</li>
  </ul>
</div>
</div>
<div id="tip"></div>
<script>
const DATA = {data};
const tip = document.getElementById('tip');
const W = 300, H = 200, M = {{t: 10, r: 12, b: 30, l: 34}};
const PW = W - M.l - M.r, PH = H - M.t - M.b;
const XS = [-3, 4], YS = [0, 1];
const sx = x => M.l + (x - XS[0]) / (XS[1] - XS[0]) * PW;
const sy = y => M.t + (1 - (y - YS[0]) / (YS[1] - YS[0])) * PH;
const svgns = 'http://www.w3.org/2000/svg';
let curLv = '1';

function mk(t, a, txt) {{
  const e = document.createElementNS(svgns, t);
  for (const k in a) e.setAttribute(k, a[k]);
  if (txt != null) e.textContent = txt;
  return e;
}}
function bandPath(pts, lo, hi) {{
  let d = 'M';
  pts.forEach(p => {{ d += ` ${{sx(p.x)}} ${{sy(hi(p))}}`; }});
  for (let i = pts.length - 1; i >= 0; i--) {{ d += ` L ${{sx(pts[i].x)}} ${{sy(lo(pts[i]))}}`; }}
  return d + ' Z';
}}
function line(pts, fy, color, dash) {{
  const pl = pts.map(p => `${{sx(p.x)}},${{sy(fy(p))}}`).join(' ');
  return mk('polyline', {{points: pl, fill: 'none', stroke: color, 'stroke-width': 2,
    'stroke-dasharray': dash || '0',
    'stroke-linejoin': 'round', 'stroke-linecap': 'round'}});
}}

function panelSVG(pts) {{
  const s = mk('svg', {{viewBox: `0 0 ${{W}} ${{H}}`}});
  // gridlines + y labels
  [0, 0.25, 0.5, 0.75, 1.0].forEach(v => {{
    const y = sy(v);
    s.appendChild(mk('line', {{x1: M.l, y1: y, x2: M.l + PW, y2: y,
      stroke: v === 0.5 ? 'var(--axis)' : 'var(--line)',
      'stroke-dasharray': v === 0.5 ? '4 3' : '0', 'stroke-width': 1}}));
    s.appendChild(mk('text', {{x: M.l - 5, y: y + 3, 'text-anchor': 'end', class: 'ax'}}, v.toFixed(2)));
  }});
  for (let e = -3; e <= 4; e++)
    s.appendChild(mk('text', {{x: sx(e), y: H - M.b + 13, 'text-anchor': 'middle', class: 'ax'}},
      e > 0 ? '+' + e : '' + e));
  s.appendChild(mk('text', {{x: M.l + PW / 2, y: H - 3, 'text-anchor': 'middle', class: 'axtitle'}}, 'warrior edge'));
  s.appendChild(mk('text', {{x: -(M.t + PH / 2), y: 10, 'text-anchor': 'middle', class: 'axtitle',
    transform: 'rotate(-90)'}}, 'P(attacker wins)'));

  if (pts.length) {{
    // bands: downside (vD..vS) red, upside (vS..vK) green
    s.appendChild(mk('path', {{d: bandPath(pts, p => p.vD, p => p.vS),
      fill: 'var(--down)', 'fill-opacity': 0.16, stroke: 'none'}}));
    s.appendChild(mk('path', {{d: bandPath(pts, p => p.vS, p => p.vK),
      fill: 'var(--up)', 'fill-opacity': 0.16, stroke: 'none'}}));
    // lines
    s.appendChild(line(pts, p => p.vK, 'var(--upline)'));
    s.appendChild(line(pts, p => p.vD, 'var(--down)'));
    s.appendChild(line(pts, p => p.vS, 'var(--sqline)'));
    // hover hit targets at each edge (whole vertical strip)
    pts.forEach(p => {{
      s.appendChild(mk('circle', {{cx: sx(p.x), cy: sy(p.vS), r: 2.4, fill: 'var(--sqline)',
        stroke: 'var(--surface)', 'stroke-width': 1}}));
      const hit = mk('rect', {{x: sx(p.x) - 9, y: M.t, width: 18, height: PH,
        fill: 'transparent'}});
      hit.addEventListener('mousemove', ev => showTip(ev, p));
      hit.addEventListener('mouseleave', () => tip.style.opacity = 0);
      s.appendChild(hit);
    }});
  }}
  return s;
}}

function showTip(ev, p) {{
  const ps = p.pstar == null ? '—' : p.pstar.toFixed(3);
  tip.innerHTML =
    `<span class="k">warrior edge</span> ${{p.x > 0 ? '+' + p.x : p.x}}<br>`
    + `<b>p* = ${{ps}}</b> &nbsp; <span class="k">n</span> ${{p.n.toLocaleString()}}<br>`
    + `<span class="k">V<sub>kill</sub></span> ${{p.vK.toFixed(3)}}<br>`
    + `<span class="k">V(S)</span> ${{p.vS.toFixed(3)}}<br>`
    + `<span class="k">V<sub>death</sub></span> ${{p.vD.toFixed(3)}}`;
  positionTip(ev);
}}

function positionTip(ev) {{
  tip.style.opacity = 1;
  let x = ev.clientX + 14, y = ev.clientY + 14;
  const r = tip.getBoundingClientRect();
  if (x + r.width > innerWidth) x = ev.clientX - r.width - 14;
  if (y + r.height > innerHeight) y = ev.clientY - r.height - 14;
  tip.style.left = x + 'px'; tip.style.top = y + 'px';
}}

function showTipP(ev, p) {{
  const ps = p.pstar == null ? '—' : p.pstar.toFixed(3);
  const md = p.med == null ? '—' : p.med.toFixed(3);
  const iqr = (p.p25 == null || p.p75 == null) ? '—'
    : `[${{p.p25.toFixed(2)}}, ${{p.p75.toFixed(2)}}]`;
  tip.innerHTML =
    `<span class="k">warrior edge</span> ${{p.x > 0 ? '+' + p.x : p.x}}<br>`
    + `<b>p* = ${{ps}}</b> &nbsp; <span class="k">n</span> ${{p.n.toLocaleString()}}<br>`
    + `<span class="k">median</span> ${{md}}<br>`
    + `<span class="k">IQR</span> ${{iqr}}`;
  positionTip(ev);
}}

function pstarPanelSVG(pts) {{
  const s = mk('svg', {{viewBox: `0 0 ${{W}} ${{H}}`}});
  // gridlines + y labels (0.5 pivot emphasized, matching the panels above)
  [0, 0.25, 0.5, 0.75, 1.0].forEach(v => {{
    const y = sy(v);
    s.appendChild(mk('line', {{x1: M.l, y1: y, x2: M.l + PW, y2: y,
      stroke: v === 0.5 ? 'var(--axis)' : 'var(--line)',
      'stroke-dasharray': v === 0.5 ? '4 3' : '0', 'stroke-width': 1}}));
    s.appendChild(mk('text', {{x: M.l - 5, y: y + 3, 'text-anchor': 'end', class: 'ax'}}, v.toFixed(2)));
  }});
  for (let e = -3; e <= 4; e++)
    s.appendChild(mk('text', {{x: sx(e), y: H - M.b + 13, 'text-anchor': 'middle', class: 'ax'}},
      e > 0 ? '+' + e : '' + e));
  s.appendChild(mk('text', {{x: M.l + PW / 2, y: H - 3, 'text-anchor': 'middle', class: 'axtitle'}}, 'warrior edge'));
  s.appendChild(mk('text', {{x: -(M.t + PH / 2), y: 10, 'text-anchor': 'middle', class: 'axtitle',
    transform: 'rotate(-90)'}}, 'break-even p*'));

  const band = pts.filter(p => p.p25 != null && p.p75 != null);
  if (band.length) {{
    s.appendChild(mk('path', {{d: bandPath(band, p => p.p25, p => p.p75),
      fill: 'var(--pstar)', 'fill-opacity': 0.16, stroke: 'none'}}));
    s.appendChild(line(band, p => p.med, 'var(--pstar)', '4 3'));  // median, dashed
  }}
  const mean = pts.filter(p => p.pstar != null);
  if (mean.length) {{
    s.appendChild(line(mean, p => p.pstar, 'var(--pstarline)'));   // bucket-mean p*
    mean.forEach(p => s.appendChild(mk('circle', {{cx: sx(p.x), cy: sy(p.pstar), r: 2.4,
      fill: 'var(--pstarline)', stroke: 'var(--surface)', 'stroke-width': 1}})));
  }}
  // hover hit targets at each edge
  pts.forEach(p => {{
    const hit = mk('rect', {{x: sx(p.x) - 9, y: M.t, width: 18, height: PH, fill: 'transparent'}});
    hit.addEventListener('mousemove', ev => showTipP(ev, p));
    hit.addEventListener('mouseleave', () => tip.style.opacity = 0);
    s.appendChild(hit);
  }});
  return s;
}}

function pieceLabel(p) {{ return DATA.piece_label[p] || p; }}

function makeCard(mchp, panelFn) {{
  const pts = mchp.lives[curLv] || [];
  const card = document.createElement('div'); card.className = 'panel';
  card.title = mchp.subtitle;
  card.appendChild(panelFn(pts));
  return card;
}}

// Lay the matchups out as an attacker (rows) x defender (cols) matrix, so a
// matchup A->B sits transposed from its reverse B->A across the diagonal.
function renderMatrix(rootId, panelFn) {{
  const root = document.getElementById(rootId);
  root.innerHTML = '';
  const P = DATA.pieces;
  root.style.gridTemplateColumns = `max-content repeat(${{P.length}}, minmax(220px, 1fr))`;
  const idx = {{}};
  DATA.matchups.forEach(m => idx[m.atk + '|' + m.dfn] = m);

  // header row: corner + defender labels
  const corner = document.createElement('div'); corner.className = 'mcorner';
  corner.innerHTML = 'attacker&nbsp;↓<br>defender&nbsp;→';
  root.appendChild(corner);
  P.forEach(d => {{
    const h = document.createElement('div'); h.className = 'mhead';
    h.textContent = pieceLabel(d);
    root.appendChild(h);
  }});

  // one row per attacker piece (full 4x4; empty where no matchup exists)
  P.forEach(a => {{
    const rh = document.createElement('div'); rh.className = 'mrow-head';
    rh.textContent = pieceLabel(a);
    root.appendChild(rh);
    P.forEach(d => {{
      const m = idx[a + '|' + d];
      if (m) root.appendChild(makeCard(m, panelFn));
      else {{
        const e = document.createElement('div'); e.className = 'mempty';
        root.appendChild(e);
      }}
    }});
  }});
}}

function render() {{
  renderMatrix('grid', panelSVG);
  renderMatrix('pgrid', pstarPanelSVG);
}}

document.getElementById('seg').addEventListener('click', ev => {{
  const b = ev.target.closest('button'); if (!b) return;
  curLv = b.dataset.lv;
  [...document.querySelectorAll('#seg button')].forEach(x =>
    x.setAttribute('aria-pressed', x === b ? 'true' : 'false'));
  render();
}});
render();
</script>
</body></html>'''


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='current_preferred_model.mdl')
    ap.add_argument('--data', default='quality_filtered/encoded/all_games.bin')
    ap.add_argument('--max-games', type=int, default=3000)
    ap.add_argument('--out', default='combat_value/break_even_vdecomp.html')
    ap.add_argument('--from-cache', default=None,
                    help='build from a stream.py bucket cache (full-scale data)')
    ap.add_argument('--open', action='store_true')
    args = ap.parse_args()

    if args.from_cache:
        from combat_value import stream
        cache = stream.load_cache(args.from_cache)
        grids = {(a, d): stream.cache_grid(cache, a, d) for a, d, *_ in MATRIX_MATCHUPS}
        summaries = {(a, d): stream.cache_summary(cache, a, d)
                     for a, d, *_ in MATRIX_MATCHUPS}
        n_games, n_states, data_src = (cache['meta']['n_games'],
                                       cache['meta']['n_states'], cache['meta']['data'])
    else:
        print(f"Loading model {args.model}")
        model = lgb.Booster(model_file=args.model)
        print(f"Materializing up to {args.max_games} games from {args.data}")
        from event_codec import fast_materialize_from_codec
        X, y, gids, ts = fast_materialize_from_codec(args.data, max_games=args.max_games)
        print(f"  {len(X):,} states")
        grids, summaries = compute_grids(model, X, gids, ts)
        n_games, n_states, data_src = args.max_games, len(X), args.data

    # Empirical kill/death matrix (model-free); optional.
    kill_counts, kill_meta = None, None
    if os.path.exists(KILL_MATRIX):
        km = json.load(open(KILL_MATRIX))
        kill_counts, kill_meta = km['counts'], km['meta']
        print(f"Loaded kill matrix: {kill_meta['n_kills']:,} kills "
              f"from {kill_meta['n_games']:,} games")
    else:
        print(f"No {KILL_MATRIX}; empirical columns omitted "
              f"(run `python -m combat_value.kills`)")

    summary_rows = build_summary_rows(summaries, kill_counts)
    for r in summary_rows:
        ps = '   n/a' if r['pstar'] is None else f'{r["pstar"]:6.3f}'
        wr = '  n/a' if r['win_rate'] is None else f'{r["win_rate"]:5.3f}'
        print(f"  {r['atk']:15s} vs {r['dfn']:15s}  p*={ps}  win%={wr}")

    data = build(grids)
    kmeta = (f" · Kills: {kill_meta['n_kills']:,} over {kill_meta['n_games']:,} games"
             if kill_meta else '')
    meta = (f"Model: {os.path.basename(os.path.realpath(args.model))} · "
            f"Data: {data_src} · {n_games:,} games, "
            f"{n_states:,} states (both attacking sides){kmeta}")
    page = PAGE.format(meta=meta, data=json.dumps(data),
                       n_games=f"{n_games:,}", n_states=f"{n_states:,}",
                       summary_table=render_summary_table(summary_rows))

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
