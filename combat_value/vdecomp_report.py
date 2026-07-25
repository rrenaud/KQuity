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
KD_WINPROB = 'combat_value/_kd_winprob.json'

# Piece sprites (pixel-art icons from the worker-state-values set) inlined into
# the page. Each piece maps to a base sprite + whether it carries the speed
# upgrade (shown with a small badge); the queen has its own icon.
_ICON_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'icons')
PIECE_ICON = {
    'queen':           ('queen',   False),
    'speed_warrior':   ('warrior', True),
    'vanilla_warrior': ('warrior', False),
    'speed_drone':     ('drone',   True),
    'drone':           ('drone',   False),
}
# The three striking ("military") pieces, used for the global aggregates grid.
MILITARY_PIECES = ['queen', 'speed_warrior', 'vanilla_warrior']


def _icon_uri(name: str) -> str:
    import base64
    with open(os.path.join(_ICON_DIR, f'{name}.svg'), 'rb') as f:
        b64 = base64.b64encode(f.read()).decode('ascii')
    return f'data:image/svg+xml;base64,{b64}'


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
    grids, summaries, wp_gs = {}, {}, {}
    for atk, dfn, _cap in MATRIX_MATCHUPS:
        res = results[(atk, dfn)]
        rows = analysis.bucket_table(res, ['def_eggs', 'net_warriors'])
        grids[(atk, dfn)] = {(r['def_eggs'], r['net_warriors']): r for r in rows}
        summaries[(atk, dfn)] = analysis.summarize(res)
        s, c = analysis.pstar_by_winprob(res)
        wp_gs[f'{atk}|{dfn}'] = {'ps_sum': [round(x, 4) for x in s.tolist()],
                                 'cnt': c.astype(int).tolist()}
        print(f"  {atk:16s} vs {dfn:16s}")
    return grids, summaries, wp_gs


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
        has = summ.get('n', 0) > 0
        rows.append({
            'atk': atk, 'dfn': dfn,
            'pstar': pstar,
            'n_states': int(summ.get('n', 0)),
            'vS': round(summ['mean_V_status_quo'], 4) if has else None,
            'vK': round(summ['mean_V_kill'], 4) if has else None,
            'vD': round(summ['mean_V_death'], 4) if has else None,
            'win_rate': wr,
            'kd': emp['kd'] if emp else None,
            'kills_for': emp['kills_for'] if emp else None,
            'kills_ag': emp['kills_ag'] if emp else None,
            'margin': (wr - pstar) if (wr is not None and pstar is not None) else None,
        })
    return rows


def build(grids, summary_rows, winprob=None, winprob_gs=None):
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
    summary = {f"{r['atk']}|{r['dfn']}": r for r in summary_rows}
    return {'matchups': matchups, 'edges': EDGES, 'pieces': pieces,
            'piece_label': {p: PIECE_LABEL[p] for p in pieces},
            'piece_icon': {p: {'icon': PIECE_ICON[p][0], 'speed': PIECE_ICON[p][1]}
                           for p in pieces},
            'icons': {k: _icon_uri(k) for k in ('queen', 'warrior', 'drone')},
            'military': [p for p in MILITARY_PIECES if p in used],
            'summary': summary,
            'winprob': (winprob or {}).get('matchups', {}),
            'winprob_bins': (winprob or {}).get('bins', 50),
            'winprob_gs': winprob_gs or {},
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
.panel.zoomable {{ cursor:zoom-in; }}
#zoom {{ position:fixed; inset:0; z-index:50; display:none; align-items:center;
  justify-content:center; background:rgba(0,0,0,.72); padding:24px; cursor:zoom-out; }}
#zoom.open {{ display:flex; }}
#zoom .zbox {{ background:var(--surface); border:1px solid var(--ring); border-radius:14px;
  padding:20px 24px 22px; width:min(94vw, 940px); max-height:94vh; overflow:auto;
  box-shadow:0 24px 70px rgba(0,0,0,.55); }}
#zoom .zttl {{ font-size:15px; font-weight:650; color:var(--ink); margin:0 0 10px;
  display:flex; align-items:center; gap:8px; }}
#zoom .zttl .esc {{ margin-left:auto; font-size:11px; font-weight:400; color:var(--mut); }}
#zoom svg {{ width:100%; height:auto; display:block; overflow:visible; }}
.ax {{ fill:var(--mut); font-size:10px; }}
.axtitle {{ fill:var(--mut); font-size:10px; }}
/* piece icon headers (shared by every grid) */
.phead {{ display:flex; align-items:center; gap:6px; }}
.phead.col {{ flex-direction:column; gap:3px; }}
.picon {{ position:relative; display:inline-flex; width:32px; height:32px;
  flex:0 0 auto; }}
.picon img {{ width:100%; height:100%; image-rendering:pixelated; }}
.picon .spd {{ position:absolute; right:-3px; bottom:-3px; width:14px; height:14px;
  line-height:0; filter:drop-shadow(0 0 1px rgba(0,0,0,.7)); }}
.picon .spd svg {{ width:100%; height:100%; display:block; }}
.plabel {{ font-size:12.5px; font-weight:600; color:var(--ink); }}
/* aggregate matrix */
.aggmatrix {{ min-width:560px; }}
.aggcell {{ display:flex; flex-direction:column; gap:3px; padding:11px 13px;
  font-variant-numeric:tabular-nums; }}
.aggcell .arow {{ display:flex; justify-content:space-between; align-items:baseline;
  gap:12px; font-size:13px; }}
.aggcell .arow .k {{ color:var(--mut); font-size:11.5px; }}
.aggcell .arow .v {{ font-weight:600; }}
.aggcell .marg {{ margin-top:2px; padding-top:4px; border-top:1px solid var(--line);
  font-weight:700; }}
.aggcell .marg.mpos {{ color:var(--up); }}
.aggcell .marg.mneg {{ color:var(--down); }}
.aggcell .marg.mzero {{ color:var(--mut); }}
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

<h2 class="sec">Global aggregates: model break-even vs empirical outcome</h2>
<p class="sub">Before the per-state detail below, the bottom line for the three
striking pieces. Each cell is one matchup (attacker <b>row</b> → defender
<b>column</b>). <b>model p*</b> is the globally-averaged break-even &mdash; the
odds a shot needs to be worth it. <b>emp win%</b> and <b>K/D</b> are model-free,
from real kills: how often that attacker piece actually killed that defender
versus the reverse (K/D = kills&nbsp;for ÷ against). The
<b>margin = win% − p*</b> is the verdict: <b style="color:var(--up)">positive</b>
(green) means players win these exchanges more often than the break-even demands
&mdash; the shot is <b>+EV on average</b>; <b style="color:var(--down)">negative</b>
(red) means a losing proposition even before opportunity cost.</p>
<div class="matrix-wrap"><div class="matrix aggmatrix" id="aggrid"></div></div>

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

<h2 class="sec">Empirical win rate vs break-even, across the game</h2>
<p class="sub">Does real play clear the bar as the game swings? Every state is
scored by the model and dropped into one of 50 buckets along
<b>P(attacker wins)</b>. Per matchup, the
<b style="color:var(--sqline)">solid line</b> is the <b>empirical win rate</b> in
each bucket (the attacker's share of the A&harr;B kills, K/(K+D)). The two dashed
lines are the model's <b>break-even p*</b> &mdash; what it thinks is optimal: the
<b style="color:var(--pstar)">violet</b> one is p* at the <b>fights that actually
happened</b>, the <b style="color:var(--upline)">green</b> one is p* over
<b>all game states</b> in the bucket (its general prescription, not conditioned
on a kill). Where the solid line sits <b>above</b> a dashed one, those shots pay
off on average; read left-to-right to see how the verdict shifts from behind
(left) to ahead (right).</p>
<div class="legend">
  <div class="it"><span class="sw" style="border-color:var(--sqline)"></span>empirical win rate (K/(K+D))</div>
  <div class="it"><span class="sw" style="border-color:var(--pstar);border-top-style:dashed"></span>model p* · at these fights</div>
  <div class="it"><span class="sw" style="border-color:var(--upline);border-top-style:dotted"></span>model p* · all game states</div>
</div>
<div class="matrix-wrap"><div class="matrix" id="wpgrid"></div></div>

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
    <li>The aggregates cover only the three <b>striking</b> pieces (queen and the
      two warriors). Speed drones can't take kill credit &mdash; a baited kill is
      scored to the teammate who lands it &mdash; so they're left out of the
      empirical grid, though they remain in the per-state panels below as an
      attacker via that bump mechanic.</li>
    <li>Hover any point for the exact V's, p*, median, IQR and state count.</li>
  </ul>
</div>
</div>
<div id="tip"></div>
<div id="zoom"><div class="zbox">
  <div class="zttl"><span id="ztitle"></span><span class="esc">click anywhere or Esc to close</span></div>
  <div id="zsvg"></div>
</div></div>
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

// Icon + optional speed badge + label, for a grid's row/column header.
function pieceHeader(piece, col) {{
  const info = DATA.piece_icon[piece] || {{}};
  const wrap = document.createElement('div');
  wrap.className = 'phead' + (col ? ' col' : '');
  const ic = document.createElement('span'); ic.className = 'picon';
  const src = DATA.icons[info.icon];
  if (src) {{
    const img = document.createElement('img'); img.src = src; img.alt = piece;
    ic.appendChild(img);
    if (info.speed) {{
      const b = document.createElement('span'); b.className = 'spd'; b.title = 'speed';
      b.innerHTML = '<svg viewBox="0 0 24 24"><polygon points="14,2 9,13 12,13 8,22'
        + ' 18,9 13,9 16,2" fill="#ffeb3b"/></svg>';
      ic.appendChild(b);
    }}
  }}
  wrap.appendChild(ic);
  const lab = document.createElement('span'); lab.className = 'plabel';
  lab.textContent = pieceLabel(piece);
  wrap.appendChild(lab);
  return wrap;
}}

function fmtKD(s) {{
  if (s.kills_ag === 0 && s.kills_for) return '∞';
  if (s.kd == null) return '—';
  return s.kd >= 100 ? Math.round(s.kd).toLocaleString() : s.kd.toFixed(2);
}}

function makeCard(mchp, panelFn) {{
  const pts = mchp.lives[curLv] || [];
  const card = document.createElement('div'); card.className = 'panel zoomable';
  card.title = mchp.subtitle;
  card.dataset.label = mchp.title;
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
    h.appendChild(pieceHeader(d, true));
    root.appendChild(h);
  }});

  // one row per attacker piece (full 4x4; empty where no matchup exists)
  P.forEach(a => {{
    const rh = document.createElement('div'); rh.className = 'mrow-head';
    rh.appendChild(pieceHeader(a, false));
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

// Global aggregates grid: attacker x defender over the striking pieces, each
// cell a compact model-p* / empirical-win% / K/D / margin block.
function aggCell(atk, dfn) {{
  const s = DATA.summary[atk + '|' + dfn];
  const cell = document.createElement('div');
  if (!s) {{ cell.className = 'mempty'; return cell; }}
  cell.className = 'panel aggcell';
  const stat = (k, v) =>
    `<div class="arow"><span class="k">${{k}}</span><span class="v">${{v}}</span></div>`;
  const ps = s.pstar == null ? '—' : s.pstar.toFixed(2);
  const wr = s.win_rate == null ? '—' : s.win_rate.toFixed(2);
  const m = s.margin;
  const mcls = m == null ? 'mzero' : (m > 0.02 ? 'mpos' : (m < -0.02 ? 'mneg' : 'mzero'));
  const ms = m == null ? '—' : (m >= 0 ? '+' : '') + m.toFixed(2);
  cell.innerHTML =
    stat('model p*', ps) + stat('emp win%', wr) + stat('K/D', fmtKD(s))
    + `<div class="arow marg ${{mcls}}"><span class="k">margin</span>`
    + `<span class="v">${{ms}}</span></div>`;
  return cell;
}}

function renderAggregates() {{
  const root = document.getElementById('aggrid');
  if (!root || !DATA.military) return;
  root.innerHTML = '';
  const P = DATA.military;
  root.style.gridTemplateColumns = `max-content repeat(${{P.length}}, minmax(150px, 1fr))`;
  const corner = document.createElement('div'); corner.className = 'mcorner';
  corner.innerHTML = 'attacker&nbsp;↓<br>defender&nbsp;→';
  root.appendChild(corner);
  P.forEach(d => {{
    const h = document.createElement('div'); h.className = 'mhead';
    h.appendChild(pieceHeader(d, true));
    root.appendChild(h);
  }});
  P.forEach(a => {{
    const rh = document.createElement('div'); rh.className = 'mrow-head';
    rh.appendChild(pieceHeader(a, false));
    root.appendChild(rh);
    P.forEach(d => {{
      if (a === d) {{  // drop the diagonal: mirror matchups are trivial coin flips
        const e = document.createElement('div'); e.className = 'mempty';
        root.appendChild(e);
      }} else {{
        root.appendChild(aggCell(a, d));
      }}
    }});
  }});
}}

// Per matchup: empirical win rate (solid) and model p* (dashed) across the
// 50 P(attacker-wins) buckets.
function winprobPanelSVG(m, gs, nb) {{
  const svg = mk('svg', {{viewBox: `0 0 ${{W}} ${{H}}`}});
  const sxp = p => M.l + p * PW;
  [0, 0.25, 0.5, 0.75, 1.0].forEach(v => {{
    const y = sy(v);
    svg.appendChild(mk('line', {{x1: M.l, y1: y, x2: M.l + PW, y2: y,
      stroke: v === 0.5 ? 'var(--axis)' : 'var(--line)',
      'stroke-dasharray': v === 0.5 ? '4 3' : '0', 'stroke-width': 1}}));
    svg.appendChild(mk('text', {{x: M.l - 5, y: y + 3, 'text-anchor': 'end', class: 'ax'}}, v.toFixed(2)));
    svg.appendChild(mk('text', {{x: sxp(v), y: H - M.b + 13, 'text-anchor': 'middle', class: 'ax'}}, v.toFixed(2)));
  }});
  svg.appendChild(mk('text', {{x: M.l + PW / 2, y: H - 3, 'text-anchor': 'middle', class: 'axtitle'}}, 'P(attacker wins)'));
  svg.appendChild(mk('text', {{x: -(M.t + PH / 2), y: 10, 'text-anchor': 'middle', class: 'axtitle',
    transform: 'rotate(-90)'}}, 'win rate / p*'));

  const MINN = 200;
  const emp = [], ps = [], gsp = [];
  for (let i = 0; i < nb; i++) {{
    const n = m.wins[i] + m.losses[i];
    const x = (i + 0.5) / nb;
    if (n >= MINN) emp.push({{x: x, y: m.wins[i] / n}});
    if (m.pstar_cnt[i] >= MINN) ps.push({{x: x, y: m.pstar_sum[i] / m.pstar_cnt[i]}});
    if (gs && gs.cnt[i] >= MINN) gsp.push({{x: x, y: gs.ps_sum[i] / gs.cnt[i]}});
  }}
  const poly = (pts, color, dash, wd) => {{
    if (pts.length < 2) return;
    svg.appendChild(mk('polyline', {{points: pts.map(p => `${{sxp(p.x)}},${{sy(p.y)}}`).join(' '),
      fill: 'none', stroke: color, 'stroke-width': wd, 'stroke-dasharray': dash || '0',
      'stroke-linejoin': 'round', 'stroke-linecap': 'round'}}));
  }};
  poly(gsp, 'var(--upline)', '1 3', 2);      // model p* over all game states (dotted green)
  poly(ps, 'var(--pstar)', '5 3', 2);        // model p* at these fights (dashed violet)
  poly(emp, 'var(--sqline)', '0', 2.4);      // empirical win rate (solid ink)
  return svg;
}}

function renderWinprob() {{
  const root = document.getElementById('wpgrid');
  if (!root || !DATA.winprob || !DATA.military) return;
  root.innerHTML = '';
  const P = DATA.military, nb = DATA.winprob_bins;
  root.style.gridTemplateColumns = `max-content repeat(${{P.length}}, minmax(220px, 1fr))`;
  const corner = document.createElement('div'); corner.className = 'mcorner';
  corner.innerHTML = 'attacker&nbsp;↓<br>defender&nbsp;→';
  root.appendChild(corner);
  P.forEach(d => {{
    const h = document.createElement('div'); h.className = 'mhead';
    h.appendChild(pieceHeader(d, true));
    root.appendChild(h);
  }});
  P.forEach(a => {{
    const rh = document.createElement('div'); rh.className = 'mrow-head';
    rh.appendChild(pieceHeader(a, false));
    root.appendChild(rh);
    P.forEach(d => {{
      const m = (a === d) ? null : DATA.winprob[a + '|' + d];
      if (!m) {{ const e = document.createElement('div'); e.className = 'mempty'; root.appendChild(e); return; }}
      const card = document.createElement('div'); card.className = 'panel zoomable';
      card.dataset.label = pieceLabel(a) + ' → ' + pieceLabel(d) + '  ·  win rate vs p*';
      card.appendChild(winprobPanelSVG(m, (DATA.winprob_gs || {{}})[a + '|' + d], nb));
      root.appendChild(card);
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
renderAggregates();
renderWinprob();
render();

// Click a plot to open an enlarged copy; click anywhere / Esc to close.
const zoom = document.getElementById('zoom');
document.addEventListener('click', e => {{
  if (zoom.contains(e.target)) return;
  const panel = e.target.closest && e.target.closest('.panel.zoomable');
  const svg = panel && panel.querySelector(':scope > svg');
  if (!svg) return;
  document.getElementById('ztitle').textContent = panel.dataset.label || '';
  const zsvg = document.getElementById('zsvg');
  zsvg.innerHTML = '';
  zsvg.appendChild(svg.cloneNode(true));
  zoom.classList.add('open');
}});
zoom.addEventListener('click', () => zoom.classList.remove('open'));
document.addEventListener('keydown', e => {{
  if (e.key === 'Escape') zoom.classList.remove('open');
}});
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
        winprob_gs = {}
        for a, d, *_ in MATRIX_MATCHUPS:
            e = cache['matchups'].get(stream.matchup_key(a, d), {})
            if 'winprob_pstar_sum' in e:
                winprob_gs[f'{a}|{d}'] = {'ps_sum': e['winprob_pstar_sum'],
                                         'cnt': e['winprob_cnt']}
        n_games, n_states, data_src = (cache['meta']['n_games'],
                                       cache['meta']['n_states'], cache['meta']['data'])
    else:
        print(f"Loading model {args.model}")
        model = lgb.Booster(model_file=args.model)
        print(f"Materializing up to {args.max_games} games from {args.data}")
        from event_codec import fast_materialize_from_codec
        X, y, gids, ts = fast_materialize_from_codec(args.data, max_games=args.max_games)
        print(f"  {len(X):,} states")
        grids, summaries, winprob_gs = compute_grids(model, X, gids, ts)
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

    winprob = None
    if os.path.exists(KD_WINPROB):
        winprob = json.load(open(KD_WINPROB))
        print(f"Loaded win-prob buckets: {winprob['meta']['n_kills']:,} kills, "
              f"{winprob['bins']} bins")

    summary_rows = build_summary_rows(summaries, kill_counts)
    for r in summary_rows:
        ps = '   n/a' if r['pstar'] is None else f'{r["pstar"]:6.3f}'
        wr = '  n/a' if r['win_rate'] is None else f'{r["win_rate"]:5.3f}'
        print(f"  {r['atk']:15s} vs {r['dfn']:15s}  p*={ps}  win%={wr}")

    data = build(grids, summary_rows, winprob, winprob_gs)
    kmeta = (f" · Kills: {kill_meta['n_kills']:,} over {kill_meta['n_games']:,} games"
             if kill_meta else '')
    meta = (f"Model: {os.path.basename(os.path.realpath(args.model))} · "
            f"Data: {data_src} · {n_games:,} games, "
            f"{n_states:,} states (both attacking sides){kmeta}")
    page = PAGE.format(meta=meta, data=json.dumps(data),
                       n_games=f"{n_games:,}", n_states=f"{n_states:,}")

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
