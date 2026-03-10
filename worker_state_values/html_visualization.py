"""HTML visualization for worker state values."""

import base64
import json
import re
import shutil
from pathlib import Path

import numpy as np
import numpy.typing as npt

from worker_state_values.team_state_value import (
    WorkerState, assign_characters, compute_transitions,
    CHARACTER_PRIORITY, build_worker_state_index, enumerate_worker_states,
)

# Map character type to filename suffix in kq_sprites/.
_CHAR_TO_FILENAME = {
    'skull': 'skulls',
    'abs': 'abs',
    'stripes': 'stripes',
    'checkers': 'checks',
}

_SPRITE_DIR = Path(__file__).parent.parent / 'kq_sprites'

# Regex patterns to strip wing elements from warrior SVGs (sword is kept).
_WING_RE = re.compile(
    r'<path[^>]*\bd="M16,17h-3[^"]*"[^>]*/>'
    r'|<path[^>]*\bd="M10,18v1h-2[^"]*"[^>]*/>'
)

# Lightning bolt overlay (speed) — yellow bolt in bottom-right corner.
_LIGHTNING_PATH = (
    '<polygon points="14,2 9,13 12,13 8,22 18,9 13,9 16,2" fill="#ffeb3b"/>'
)

# Directory containing static CSS/JS files.
_STATIC_DIR = Path(__file__).parent / 'static'

# States below this fraction of total tab observations go in the "low frequency" section.
_RARE_FRACTION = 0.001  # 0.1%


def _load_sprites(
    sprite_dir: Path = _SPRITE_DIR,
) -> dict[tuple[str, bool], str]:
    """Load all worker SVG sprites as base64 data URIs.

    Returns dict mapping (char_type, is_warrior) -> data URI string.
    When _SHOW_WARRIOR_WINGS is False, wing elements are stripped from
    warrior SVGs (the sword is kept).
    """
    sprites: dict[tuple[str, bool], str] = {}
    for char_type, fname in _CHAR_TO_FILENAME.items():
        for is_warrior in (False, True):
            role = 'warrior' if is_warrior else 'drone'
            path = sprite_dir / f'blue-{role}-{fname}.svg'
            svg_bytes = path.read_bytes()
            if is_warrior:
                svg_text = svg_bytes.decode('utf-8')
                svg_text = _WING_RE.sub('', svg_text)
                svg_bytes = svg_text.encode('utf-8')
            b64 = base64.b64encode(svg_bytes).decode('ascii')
            sprites[(char_type, is_warrior)] = (
                f'data:image/svg+xml;base64,{b64}')
    return sprites


def _svg_overlay_speed(size: int = 40) -> str:
    """Return inline SVG for lightning bolt overlay."""
    s = size * 2 // 5
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"'
        f' width="{s}" height="{s}"'
        f' style="position:absolute;right:-2px;bottom:-2px;z-index:3;">'
        f'{_LIGHTNING_PATH}</svg>'
    )


def _render_worker_icon(
    char_type: str, is_warrior: bool, is_speed: bool,
    sprites: dict[tuple[str, bool], str],
    size: int = 40,
    state_idx: int | None = None,
    char_idx: int | None = None,
) -> str:
    """Render a single worker icon using game sprites with optional speed overlay."""
    data_uri = sprites[(char_type, is_warrior)]
    if state_idx is not None and char_idx is not None:
        cls = ' class="worker-icon"'
        data = f' data-state-idx="{state_idx}" data-char-idx="{char_idx}"'
    else:
        cls = ''
        data = ''
    parts = [
        f'<div{cls}{data} style="position:relative;width:{size}px;height:{size}px;'
        f'display:inline-block;margin-right:2px;">'
        f'<img src="{data_uri}" width="{size}" height="{size}"'
        f' style="image-rendering:pixelated;">'
    ]
    if is_speed:
        parts.append(_svg_overlay_speed(size))
    parts.append('</div>')
    return ''.join(parts)


def _render_row(
    idx: int,
    worker_states: list[WorkerState],
    count: float,
    sprites: dict[tuple[str, bool], str],
    win_pct: float,
    freq_pct: float,
    win_min: float,
    win_range: float,
    is_rare: bool,
) -> str:
    """Render a single table row for a worker state."""
    state = worker_states[idx]

    assignments = assign_characters(state)
    icons = ''.join(
        _render_worker_icon(ct, iw, isp, sprites,
                            state_idx=int(idx), char_idx=ci)
        for ci, (ct, iw, isp) in enumerate(assignments)
    )

    bar_pct = (win_pct - win_min) / win_range * 100
    t = (win_pct - win_min) / win_range
    r = int(244 * (1 - t) + 76 * t)
    g = int(67 * (1 - t) + 175 * t)
    b = int(54 * (1 - t) + 80 * t)
    bar_color = f'rgb({r},{g},{b})'
    tint = f'rgba({r},{g},{b},0.08)'

    # Format count: integer format if whole number, else 1 decimal
    if count == int(count):
        count_str = f'{int(count):,}'
    else:
        count_str = f'{count:,.1f}'

    classes = 'state-row'
    if is_rare:
        classes += ' rare'
    return (
        f'<tr class="{classes}" data-state-idx="{int(idx)}"'
        f' style="background:{tint};">'
        f'<td class="icons">{icons}</td>'
        f'<td class="bar-cell">'
        f'<div class="bar-container">'
        f'<div class="bar-fill" style="width:{bar_pct:.1f}%;'
        f'background:{bar_color};"></div>'
        f'</div></td>'
        f'<td class="win-pct">{win_pct:.1f}%</td>'
        f'<td class="count" style="color:#555;">{freq_pct:.2f}%</td>'
        f'<td class="count">{count_str}</td>'
        f'</tr>'
    )


def _render_tab_content(
    worker_states: list[WorkerState],
    counts_arr: npt.NDArray,
    win_pcts: npt.NDArray[np.float64],
    sprites: dict[tuple[str, bool], str],
    win_min: float,
    win_range: float,
) -> str:
    """Render a full table for one tab (sorted by win%, split at 0.1% freq)."""
    total = float(np.sum(counts_arr))
    rare_thresh = _RARE_FRACTION * total

    sorted_indices = list(np.argsort(-win_pcts))
    common = [i for i in sorted_indices if counts_arr[i] >= rare_thresh]
    rare   = [i for i in sorted_indices if counts_arr[i] <  rare_thresh]

    rows: list[str] = []
    for idx in common:
        freq_pct = float(counts_arr[idx]) / total * 100 if total > 0 else 0
        rows.append(_render_row(
            idx, worker_states, float(counts_arr[idx]), sprites,
            float(win_pcts[idx]), freq_pct, win_min, win_range, is_rare=False))

    if rare:
        rare_count = sum(float(counts_arr[i]) for i in rare)
        rare_pct = rare_count / total * 100 if total > 0 else 0
        if total == int(total):
            total_str = f'{int(total):,}'
            rare_str = f'{int(rare_count):,}'
        else:
            total_str = f'{total:,.1f}'
            rare_str = f'{rare_count:,.1f}'
        rows.append(
            f'<tr class="rare-separator"><td colspan="5">'
            f'Low frequency states ({len(rare)}) &mdash; fewer than 0.1% of '
            f'observations each &mdash; '
            f'{rare_str} / {total_str} total ({rare_pct:.2f}%)'
            f'</td></tr>')
        for idx in rare:
            freq_pct = float(counts_arr[idx]) / total * 100 if total > 0 else 0
            rows.append(_render_row(
                idx, worker_states, float(counts_arr[idx]), sprites,
                float(win_pcts[idx]), freq_pct, win_min, win_range, is_rare=True))

    header = (
        '<table><thead><tr>'
        '<th>Workers</th>'
        '<th></th>'
        '<th style="text-align:right;">Win%</th>'
        '<th style="text-align:right;">Freq%</th>'
        '<th style="text-align:right;">Count</th>'
        '</tr></thead><tbody>'
    )
    return header + '\n'.join(rows) + '</tbody></table>'


WEIGHTING_MODE_LABELS = {
    'naive': 'Naive',
    'time': 'Time-weighted',
    'unique': 'Unique formations',
}


def generate_html(
    worker_states: list[WorkerState],
    counts: npt.NDArray[np.int64],
    win_prob: npt.NDArray[np.float64],
    path: str,
    win_prob_per_map: dict[str, list[float]] | None = None,
    counts_per_map: dict[str, list[int]] | None = None,
    n_games: int | None = None,
    weighting_modes: dict[str, dict] | None = None,
) -> None:
    """Generate HTML visualization of worker state values.

    The generated HTML references external CSS/JS files (sibling files).
    After writing the HTML, copies CSS/JS from analysis/static/ to the
    same directory as the output file.
    """
    sprites = _load_sprites()

    # Compute transitions for hover interactivity (naive mode)
    worker_state_index = build_worker_state_index()
    transition_data = compute_transitions(worker_states, win_prob, worker_state_index)

    # --- Build data for all (map, weighting) combos ---
    map_names = ['day', 'night', 'dusk', 'twilight']
    has_map_data = (win_prob_per_map is not None and counts_per_map is not None)
    has_weighting = weighting_modes is not None
    w_modes = list(WEIGHTING_MODE_LABELS.keys()) if has_weighting else ['naive']

    map_specs: list[tuple[str, str]] = [('overall', 'Overall')]
    if has_map_data:
        map_specs += [(m, m.capitalize()) for m in map_names]

    # Helper to get (counts_arr, win_pcts) for a given (map_id, weighting_mode)
    def _get_pane_data(
        map_id: str, mode: str,
    ) -> tuple[npt.NDArray, npt.NDArray[np.float64]]:
        if has_weighting and mode in weighting_modes:  # type: ignore[operator]
            md = weighting_modes[mode]  # type: ignore[index]
            if map_id == 'overall':
                wn = np.array(md['win_ns'], dtype=np.float64)
                wp = np.array(md['win_prob'], dtype=np.float64) * 100.0
            else:
                wn = np.array(md['win_ns_per_map'][map_id], dtype=np.float64)
                wp = np.array(md['win_prob_per_map'][map_id]) * 100.0
            return wn, wp
        # Fallback: use top-level naive data
        if map_id == 'overall':
            return counts.astype(np.float64), win_prob * 100.0
        return (np.array(counts_per_map[map_id], dtype=np.float64),  # type: ignore[index]
                np.array(win_prob_per_map[map_id]) * 100.0)  # type: ignore[index]

    # Compute win range per weighting mode (across all maps but not across modes),
    # so the bar scale is consistent across maps within a mode but each mode
    # gets its own scale.
    mode_win_range: dict[str, tuple[float, float]] = {}
    for mode in w_modes:
        common_win_pcts: list[float] = []
        for mid, _ in map_specs:
            c_arr, w_pcts = _get_pane_data(mid, mode)
            rare_thresh = _RARE_FRACTION * float(np.sum(c_arr))
            common_win_pcts.extend(w_pcts[c_arr >= rare_thresh].tolist())
        wmin = float(min(common_win_pcts))
        wmax = float(max(common_win_pcts))
        mode_win_range[mode] = (wmin, wmax - wmin or 1.0)

    # --- Map tab bar ---
    map_buttons = '\n  '.join(
        f'<button class="tab-btn map-tab{" active" if i == 0 else ""}" data-map="{mid}">'
        f'{label}</button>'
        for i, (mid, label) in enumerate(map_specs)
    )

    # --- Weighting tab bar (only if we have multi-mode data) ---
    if has_weighting:
        weight_buttons = '\n  '.join(
            f'<button class="tab-btn weight-tab{" active" if i == 0 else ""}" '
            f'data-weight="{mode}">{WEIGHTING_MODE_LABELS[mode]}</button>'
            for i, mode in enumerate(w_modes)
        )
        weight_bar_html = f'<div class="tab-bar weight-bar">\n  {weight_buttons}\n</div>'
    else:
        weight_bar_html = ''

    # --- Render all panes ---
    tab_panes: list[str] = []
    for mi, (mid, _) in enumerate(map_specs):
        for wi, mode in enumerate(w_modes):
            active = ' active' if (mi == 0 and wi == 0) else ''
            c_arr, w_pcts = _get_pane_data(mid, mode)
            win_min, win_range = mode_win_range[mode]
            table_html = _render_tab_content(
                worker_states, c_arr, w_pcts, sprites,
                win_min, win_range)
            pane_id = f'tab-{mid}-{mode}' if has_weighting else f'tab-{mid}'
            tab_panes.append(
                f'<div class="tab-pane{active}" id="{pane_id}">\n{table_html}\n</div>')

    tab_panes_html = '\n'.join(tab_panes)

    # Legend icons
    sz = 32
    char_skull   = _render_worker_icon('skull',    False, False, sprites, sz)
    char_abs     = _render_worker_icon('abs',      False, False, sprites, sz)
    char_stripes = _render_worker_icon('stripes',  False, False, sprites, sz)
    char_checkers= _render_worker_icon('checkers', False, False, sprites, sz)
    legend_drone  = _render_worker_icon('skull', False, False, sprites, sz)
    legend_speed  = _render_worker_icon('skull', False, True,  sprites, sz)
    legend_warrior= _render_worker_icon('skull', True,  False, sprites, sz)
    legend_sw     = _render_worker_icon('skull', True,  True,  sprites, sz)

    transition_json = json.dumps(transition_data, separators=(',', ':'))
    sprites_for_js = {
        ct: {'drone': sprites[(ct, False)], 'warrior': sprites[(ct, True)]}
        for ct in CHARACTER_PRIORITY
    }
    sprites_json = json.dumps(sprites_for_js, separators=(',', ':'))
    char_priority_json = json.dumps(CHARACTER_PRIORITY)

    # Per-mode win probs for JS tooltip computation
    if has_weighting:
        win_probs_by_mode: dict = {}
        map_win_probs_by_mode: dict = {}
        for mode in w_modes:
            md = weighting_modes[mode]  # type: ignore[index]
            win_probs_by_mode[mode] = [round(float(v), 6) for v in md['win_prob']]
            map_win_probs_by_mode[mode] = {
                m: [round(float(v), 6) for v in md['win_prob_per_map'][m]]
                for m in map_names
            } if 'win_prob_per_map' in md else {}
        win_probs_json = json.dumps(win_probs_by_mode['naive'], separators=(',', ':'))
        map_win_probs_json = json.dumps(
            map_win_probs_by_mode.get('naive', {}), separators=(',', ':'))
        win_probs_by_mode_json = json.dumps(win_probs_by_mode, separators=(',', ':'))
        map_win_probs_by_mode_json = json.dumps(map_win_probs_by_mode, separators=(',', ':'))
    else:
        win_probs_json = json.dumps(
            [round(float(p), 6) for p in win_prob], separators=(',', ':'))
        map_win_probs_json = json.dumps(
            {m: [round(float(v), 6) for v in win_prob_per_map[m]] for m in map_names}
            if win_prob_per_map else {},
            separators=(',', ':'))
        win_probs_by_mode_json = 'null'
        map_win_probs_by_mode_json = 'null'

    has_weighting_js = 'true' if has_weighting else 'false'

    total_obs = int(np.sum(counts))
    games_blurb = (f'{n_games:,} quality-filtered Killer Queen games'
                   if n_games is not None
                   else 'quality-filtered Killer Queen games')

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Worker State Values — KQuity</title>
<link rel="stylesheet" href="worker_state_values.css">
</head>
<body>
<h1>Worker State Values</h1>
<div class="methodology">
  <h2>How this works</h2>
  <p>
  For each worker formation, we computed the empirical win rate: the actual
  fraction of games won by teams with that formation, across all
  {total_obs:,} game-state observations from {games_blurb}.
  Each observation contributes once as blue and once as gold,
  marginalizing over all other factors (berries, snail, maidens, queen lives, opponents).
  </p>
  <p>
  <strong>Weighting modes:</strong>
  <ul class="mode-list">
    <li><strong>Naive</strong> &mdash; treats every event equally.</li>
    <li><strong>Time-weighted</strong> &mdash; weights each formation by how long it lasted before the next event.</li>
    <li><strong>Unique formations</strong> &mdash; counts each formation only once per continuous stretch
    (ignoring repeated events that don&rsquo;t change a team&rsquo;s worker composition).</li>
  </ul>
  </p>
  <p>
  Hover over a worker icon to see how the win rate changes when that worker
  dies or gets upgraded, and how the formation performs on each map.
  </p>
</div>
<div class="tab-bar map-bar">
  {map_buttons}
</div>
{weight_bar_html}
{tab_panes_html}
<div class="legend" style="margin-top:20px;">
  <div class="legend-section">
    <h3>Characters (priority order)</h3>
    <div class="legend-item">{char_skull} Skull (highest)</div>
    <div class="legend-item">{char_abs} Abs</div>
    <div class="legend-item">{char_stripes} Stripes</div>
    <div class="legend-item">{char_checkers} Checkers (lowest)</div>
  </div>
  <div class="legend-section">
    <h3>Upgrades (highest to lowest)</h3>
    <div class="legend-item">{legend_sw} Speed warrior (+ lightning)</div>
    <div class="legend-item">{legend_warrior} Warrior (wings + sword)</div>
    <div class="legend-item">{legend_speed} Speed drone (+ lightning)</div>
    <div class="legend-item">{legend_drone} Drone</div>
  </div>
</div>
<div id="transition-tooltip"></div>
<script>
const TRANSITIONS = {transition_json};
const SPRITES = {sprites_json};
const CHAR_PRIORITY = {char_priority_json};
const LIGHTNING_SVG = '{_LIGHTNING_PATH}';
const STATE_WIN_PROBS = {win_probs_json};
const STATE_MAP_WIN_PROBS = {map_win_probs_json};
const HAS_WEIGHTING = {has_weighting_js};
const WIN_PROBS_BY_MODE = {win_probs_by_mode_json};
const MAP_WIN_PROBS_BY_MODE = {map_win_probs_by_mode_json};
</script>
<script>
// 2D tab switching: map × weighting mode
(function() {{
  let activeMap = '{map_specs[0][0]}';
  let activeWeight = '{w_modes[0]}';

  function showActivePane() {{
    document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
    const id = HAS_WEIGHTING
      ? 'tab-' + activeMap + '-' + activeWeight
      : 'tab-' + activeMap;
    const pane = document.getElementById(id);
    if (pane) pane.classList.add('active');
  }}

  document.querySelectorAll('.map-tab').forEach(btn => {{
    btn.addEventListener('click', () => {{
      document.querySelectorAll('.map-tab').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      activeMap = btn.dataset.map;
      showActivePane();
    }});
  }});

  document.querySelectorAll('.weight-tab').forEach(btn => {{
    btn.addEventListener('click', () => {{
      document.querySelectorAll('.weight-tab').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      activeWeight = btn.dataset.weight;
      showActivePane();
    }});
  }});
}})();
</script>
<script src="worker_state_values.js"></script>
</body>
</html>'''

    out_path = Path(path)
    with open(out_path, 'w') as f:
        f.write(html)
    print(f"\nHTML visualization written to {path}")

    # Copy static CSS/JS alongside the output HTML
    out_dir = out_path.parent
    for static_file in ('worker_state_values.css', 'worker_state_values.js'):
        src = _STATIC_DIR / static_file
        dst = out_dir / static_file
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
            print(f"  Copied {static_file} to {out_dir}")
