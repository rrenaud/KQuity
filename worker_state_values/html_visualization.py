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
    CHARACTER_PRIORITY, build_worker_state_index,
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

# States with fewer observations than this are shown in a dimmed "rare" section.
_RARE_THRESHOLD = 10_000


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


def _sigmoid(x: float) -> float:
    """Logistic sigmoid: 1 / (1 + exp(-x))."""
    return 1.0 / (1.0 + np.exp(-x))


def _render_row(
    idx: int,
    worker_states: list[WorkerState],
    values: npt.NDArray[np.float64],
    counts: npt.NDArray[np.int64],
    baseline_idx: int,
    sprites: dict[tuple[str, bool], str],
    val_min: float,
    val_range: float,
    is_rare: bool,
) -> str:
    """Render a single table row for a worker state."""
    state = worker_states[idx]
    val = float(values[idx])
    cnt = int(counts[idx])
    is_baseline = int(idx) == baseline_idx
    win_pct = _sigmoid(val) * 100.0

    # Icons with state/char indices for interactivity
    assignments = assign_characters(state)
    icons = ''.join(
        _render_worker_icon(ct, iw, isp, sprites,
                            state_idx=int(idx), char_idx=ci)
        for ci, (ct, iw, isp) in enumerate(assignments)
    )

    baseline_mark = (' <span style="color:#ffeb3b;font-size:11px;">'
                     '★</span>') if is_baseline else ''

    # Bar: 0% = min value, 100% = max value
    bar_pct = (val - val_min) / val_range * 100
    # Color: red at bottom, green at top
    t = (val - val_min) / val_range
    r = int(244 * (1 - t) + 76 * t)
    g = int(67 * (1 - t) + 175 * t)
    b = int(54 * (1 - t) + 80 * t)
    bar_color = f'rgb({r},{g},{b})'

    # Row background tint
    tint = f'rgba({r},{g},{b},0.08)'

    classes = 'state-row'
    if is_baseline:
        classes += ' baseline'
    if is_rare:
        classes += ' rare'
    return (
        f'<tr class="{classes}" data-state-idx="{int(idx)}"'
        f' style="background:{tint};">'
        f'<td class="icons">{icons}{baseline_mark}</td>'
        f'<td class="bar-cell">'
        f'<div class="bar-container">'
        f'<div class="bar-fill" style="width:{bar_pct:.1f}%;'
        f'background:{bar_color};"></div>'
        f'</div></td>'
        f'<td class="win-pct">{win_pct:.1f}%</td>'
        f'<td class="count">{cnt:,}</td>'
        f'</tr>'
    )


def generate_html(
    worker_states: list[WorkerState],
    values: npt.NDArray[np.float64],
    counts: npt.NDArray[np.int64],
    baseline_idx: int,
    path: str,
) -> None:
    """Generate HTML visualization of worker state values.

    The generated HTML references external CSS/JS files (sibling files).
    After writing the HTML, copies CSS/JS from analysis/static/ to the
    same directory as the output file.
    """
    sprites = _load_sprites()
    sorted_indices = np.argsort(-values)  # descending
    val_min = float(np.min(values))
    val_max = float(np.max(values))
    val_range = val_max - val_min or 1.0

    # Compute transitions for hover interactivity
    worker_state_index = build_worker_state_index()
    transition_data = compute_transitions(
        worker_states, values, worker_state_index)

    # Partition into common and rare states
    common_indices = [i for i in sorted_indices if counts[i] >= _RARE_THRESHOLD]
    rare_indices = [i for i in sorted_indices if counts[i] < _RARE_THRESHOLD]

    rows: list[str] = []
    for idx in common_indices:
        rows.append(_render_row(
            idx, worker_states, values, counts, baseline_idx,
            sprites, val_min, val_range, is_rare=False))

    if rare_indices:
        n_rare = len(rare_indices)
        rare_count = sum(int(counts[i]) for i in rare_indices)
        total_count = int(np.sum(counts))
        rare_pct = rare_count / total_count * 100
        rows.append(
            f'<tr class="rare-separator"><td colspan="4">'
            f'Rare states ({n_rare}) &mdash; fewer than '
            f'{_RARE_THRESHOLD:,} observations each, '
            f'{rare_count:,} / {total_count:,} total observations '
            f'({rare_pct:.2f}%)'
            f'</td></tr>')
        for idx in rare_indices:
            rows.append(_render_row(
                idx, worker_states, values, counts, baseline_idx,
                sprites, val_min, val_range, is_rare=True))

    table_rows = '\n'.join(rows)

    # Legend icons — show each character as a drone
    sz = 32
    char_skull = _render_worker_icon('skull', False, False, sprites, sz)
    char_abs = _render_worker_icon('abs', False, False, sprites, sz)
    char_stripes = _render_worker_icon('stripes', False, False, sprites, sz)
    char_checkers = _render_worker_icon('checkers', False, False, sprites, sz)

    # Upgrade variants using skull as example
    legend_drone = _render_worker_icon('skull', False, False, sprites, sz)
    legend_speed = _render_worker_icon('skull', False, True, sprites, sz)
    legend_warrior = _render_worker_icon('skull', True, False, sprites, sz)
    legend_sw = _render_worker_icon('skull', True, True, sprites, sz)

    transition_json = json.dumps(transition_data, separators=(',', ':'))

    # Build sprites dict for JS: {char_type: {drone: dataURI, warrior: dataURI}}
    sprites_for_js = {}
    for char_type in CHARACTER_PRIORITY:
        sprites_for_js[char_type] = {
            'drone': sprites[(char_type, False)],
            'warrior': sprites[(char_type, True)],
        }
    sprites_json = json.dumps(sprites_for_js, separators=(',', ':'))
    char_priority_json = json.dumps(CHARACTER_PRIORITY)

    total_obs = int(np.sum(counts))

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
  We trained an AI model on a large set of quality-filtered Killer Queen games
  to predict each team's chance of winning at any moment during a match, based
  on 52 different in-game factors.
  </p>
  <p>
  To understand how much worker composition alone matters, we isolated its
  effect by averaging out everything else&mdash;berries, snail position, maiden
  control, queen lives&mdash;across {total_obs:,} game-state observations.
  <span style="color:#888;">(Technical detail, safe to skip: we fit
  <code>logit(P) &approx; f(blue_workers) &minus; f(gold_workers)</code>
  via OLS on the model's logit predictions, giving each worker state a single
  scalar value.)</span>
  The baseline is a team of 4 drones, which is set to exactly 50% (no
  advantage). The <strong>Win%</strong> column shows how much better or worse
  a given worker lineup is compared to that baseline, assuming everything else
  is equal.
  </p>
  <p>
  Hover over a worker icon to see how much the win probability changes when
  that worker dies or gets upgraded.
  </p>
</div>
<table>
<thead>
  <tr>
    <th>Workers</th>
    <th></th>
    <th style="text-align:right;">Win%</th>
    <th style="text-align:right;">Count</th>
  </tr>
</thead>
<tbody>
{table_rows}
</tbody>
</table>
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
