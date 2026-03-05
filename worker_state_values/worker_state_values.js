const TYPE_SYMBOL = {death: "\u2620", speed: "\u26a1", warrior: "\u2694"};
const MAP_KEYS = ['day', 'night', 'dusk', 'twilight'];
const MAP_LABELS = {day: 'Day', night: 'Ngt', dusk: 'Dsk', twilight: 'Twi'};
const tooltip = document.getElementById("transition-tooltip");

const transMap = {};
TRANSITIONS.forEach(t => {
  transMap[t.src + ":" + t.char] = t;
});
const rowByState = {};
document.querySelectorAll("tr[data-state-idx]").forEach(tr => {
  rowByState[tr.dataset.stateIdx] = tr;
});

function clearHighlights() {
  document.querySelectorAll(".highlight-source,.highlight-upgrade,.highlight-death")
    .forEach(el => el.classList.remove("highlight-source","highlight-upgrade","highlight-death"));
  document.querySelectorAll(".delta-badge").forEach(el => el.remove());
  tooltip.style.display = "none";
}

function renderMiniIcon(charType, isWarrior, isSpeed) {
  const size = 36;
  const dataUri = SPRITES[charType][isWarrior ? "warrior" : "drone"];
  let html = '<div style="position:relative;width:' + size + 'px;height:' + size + 'px;display:inline-block;">';
  html += '<img src="' + dataUri + '" width="' + size + '" height="' + size + '" style="image-rendering:pixelated;">';
  if (isSpeed) {
    const s = Math.round(size * 2 / 5);
    html += '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="' + s + '" height="' + s
      + '" style="position:absolute;right:-2px;bottom:-2px;z-index:3;">' + LIGHTNING_SVG + '</svg>';
  }
  html += '</div>';
  return html;
}

function targetAppearance(type, srcWarrior, srcSpeed) {
  if (type === "death") return {warrior: false, speed: false};
  if (type === "speed") return {warrior: false, speed: true};
  return {warrior: true, speed: srcSpeed};
}

function fmtDelta(v) {
  return (v >= 0 ? '+' : '') + v.toFixed(1) + 'pp';
}

document.addEventListener("mouseover", function(e) {
  const icon = e.target.closest(".worker-icon");
  if (!icon) return;
  const key = icon.dataset.stateIdx + ":" + icon.dataset.charIdx;
  const trans = transMap[key];
  if (!trans) return;
  clearHighlights();

  const srcRow = icon.closest("tr");
  if (srcRow) srcRow.classList.add("highlight-source");

  const charType = CHAR_PRIORITY[trans.char];
  const hasMapData = STATE_MAP_WIN_PROBS && Object.keys(STATE_MAP_WIN_PROBS).length > 0;

  const activeTab = (document.querySelector('.tab-btn.active') || {dataset: {tab: 'overall'}}).dataset.tab;
  const colActive = col => col === activeTab ? ' tooltip-col-active' : '';

  let tooltipHtml = '<table class="tooltip-table"><thead><tr><th></th>'
    + '<th class="' + colActive('overall').trim() + '">Overall</th>';
  if (hasMapData) MAP_KEYS.forEach(m => {
    tooltipHtml += '<th class="' + colActive(m).trim() + '">' + MAP_LABELS[m] + '</th>';
  });
  tooltipHtml += '</tr></thead><tbody>';

  trans.actions.forEach(a => {
    const tgt = targetAppearance(a.type, trans.src_warrior, trans.src_speed);
    const srcIcon = renderMiniIcon(charType, trans.src_warrior, trans.src_speed);
    const tgtIcon = renderMiniIcon(charType, tgt.warrior, tgt.speed);
    const arrowCls = "tooltip-arrow tooltip-arrow-" + a.type;

    tooltipHtml += '<tr><td class="tooltip-action-cell">'
      + srcIcon
      + '<span class="' + arrowCls + '">' + TYPE_SYMBOL[a.type] + '</span>'
      + tgtIcon
      + '</td>';

    const cls = a.win_delta >= 0 ? 'positive' : 'negative';
    tooltipHtml += '<td class="tooltip-delta ' + cls + colActive('overall') + '">' + fmtDelta(a.win_delta) + '</td>';

    if (hasMapData) {
      MAP_KEYS.forEach(m => {
        const probs = STATE_MAP_WIN_PROBS[m];
        if (!probs) { tooltipHtml += '<td>—</td>'; return; }
        const delta = (probs[a.target] - probs[trans.src]) * 100;
        const mcls = delta >= 0 ? 'positive' : 'negative';
        tooltipHtml += '<td class="tooltip-delta ' + mcls + colActive(m) + '">' + fmtDelta(delta) + '</td>';
      });
    }
    tooltipHtml += '</tr>';
  });

  tooltipHtml += '</tbody></table>';

  tooltip.innerHTML = tooltipHtml;
  tooltip.style.display = 'block';

  // Anchor tooltip to the right of the icons cell of the actual hovered row,
  // so it never covers the worker icons.
  const iconsCell = srcRow ? srcRow.querySelector('.icons') : icon;
  const anchorRect = (iconsCell || icon).getBoundingClientRect();
  const tipW = tooltip.offsetWidth;
  const tipH = tooltip.offsetHeight;
  let left = anchorRect.right + 16;
  let top = anchorRect.top;
  if (left + tipW > window.innerWidth - 8) left = anchorRect.left - tipW - 16;
  if (top + tipH > window.innerHeight - 8) top = window.innerHeight - tipH - 8;
  if (top < 0) top = 0;
  tooltip.style.left = left + 'px';
  tooltip.style.top = top + 'px';

  // Highlight target rows + add delta badges (match active tab rows)
  trans.actions.forEach(a => {
    const row = rowByState[a.target];
    if (!row) return;
    row.classList.add(a.type === "death" ? "highlight-death" : "highlight-upgrade");
    const winCell = row.querySelector(".win-pct");
    if (winCell) {
      const badge = document.createElement("span");
      badge.className = "delta-badge " + (a.win_delta >= 0 ? "positive" : "negative");
      badge.textContent = TYPE_SYMBOL[a.type] + " " + fmtDelta(a.win_delta);
      winCell.appendChild(badge);
    }
  });
});

document.addEventListener("mouseout", function(e) {
  const icon = e.target.closest(".worker-icon");
  if (!icon) return;
  if (icon.contains(e.relatedTarget)) return;
  clearHighlights();
});
