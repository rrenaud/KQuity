const TYPE_SYMBOL = {death: "\u2620", speed: "\u26a1", warrior: "\u2694"};
const tooltip = document.getElementById("transition-tooltip");

// Build lookup maps — store full transition objects
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
  // warrior: drone->warrior or speed_drone->speed_warrior
  return {warrior: true, speed: srcSpeed};
}

document.addEventListener("mouseover", function(e) {
  const icon = e.target.closest(".worker-icon");
  if (!icon) return;
  const key = icon.dataset.stateIdx + ":" + icon.dataset.charIdx;
  const trans = transMap[key];
  if (!trans) return;
  clearHighlights();

  const srcRow = rowByState[icon.dataset.stateIdx];
  if (srcRow) srcRow.classList.add("highlight-source");

  const charType = CHAR_PRIORITY[trans.char];

  // Build tooltip
  let tooltipHtml = '';
  trans.actions.forEach(a => {
    const tgt = targetAppearance(a.type, trans.src_warrior, trans.src_speed);
    const srcIcon = renderMiniIcon(charType, trans.src_warrior, trans.src_speed);
    const tgtIcon = renderMiniIcon(charType, tgt.warrior, tgt.speed);
    const sign = a.win_delta >= 0 ? '+' : '';
    const cls = a.win_delta >= 0 ? 'positive' : 'negative';
    const arrowCls = "tooltip-arrow tooltip-arrow-" + a.type;
    tooltipHtml += '<div class="tooltip-row">'
      + srcIcon + '<span class="' + arrowCls + '">' + TYPE_SYMBOL[a.type] + '</span>' + tgtIcon
      + '<span class="tooltip-delta ' + cls + '">'
      + sign + a.win_delta.toFixed(1) + 'pp</span>'
      + '</div>';
  });
  tooltip.innerHTML = tooltipHtml;
  tooltip.style.display = 'block';

  // Position tooltip near the icon
  const rect = icon.getBoundingClientRect();
  tooltip.style.left = (rect.right + 24) + 'px';
  tooltip.style.top = rect.top + 'px';

  // Highlight target rows + add delta badges
  trans.actions.forEach(a => {
    const row = rowByState[a.target];
    if (!row) return;
    row.classList.add(a.type === "death" ? "highlight-death" : "highlight-upgrade");
    const winCell = row.querySelector(".win-pct");
    if (winCell) {
      const badge = document.createElement("span");
      badge.className = "delta-badge " + (a.win_delta >= 0 ? "positive" : "negative");
      badge.textContent = TYPE_SYMBOL[a.type] + " " + (a.win_delta >= 0 ? "+" : "") + a.win_delta.toFixed(1) + "pp";
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
