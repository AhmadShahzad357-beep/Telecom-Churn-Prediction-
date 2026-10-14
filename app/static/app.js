/* Churn Radar - dashboard front end (plain JS, no framework, charts drawn as SVG) */
"use strict";

const TIER_COLORS = { "Low": "#1baf7a", "Medium": "#eda100", "High": "#eb6834", "Very high": "#c2334d" };
const CHURN = "#eb6834", STAY = "#2a78d6", INK = "#101828", MUTED = "#667085", LINE = "#d0d5dd";
const state = { dash: null, current: null, changes: {}, page: 1, batchRows: [] };

/* ---------------- helpers ---------------- */
const $ = (s, root = document) => root.querySelector(s);
const $$ = (s, root = document) => [...root.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (x, d = 1) => (x == null ? "\u2013" : (x * 100).toFixed(d) + "%");
const num = (x, d = 0) => (x == null ? "\u2013" : Number(x).toLocaleString("en-US", { maximumFractionDigits: d, minimumFractionDigits: d }));
const rs = (x) => "Rs " + num(x);
const rsShort = (x) => (x >= 1e5 ? "Rs " + (x / 1000).toFixed(0) + "k" : rs(x));
const tierBadge = (t) => `<span class="badge tier-${esc(t).replace(" ", "-")}">${esc(t)}</span>`;
const debounce = (fn, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  let body = null;
  try { body = await res.json(); } catch (_) { /* not json */ }
  if (!res.ok) throw new Error((body && body.detail) || `Request failed (${res.status})`);
  return body;
}

function svg(w, h, inner, label = "") {
  return `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(label)}">${inner}</svg>`;
}
function txt(x, y, s, o = {}) {
  const a = Object.entries({ "font-size": 12, fill: INK, ...o }).map(([k, v]) => `${k}="${v}"`).join(" ");
  return `<text x="${x}" y="${y}" ${a}>${esc(s)}</text>`;
}

/* tooltip for any element with data-tip */
const tip = $("#tooltip");
document.addEventListener("mousemove", (e) => {
  const el = e.target.closest("[data-tip]");
  if (!el) { tip.hidden = true; return; }
  tip.innerHTML = el.getAttribute("data-tip");
  tip.hidden = false;
  const x = Math.min(e.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
  tip.style.left = x + "px";
  tip.style.top = (e.clientY + 14) + "px";
});

/* ---------------- routing ---------------- */
function show(view) {
  const valid = ["overview", "checker", "customers", "batch", "model"];
  if (!valid.includes(view)) view = "overview";
  $$(".view").forEach((v) => v.classList.toggle("active", v.id === "view-" + view));
  $$(".nav a").forEach((a) => a.classList.toggle("active", a.dataset.view === view));
  if (view === "customers") loadCustomers();
  redraw();
  window.scrollTo({ top: 0 });
}
window.addEventListener("hashchange", () => {
  const [view, id] = location.hash.slice(1).split("/");
  show(view);
  if (view === "checker" && id) loadCustomer(id);
});

/* ---------------- charts ---------------- */
/* Charts are drawn at the container's real pixel width, so text stays readable on phones. */
const drawn = new Map();
function draw(sel, fn) {
  const el = $(sel);
  drawn.set(sel, fn);
  el.innerHTML = fn(Math.max(300, Math.round(el.clientWidth || 600)));
}
function redraw() {
  drawn.forEach((fn, sel) => { const el = $(sel); if (el.offsetParent) el.innerHTML = fn(Math.max(300, Math.round(el.clientWidth))); });
}
window.addEventListener("resize", debounce(redraw, 150));
const clip = (s, n) => (s.length > n ? s.slice(0, n - 1) + "\u2026" : s);
function kpi(label, value, sub, color) {
  return `<div class="kpi" style="--kpi-c:${color}"><div class="k">${esc(label)}</div><div class="v">${value}</div><div class="s">${sub}</div></div>`;
}

function tierChart(tiers, W = 560) {
  const rowH = 58, H = tiers.length * rowH;
  const totalC = tiers.reduce((a, t) => a + t.customers, 0);
  const totalR = tiers.reduce((a, t) => a + t.revenue_at_risk_rs, 0);
  const maxR = Math.max(...tiers.map((t) => t.revenue_at_risk_rs));
  const x0 = 96, bw = W - x0 - 80;
  let g = "";
  [...tiers].reverse().forEach((t, i) => {
    const y = i * rowH;
    const w = Math.max(2, (t.revenue_at_risk_rs / maxR) * bw);
    const c = TIER_COLORS[t.tier];
    const tipText = `<b>${t.tier}</b><br>${num(t.customers)} customers (${pct(t.customers / totalC)})<br>Actually churned: ${pct(t.actual_churn_rate)}<br>Revenue at risk: ${rs(t.revenue_at_risk_rs)} / month`;
    g += `<g data-tip="${esc(tipText)}">`;
    g += `<rect x="0" y="${y + 4}" width="${W}" height="${rowH - 8}" fill="transparent"/>`;
    g += txt(0, y + 24, t.tier, { "font-weight": 600, "font-size": 13 });
    g += txt(0, y + 41, `${num(t.customers)} cust.`, { fill: MUTED, "font-size": 11 });
    g += `<rect x="${x0}" y="${y + 12}" width="${w}" height="22" rx="5" fill="${c}"/>`;
    g += txt(x0 + w + 8, y + 28, rsShort(t.revenue_at_risk_rs), { "font-weight": 600 });
    g += txt(x0, y + 50, W < 420 ? `${pct(t.revenue_at_risk_rs / totalR, 0)} of risk \u00b7 ${pct(t.actual_churn_rate, 0)} churned`
      : `${pct(t.revenue_at_risk_rs / totalR, 0)} of revenue at risk \u00b7 ${pct(t.actual_churn_rate)} actually churned`, { fill: MUTED, "font-size": 11 });
    g += `</g>`;
  });
  return svg(W, H, g, "Revenue at risk by risk tier");
}

function gainsChart(points, metrics, W = 560) {
  const H = Math.round(Math.min(320, Math.max(240, W * 0.55))), L = 44, R = 16, T = 30, B = 40;
  const X = (v) => L + v * (W - L - R), Y = (v) => T + (1 - v) * (H - T - B);
  let g = `<line x1="${X(0)}" y1="${Y(0)}" x2="${X(1)}" y2="${Y(0)}" stroke="${LINE}"/>`;
  g += `<line x1="${X(0)}" y1="${Y(0)}" x2="${X(0)}" y2="${Y(1)}" stroke="${LINE}"/>`;
  [0, .25, .5, .75, 1].forEach((v) => {
    g += txt(X(v), H - B + 18, pct(v, 0), { "text-anchor": "middle", fill: MUTED, "font-size": 11 });
    g += txt(L - 8, Y(v) + 4, pct(v, 0), { "text-anchor": "end", fill: MUTED, "font-size": 11 });
  });
  g += txt(X(.5), H - 4, "Share of customers contacted (highest risk first)", { "text-anchor": "middle", fill: MUTED, "font-size": 11 });
  g += `<line x1="${X(0)}" y1="${Y(0)}" x2="${X(1)}" y2="${Y(1)}" stroke="#98a2b3" stroke-dasharray="4 4"/>`;
  const area = points.map((p) => `${X(p.contacted)},${Y(p.captured)}`).join(" ");
  g += `<polygon points="${X(0)},${Y(0)} ${area} ${X(1)},${Y(0)}" fill="${STAY}" opacity=".08"/>`;
  g += `<polyline points="${area}" fill="none" stroke="${STAY}" stroke-width="2.5" stroke-linejoin="round"/>`;
  [[.1, metrics.top10_capture], [.2, metrics.top20_capture]].forEach(([c, v]) => {
    g += `<line x1="${X(c)}" y1="${Y(0)}" x2="${X(c)}" y2="${Y(v)}" stroke="${CHURN}" stroke-dasharray="3 3"/>`;
    g += `<circle cx="${X(c)}" cy="${Y(v)}" r="5" fill="${CHURN}" stroke="#fff" stroke-width="2"/>`;
    g += txt(X(c) + 9, Y(v) + 16, `top ${pct(c, 0)} \u2192 ${pct(v, 0)} of churners`, { "font-size": 11.5, "font-weight": 600, fill: INK });
  });
  g += txt(X(.62), Y(.5), "random calling", { fill: "#98a2b3", "font-size": 11, transform: `rotate(-31 ${X(.62)} ${Y(.5)})` });
  g += txt(0, 12, "Churners caught", { fill: MUTED, "font-size": 11 });
  return svg(W, H, g, "Gains curve");
}

function hbar(items, opts = {}) {
  const W = opts.width || 640, rowH = opts.rowH || 26, labelW = Math.min(opts.labelW || 250, Math.round(W * 0.5));
  const H = items.length * rowH + 4;
  const max = Math.max(...items.map((d) => d.value));
  const bw = W - labelW - 60;
  let g = "";
  items.forEach((d, i) => {
    const y = i * rowH;
    const w = Math.max(2, (d.value / max) * bw);
    g += `<g data-tip="${esc(d.tip || "")}">`;
    g += `<rect x="0" y="${y}" width="${W}" height="${rowH}" fill="transparent"/>`;
    g += txt(labelW - 10, y + rowH / 2 + 4, clip(d.label, Math.floor(labelW / 6.4)), { "text-anchor": "end", "font-size": 12, fill: "#344054" });
    g += `<rect x="${labelW}" y="${y + 5}" width="${w}" height="${rowH - 10}" rx="4" fill="${d.color}"/>`;
    g += txt(labelW + w + 6, y + rowH / 2 + 4, d.text, { "font-size": 11.5, fill: MUTED });
    g += `</g>`;
  });
  return svg(W, H, g, opts.label || "bar chart");
}

function gauge(p, color) {
  const W = 260, H = 158, cx = 130, cy = 130, r = 100, sw = 20;
  const pt = (a) => [cx + r * Math.cos(Math.PI * (1 - a)), cy - r * Math.sin(Math.PI * (1 - a))];
  const arc = (a0, a1, c) => {
    const [x0, y0] = pt(a0), [x1, y1] = pt(a1);
    return `<path d="M${x0},${y0} A${r},${r} 0 0 1 ${x1},${y1}" fill="none" stroke="${c}" stroke-width="${sw}" stroke-linecap="butt"/>`;
  };
  let g = arc(0, 1, "#eef1f5");
  // tier bands as a thin outer ring
  const bands = [[0, .1, "Low"], [.1, .3, "Medium"], [.3, .6, "High"], [.6, 1, "Very high"]];
  bands.forEach(([a, b, t]) => {
    const rr = r + 16, p0 = [cx + rr * Math.cos(Math.PI * (1 - a)), cy - rr * Math.sin(Math.PI * (1 - a))],
      p1 = [cx + rr * Math.cos(Math.PI * (1 - b)), cy - rr * Math.sin(Math.PI * (1 - b))];
    g += `<path d="M${p0[0]},${p0[1]} A${rr},${rr} 0 0 1 ${p1[0]},${p1[1]}" fill="none" stroke="${TIER_COLORS[t]}" stroke-width="4" opacity=".85"/>`;
  });
  if (p > 0.002) g += arc(0, Math.min(p, 0.999), color);
  const [nx, ny] = pt(Math.min(Math.max(p, 0), 1));
  g += `<circle cx="${nx}" cy="${ny}" r="7" fill="#fff" stroke="${color}" stroke-width="3"/>`;
  g += txt(cx, cy - 18, pct(p, 1), { "text-anchor": "middle", "font-size": 36, "font-weight": 700 });
  g += txt(cx, cy + 4, "churn probability", { "text-anchor": "middle", "font-size": 12, fill: MUTED });
  return svg(W, H, g, "Churn probability gauge");
}

function whyChart(raising, lowering, W = 680) {
  const rows = [...raising.map((d) => ({ ...d, dir: 1 })), ...lowering.map((d) => ({ ...d, dir: -1 }))];
  if (!rows.length) return `<p class="hint">No strong drivers for this customer.</p>`;
  const rowH = 30, labelW = Math.round(W * 0.48), H = rows.length * rowH + 8;
  const max = Math.max(...rows.map((d) => Math.abs(d.shap)));
  const mid = labelW + (W - labelW) / 2, half = (W - labelW) / 2 - 46;
  let g = `<line x1="${mid}" y1="0" x2="${mid}" y2="${H - 4}" stroke="${LINE}"/>`;
  rows.forEach((d, i) => {
    const y = i * rowH + 2, w = Math.max(2, (Math.abs(d.shap) / max) * half);
    const c = d.dir > 0 ? CHURN : STAY;
    const x = d.dir > 0 ? mid : mid - w;
    if (i === raising.length && raising.length) g += `<line x1="0" y1="${y - 1}" x2="${W}" y2="${y - 1}" stroke="#eef1f5"/>`;
    const tipText = `<b>${esc(d.label)}</b><br>${esc(d.group)}<br>SHAP ${d.shap > 0 ? "+" : ""}${d.shap.toFixed(3)} log-odds`;
    g += `<g data-tip="${esc(tipText)}"><rect x="0" y="${y}" width="${W}" height="${rowH}" fill="transparent"/>`;
    g += txt(labelW - 12, y + rowH / 2 + 4, clip(d.text, Math.floor(labelW / 6.3)), { "text-anchor": "end", "font-size": 12, fill: "#344054" });
    g += `<rect x="${x}" y="${y + 7}" width="${w}" height="${rowH - 14}" rx="3" fill="${c}"/>`;
    const lx = d.dir > 0 ? mid + w + 6 : mid - w - 6;
    g += txt(lx, y + rowH / 2 + 4, (d.shap > 0 ? "+" : "") + d.shap.toFixed(2), { "text-anchor": d.dir > 0 ? "start" : "end", "font-size": 11, fill: MUTED });
    g += `</g>`;
  });
  return svg(W, H, g, "Reasons for this score");
}

/* ---------------- overview ---------------- */
function renderOverview(d) {
  const m = d.metrics, dec = m.at_decision_threshold;
  $("#kpis").innerHTML = [
    kpi("Customers scored", num(d.n_customers), `held-out test set \u00b7 ${pct(d.data.churn_rate)} churn rate`, "#6250d6"),
    kpi("On retention list", num(d.flagged_customers), `${pct(d.flagged_customers / d.n_customers)} of customers contacted`, CHURN),
    kpi("Churners caught", pct(dec.recall, 0), `${num(dec.tp)} of ${num(d.actual_churners)} real churners`, "#1baf7a"),
    kpi("Revenue at risk", rsShort(d.revenue_at_risk_total), "per month \u00b7 probability \u00d7 monthly value", "#c2334d"),
    kpi("Ranking quality", m.roc_auc.toFixed(3), `ROC-AUC \u00b7 PR-AUC ${m.pr_auc.toFixed(2)} (random ${m.random_pr_auc.toFixed(2)})`, STAY),
  ].join("");
  draw("#tier-chart", (w) => tierChart(d.tiers, w));
  draw("#gains-chart", (w) => gainsChart(d.gains, m, w));

  const groups = [...new Set(d.drivers.map((x) => x.group))];
  $("#drivers-legend").innerHTML = groups.map((g) => `<span><i style="background:${d.group_colors[g]}"></i>${esc(g)}</span>`).join("");
  const driverItems = d.drivers.map((x) => ({
    label: x.label, value: x.value, color: d.group_colors[x.group], text: x.value.toFixed(3),
    tip: `<b>${esc(x.label)}</b><br>${esc(x.group)}<br>${esc(x.effect)}`,
  }));
  draw("#drivers-chart", (w) => hbar(driverItems, { label: "Top churn drivers", labelW: 270, width: w }));
  const groupItems = d.group_share.map((x) => ({
    label: x.group, value: x.share, color: x.color, text: pct(x.share), tip: `<b>${esc(x.group)}</b><br>${pct(x.share)} of total importance`,
  }));
  draw("#group-chart", (w) => hbar(groupItems, { label: "Driver groups", labelW: 150, width: w, rowH: 30 }));
  const vh = d.tiers.find((t) => t.tier === "Very high"), totalR = d.tiers.reduce((a, t) => a + t.revenue_at_risk_rs, 0);
  $("#overview-insight").innerHTML = `<strong>Reading it:</strong> the strongest early-warning signal is a customer who stops recharging in August.
    The <b>Very high</b> tier holds ${pct(vh.customers / d.n_customers)} of customers but ${pct(vh.revenue_at_risk_rs / totalR, 0)} of the revenue at risk,
    and ${pct(vh.actual_churn_rate, 0)} of them really churned.`;
}

/* ---------------- checker ---------------- */
const ACTIONS = {
  "Recharge recency": "No recent recharge. Send a recharge reminder with a small top-up bonus within the next few days.",
  "Recharge & revenue": "Recharge spend has dropped. Offer a value pack priced near their usual recharge.",
  "Calls": "Call usage has collapsed, which often means a second SIM. Offer a calling bundle or on-net minutes.",
  "Mobile data": "Data use or data recharges dropped. Offer a discounted data pack trial.",
  "Roaming": "Heavy roaming, possibly relocating. Offer a roaming or national pack.",
  "Tenure": "Fairly new customer. Use an onboarding / loyalty welcome offer.",
  "Other": "Review the account manually.",
};

async function loadCustomer(id) {
  const err = $("#search-error");
  err.hidden = true;
  id = String(id).trim();
  if (!/^\d+$/.test(id)) { err.textContent = "Please enter a numeric customer ID."; err.hidden = false; return; }
  try {
    const d = await api(`/api/customer/${id}`);
    state.current = d;
    state.changes = {};
    $("#search-id").value = id;
    if (location.hash !== `#checker/${id}`) history.replaceState(null, "", `#checker/${id}`);
    renderCustomer(d);
  } catch (e) {
    err.textContent = e.message;
    err.hidden = false;
  }
}

function renderCustomer(d) {
  const r = d.result, p = d.profile;
  $("#checker-empty").hidden = true;
  $("#checker-body").hidden = false;
  $("#c-id").textContent = d.customer_id;
  $("#gauge").innerHTML = gauge(r.churn_probability, TIER_COLORS[r.risk_tier]);
  $("#c-tier").outerHTML = `<span id="c-tier">${tierBadge(r.risk_tier)}</span>`;
  $("#c-flag").innerHTML = r.flagged ? `<span class="pill pill-yes">Contact</span>` : `<span class="pill pill-no">Not needed</span>`;
  $("#c-value").textContent = rs(r.monthly_value_rs);
  $("#c-rar").textContent = rs(r.revenue_at_risk_rs);
  $("#c-tenure").textContent = p.tenure_years.toFixed(1) + " years";
  $("#c-actual").innerHTML = d.actual_churn ? `<span class="pill pill-yes">Churned</span>` : `<span class="pill pill-no">Stayed</span>`;
  draw("#why-chart", (w) => whyChart(r.raising, r.lowering, w));
  const top = r.raising[0];
  $("#action").innerHTML = (r.flagged || ["High", "Very high"].includes(r.risk_tier)) && top
    ? `<strong>Suggested action:</strong> ${esc(ACTIONS[top.group] || ACTIONS.Other)}`
    : `<strong>Suggested action:</strong> low risk, no retention spend needed. Keep normal communication.`;
  if (!r.high_value) $("#action").innerHTML += `<br><span class="pill pill-warn">Outside model scope</span> average Jun\u2013Jul spend is below the high-value cut-off the model was trained on.`;
  buildControls(p);
  renderWhatIf(d);
}

function ctrlRange(key, label, value, max, step, base, unit) {
  return `<div class="ctrl" data-key="${key}">
    <label for="ctl-${key}">${label}<span class="val" id="val-${key}">${unit(value)}</span></label>
    <input type="range" id="ctl-${key}" min="0" max="${max}" step="${step}" value="${value}">
    <div class="base">Actual Aug: ${unit(value)} \u00b7 Jun\u2013Jul avg: ${unit(base)}</div></div>`;
}
function ctrlDay(key, label, value, disabled, note) {
  const opts = [`<option value="none">No ${key === "recharge_day" ? "" : "data "}recharge in August</option>`]
    .concat(Array.from({ length: 31 }, (_, i) => `<option value="${i + 1}">Day ${i + 1} (${31 - i - 1} days before month end)</option>`));
  return `<div class="ctrl" data-key="${key}">
    <label for="ctl-${key}">${label}</label>
    <select class="input" id="ctl-${key}" ${disabled ? "disabled" : ""}>${opts.join("")}</select>
    <div class="base">${note}</div></div>`;
}

function buildControls(p) {
  const mins = (v) => num(v) + " min", mb = (v) => num(v) + " MB", money = (v) => rs(v);
  const cap = (a, b, floor) => Math.ceil(Math.max(a * 2, b * 1.2, floor) / 10) * 10;
  $("#controls").innerHTML = [
    ctrlDay("recharge_day", "Last recharge in August", p.recharge_day, false,
      p.recharge_day ? `Actual: day ${p.recharge_day}` : "Actual: no recharge in August"),
    ctrlRange("recharge_amount", "August recharge amount", p.recharge_amount, cap(p.recharge_amount_avg, p.recharge_amount, 2000), 10, p.recharge_amount_avg, money),
    ctrlRange("incoming_min", "August incoming minutes", p.incoming_min, cap(p.incoming_min_avg, p.incoming_min, 600), 5, p.incoming_min_avg, mins),
    ctrlRange("outgoing_min", "August outgoing minutes", p.outgoing_min, cap(p.outgoing_min_avg, p.outgoing_min, 600), 5, p.outgoing_min_avg, mins),
    ctrlRange("data_mb", "August mobile data", p.data_mb, cap(p.data_mb_avg, p.data_mb, 1000), 10, p.data_mb_avg, mb),
    ctrlDay("data_recharge_day", "Last data recharge in August", p.data_recharge_day, !p.data_recharge_possible,
      p.data_recharge_possible ? (p.data_recharge_day ? `Actual: day ${p.data_recharge_day}` : "Actual: no data recharge in August")
        : "Never recharged data, so nothing to simulate"),
  ].join("");
  $("#ctl-recharge_day").value = p.recharge_day ?? "none";
  $("#ctl-data_recharge_day").value = p.data_recharge_day ?? "none";
  $("#ctl-recharge_amount").disabled = p.recharge_day == null;

  const fmt = { recharge_amount: money, incoming_min: mins, outgoing_min: mins, data_mb: mb };
  $$("#controls input[type=range]").forEach((el) => {
    const key = el.id.replace("ctl-", "");
    el.addEventListener("input", () => {
      $("#val-" + key).textContent = fmt[key](+el.value);
      state.changes[key] = +el.value;
      el.closest(".ctrl").classList.add("changed");
      runWhatIf();
    });
  });
  $$("#controls select").forEach((el) => {
    const key = el.id.replace("ctl-", "");
    el.addEventListener("change", () => {
      const v = el.value === "none" ? null : +el.value;
      state.changes[key] = v;
      el.closest(".ctrl").classList.add("changed");
      if (key === "recharge_day") {
        const amt = $("#ctl-recharge_amount");
        amt.disabled = v == null;
        if (v != null && p.recharge_day == null && !("recharge_amount" in state.changes)) {
          const guess = Math.round((p.recharge_amount_avg || 200) / 10) * 10;
          amt.value = guess;
          $("#val-recharge_amount").textContent = money(guess);
          state.changes.recharge_amount = guess;
        }
        if (v == null) delete state.changes.recharge_amount;
      }
      runWhatIf();
    });
  });
}

const runWhatIf = debounce(async () => {
  const d = state.current;
  if (!d) return;
  $("#w-error").hidden = true;
  if (!Object.keys(state.changes).length) { renderWhatIf(d); return; }
  try {
    const w = await api("/api/what-if", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ customer_id: d.customer_id, changes: state.changes }),
    });
    renderWhatIf(d, w);
  } catch (e) {
    $("#w-error").textContent = e.message;
    $("#w-error").hidden = false;
  }
}, 220);

function renderWhatIf(d, w) {
  const a = d.result, b = (w || d).result;
  $("#w-orig").textContent = pct(a.churn_probability);
  $("#w-orig").style.color = TIER_COLORS[a.risk_tier];
  $("#w-orig-tier").innerHTML = tierBadge(a.risk_tier);
  $("#w-new").textContent = pct(b.churn_probability);
  $("#w-new").style.color = TIER_COLORS[b.risk_tier];
  $("#w-new-tier").innerHTML = tierBadge(b.risk_tier);
  const diff = b.churn_probability - a.churn_probability;
  const del = $("#w-delta");
  if (!w || Math.abs(diff) < 0.0005) {
    del.className = "delta"; del.textContent = w ? "No change in risk" : "Move a control to simulate";
  } else {
    del.className = "delta " + (diff < 0 ? "down" : "up");
    del.textContent = `${diff < 0 ? "\u25bc" : "\u25b2"} ${Math.abs(diff * 100).toFixed(1)} percentage points \u00b7 revenue at risk ${rs(b.revenue_at_risk_rs)}`;
  }
  const list = b.raising.slice(0, 3).map((x) => `<li>${esc(x.text)}</li>`).join("");
  $("#w-reasons").innerHTML = list ? `<b>Top reasons ${w ? "in the simulation" : "now"}:</b><ol>${list}</ol>` : "<b>No risk-raising reasons left.</b>";
}

$("#whatif-reset").addEventListener("click", () => {
  if (!state.current) return;
  state.changes = {};
  buildControls(state.current.profile);
  renderWhatIf(state.current);
});
$("#search-form").addEventListener("submit", (e) => { e.preventDefault(); loadCustomer($("#search-id").value); });
$$("[data-random]").forEach((b) => b.addEventListener("click", async () => {
  try {
    const r = await api(`/api/random-customer?tier=${encodeURIComponent(b.dataset.random)}`);
    loadCustomer(r.customer_id);
  } catch (e) { $("#search-error").textContent = e.message; $("#search-error").hidden = false; }
}));

/* ---------------- customers table ---------------- */
function reasonsCell(r) {
  return `<div class="reasons">${[r.reason_1, r.reason_2, r.reason_3].filter(Boolean).map((s) => `<div>${esc(s)}</div>`).join("") || "<span class='hint'>none</span>"}</div>`;
}
async function loadCustomers() {
  const q = $("#f-q").value, tier = $("#f-tier").value, sort = $("#f-sort").value, flagged = $("#f-flagged").checked;
  const qs = new URLSearchParams({ q, tier, sort, flagged, page: state.page, size: 15 });
  const d = await api(`/api/customers?${qs}`);
  const pages = Math.max(1, Math.ceil(d.total / d.size));
  $("#f-count").textContent = `${num(d.total)} customers`;
  $("#p-info").textContent = `Page ${d.page} of ${pages}`;
  $("#p-prev").disabled = d.page <= 1;
  $("#p-next").disabled = d.page >= pages;
  $("#cust-table tbody").innerHTML = d.rows.map((r) => `<tr data-id="${r.customer_id}">
      <td class="mono">${r.customer_id}</td><td>${tierBadge(r.risk_tier)}</td>
      <td class="num">${pct(r.churn_probability)}</td><td class="num">${rs(r.monthly_value_rs)}</td>
      <td class="num"><b>${rs(r.revenue_at_risk_rs)}</b></td><td>${reasonsCell(r)}</td>
      <td>${r.actual_churn ? '<span class="pill pill-yes">Churned</span>' : '<span class="pill pill-no">Stayed</span>'}</td></tr>`).join("")
    || `<tr><td colspan="7" class="hint" style="text-align:center;padding:30px">No customers match these filters.</td></tr>`;
}
const reloadCustomers = debounce(() => { state.page = 1; loadCustomers(); }, 250);
["#f-q", "#f-tier", "#f-sort", "#f-flagged"].forEach((s) => $(s).addEventListener(s === "#f-q" ? "input" : "change", reloadCustomers));
$("#p-prev").addEventListener("click", () => { state.page--; loadCustomers(); });
$("#p-next").addEventListener("click", () => { state.page++; loadCustomers(); });
$("#cust-table tbody").addEventListener("click", (e) => {
  const tr = e.target.closest("tr[data-id]");
  if (tr) location.hash = `#checker/${tr.dataset.id}`;
});

/* ---------------- batch scoring ---------------- */
const drop = $("#drop");
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => { if (e.dataTransfer.files[0]) scoreFile(e.dataTransfer.files[0]); });
$("#file").addEventListener("change", (e) => { if (e.target.files[0]) scoreFile(e.target.files[0]); e.target.value = ""; });

async function scoreFile(file) {
  const err = $("#batch-error");
  err.hidden = true;
  $("#batch-result").hidden = true;
  drop.querySelector("strong").textContent = `Scoring ${file.name}\u2026`;
  const fd = new FormData();
  fd.append("file", file);
  try {
    const d = await api("/api/score-file", { method: "POST", body: fd });
    state.batchRows = d.rows;
    renderBatch(d, file.name);
  } catch (e) {
    err.textContent = e.message;
    err.hidden = false;
  } finally {
    drop.querySelector("strong").textContent = "Drop a CSV here";
  }
}

function renderBatch(d, name) {
  const s = d.summary;
  $("#batch-kpis").innerHTML = [
    kpi("Customers scored", num(s.customers), esc(name), "#6250d6"),
    kpi("On retention list", num(s.flagged), `${pct(s.flagged / s.customers)} of the file`, CHURN),
    kpi("Very high risk", num(s.tiers["Very high"] || 0), `High: ${num(s.tiers["High"] || 0)} \u00b7 Medium: ${num(s.tiers["Medium"] || 0)}`, TIER_COLORS["Very high"]),
    kpi("Revenue at risk", rsShort(s.revenue_at_risk_rs), "per month", "#c2334d"),
    kpi("Outside scope", num(s.outside_scope), "below the high-value spend cut-off", "#8a8984"),
  ].join("");
  const shown = d.rows.slice(0, 300);
  $("#batch-table tbody").innerHTML = shown.map((r) => `<tr>
      <td class="mono">${esc(r.customer_id)}</td><td>${tierBadge(r.risk_tier)}</td>
      <td class="num">${pct(r.churn_probability)}</td><td class="num">${rs(r.monthly_value_rs)}</td>
      <td class="num"><b>${rs(r.revenue_at_risk_rs)}</b></td><td>${reasonsCell(r)}</td>
      <td>${r.high_value ? '<span class="pill pill-no">In scope</span>' : '<span class="pill pill-warn">Below cut-off</span>'}</td></tr>`).join("");
  $("#batch-note").textContent = d.rows.length > 300
    ? `Showing the 300 rows with the highest revenue at risk. Download the CSV for all ${num(d.rows.length)}.`
    : "Sorted by revenue at risk. Download the CSV to load it into your CRM or dialler.";
  $("#batch-result").hidden = false;
}

$("#batch-download").addEventListener("click", () => {
  const cols = ["customer_id", "churn_probability", "risk_tier", "flagged", "monthly_value_rs", "revenue_at_risk_rs", "high_value", "reason_1", "reason_2", "reason_3"];
  const cell = (v) => (typeof v === "string" && /[",\n]/.test(v) ? `"${v.replace(/"/g, '""')}"` : v ?? "");
  const csv = [cols.join(",")].concat(state.batchRows.map((r) => cols.map((c) => cell(r[c])).join(","))).join("\n");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
  a.download = "churn_scores.csv";
  a.click();
  URL.revokeObjectURL(a.href);
});

/* ---------------- model page ---------------- */
function renderModel(d) {
  const m = d.metrics, dec = d.decision, ds = d.data;
  $("#model-how").innerHTML = `
    <h2>How a score is made</h2>
    <p><b>Customers:</b> prepaid, high-value only (top 30% by average June\u2013July spend, at least Rs ${num(ds.high_value_threshold_avg_spend_jun_jul)} a month): ${num(ds.high_value_customers)} of ${num(ds.raw_rows)}.</p>
    <p><b>Churn means:</b> in September the customer made no recharge (calls or data) <i>and</i> used nothing at all (no calls in or out, no 2G/3G data). ${pct(ds.churn_rate)} of high-value customers churned.</p>
    <h3>Pipeline</h3>
    <ul>
      <li>Split first: ${num(ds.train_rows)} train / ${num(ds.test_rows)} test, stratified. All September columns removed, so the model only sees June\u2013August.</li>
      <li>Missing values filled by meaning (no usage \u2192 0, no data recharge \u2192 0), never by averages.</li>
      <li>Features compare each customer's August to their own June\u2013July baseline, plus days since last recharge and behaviour flags.</li>
      <li>${esc(d.model.best_family)} with ${esc(d.model.imbalance_strategy)}, tuned with cross-validation on train only (CV PR-AUC ${d.model.final_cv_pr_auc.toFixed(3)}).</li>
      <li>Isotonic calibration so "70%" means about 70%. Brier score ${m.brier_before.toFixed(3)} \u2192 ${m.brier_after.toFixed(3)}.</li>
      <li>Retention threshold ${dec.threshold.toFixed(2)} chosen on train out-of-fold predictions for the best campaign value, not on the test set.</li>
      <li>Reasons are exact SHAP values of the model for that customer.</li>
    </ul>`;
  const rows = [["Default 0.50", m.at_default_0_5], [`Retention ${dec.threshold.toFixed(2)} (used)`, m.at_decision_threshold], [`Best F1 ${dec.threshold_best_f1.toFixed(2)}`, m.at_best_f1]];
  $("#metric-table").innerHTML = `
    <table class="mtable"><thead><tr><th>Threshold</th><th>Precision</th><th>Recall</th><th>F1</th><th>Accuracy</th><th>Flagged</th></tr></thead><tbody>
    ${rows.map(([n, r]) => `<tr class="${n.includes("used") ? "hl" : ""}"><td>${n}</td><td>${pct(r.precision)}</td><td>${pct(r.recall)}</td><td>${r.f1.toFixed(3)}</td><td>${pct(r.accuracy)}</td><td>${num(r.flagged)}</td></tr>`).join("")}
    </tbody></table>
    <p class="footnote">ROC-AUC ${m.roc_auc.toFixed(3)} (95% CI ${m.roc_auc_95ci[0]}\u2013${m.roc_auc_95ci[1]}) \u00b7 PR-AUC ${m.pr_auc.toFixed(3)} (95% CI ${m.pr_auc_95ci[0]}\u2013${m.pr_auc_95ci[1]}) vs ${m.random_pr_auc.toFixed(3)} for random guessing.
    Accuracy alone is misleading here: predicting "nobody churns" would already score ${pct(1 - m.random_pr_auc)}.</p>`;
  const c = m.at_decision_threshold;
  $("#cm-hint").textContent = `${num(m.n_test)} test customers`;
  $("#confusion").innerHTML = `<div class="cm">
    <div></div><div class="h">Predicted: churn</div><div class="h">Predicted: stay</div>
    <div class="h">Actually churned</div><div class="cell tp"><b>${num(c.tp)}</b><span>caught</span></div><div class="cell fn"><b>${num(c.fn)}</b><span>missed</span></div>
    <div class="h">Actually stayed</div><div class="cell fp"><b>${num(c.fp)}</b><span>extra calls</span></div><div class="cell tn"><b>${num(c.tn)}</b><span>correctly left alone</span></div></div>`;
  $("#model-limits").innerHTML = `
    <h2>Limits: read before using</h2>
    <ul>
      <li><b>Data:</b> public Kaggle telecom dataset (2014, Indian prepaid market). Behaviour today may differ, so retrain on the company's own recent data before real use.</li>
      <li><b>Scope:</b> trained only on high-value customers. Lower-spend customers can be scored, but they are marked <i>outside scope</i>.</li>
      <li><b>Money figures:</b> revenue at risk = calibrated probability \u00d7 June\u2013July average ARPU. Campaign value uses illustrative assumptions (offer cost Rs ${num(dec.offer_cost_rs)}, ${pct(dec.save_rate, 0)} of contacted churners saved, ${dec.value_months} months of revenue kept). Replace them with the company's real numbers.</li>
      <li><b>Precision:</b> at the retention threshold about ${pct(c.precision, 0)} of contacted customers are real churners. This is acceptable because a missed churner costs far more than one extra call.</li>
      <li><b>What-if:</b> shows how the model reacts to different August behaviour. It explains the model and is not a causal estimate.</li>
    </ul>`;
}

/* ---------------- start ---------------- */
(async function init() {
  try {
    state.dash = await api("/api/dashboard");
    renderOverview(state.dash);
    renderModel(state.dash);
  } catch (e) {
    $("#kpis").innerHTML = `<div class="error">Could not load the dashboard: ${esc(e.message)}</div>`;
  }
  const [view, id] = location.hash.slice(1).split("/");
  show(view || "overview");          // also redraws the charts at the real width
  if (view === "checker" && id) loadCustomer(id);
})();
