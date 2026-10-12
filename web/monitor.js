// The Monitor page: polls GET /metrics every 2 s and draws the same facts Strata's Monitor tab shows - the metric
// tiles with a minute of history (SVG sparklines, no library), the request running now, the engine's own numbers
// and the hardware. Series with no reading arrive as null and are simply skipped.
"use strict";
const $ = id => document.getElementById(id);

const bytes = v => {
  if (typeof v !== "number") return "—";
  const u = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
  return v.toFixed(i >= 2 ? 2 : 0) + " " + u[i];
};
const num = (v, d = 1) => typeof v === "number" ? v.toFixed(d) : "—";
const int = v => typeof v === "number" ? v.toLocaleString() : "—";
const secs = v => typeof v === "number" ? (v < 10 ? v.toFixed(2) : v.toFixed(1)) + " s" : "—";
const pct = v => typeof v === "number" ? v.toFixed(1) + "%" : "—";

// one sparkline per series, scaled to its own range (a fixed max for the 0-100 series)
function spark(values, max) {
  const v = (values || []).filter(x => typeof x === "number");
  if (v.length < 2) return "";
  const hi = max !== undefined ? max : Math.max(...v) * 1.1 || 1;
  const n = v.length;
  const y = x => 30 - Math.max(0, Math.min(1, x / hi)) * 28;
  const pts = v.map((x, i) => `${(i / (n - 1)) * 100},${y(x).toFixed(1)}`);
  return `<svg class="spark" viewBox="0 0 100 32" preserveAspectRatio="none" aria-hidden="true">
    <polygon class="fill" points="0,32 ${pts.join(" ")} 100,32"></polygon>
    <polyline points="${pts.join(" ")}"></polyline></svg>`;
}

function facts(el, rows) {
  el.replaceChildren();
  for (const [k, v] of rows) {
    const d = document.createElement("div");
    const a = document.createElement("span"); a.textContent = k;
    const b = document.createElement("b"); b.textContent = v;
    d.append(a, b); el.append(d);
  }
}

function tiles(hw, hs, h) {
  const t = (label, value, series, max, note) => {
    const c = document.createElement("div"); c.className = "card";
    c.innerHTML = `<div class="label">${label}</div><div class="value">${value}</div>` +
                  (note ? `<div class="note">${note}</div>` : "") + spark(h[series], max);
    return c;
  };
  const gpu = hs.gpu_name || "GPU";
  const out = [
    t("GPU 负载", typeof hw.gpu_util === "number" ? hw.gpu_util + "%" : "—", "gpu_util", 100, gpu),
    t("显存占用", bytes(hw.gpu_mem_used), "gpu_mem_used", hw.gpu_mem_total,
      typeof hw.gpu_mem_total === "number" ? ("共 " + bytes(hw.gpu_mem_total)) : ""),
    t("温度", typeof hw.gpu_temp === "number" ? hw.gpu_temp + " °C" : "—", "gpu_temp", 90, ""),
    t("功耗", typeof hw.gpu_power === "number" ? num(hw.gpu_power) + " W" : "—", "gpu_power", hw.gpu_power_limit,
      typeof hw.gpu_power_limit === "number" ? ("上限 " + num(hw.gpu_power_limit) + " W") : ""),
    t("CPU", typeof hw.cpu === "number" ? hw.cpu + "%" : "—", "cpu", 100,
      hs.cores ? `${hs.cores} 核 / ${hs.threads} 线程` : ""),
    t("内存", bytes(hw.ram_used), "ram_used", hw.ram_total,
      typeof hw.ram_total === "number" ? ("共 " + bytes(hw.ram_total)) : ""),
    t("磁盘读", bytes((hw.disk_read_mb || 0) * 1048576) + "/s", "disk_read_mb", undefined, ""),
    t("PCIe 读", typeof hw.gpu_pcie_rx_mb === "number" ? num(hw.gpu_pcie_rx_mb) + " MB/s" : "—", "gpu_pcie_rx_mb", undefined,
      (typeof hw.gpu_pcie_gen === "number" ? `Gen${hw.gpu_pcie_gen}` : "") +
      (typeof hw.gpu_pcie_width === "number" ? ` x${hw.gpu_pcie_width}` : "")),
  ];
  const el = $("tiles"); el.replaceChildren(); for (const c of out) el.append(c);
}

function render(m) {
  const lv = m.live || {}, e = m.engine || {}, hw = m.hardware || {}, hs = m.hardware_static || {},
        h = m.history || {}, t = m.totals || {};
  const state = lv.state || "idle";
  const names = { idle: "空闲", reading: "读取 prompt", generating: "生成中" };
  $("state").className = "value " + (state === "generating" ? "gen" : state === "reading" ? "read" : "idle");
  $("state").innerHTML = `<span class="dot"></span>${names[state] || state}`;
  $("state-note").textContent = m.service + " · " + (e.model || "—");
  $("speed").innerHTML = (state === "generating" ? num(lv.tok_s) : num(t.last_tok_s)) + "<small>tok/s</small>";
  $("speed-note").textContent = state === "generating" ? "本次生成" : "上一次生成";
  $("prefill").innerHTML = num(typeof lv.prefill_tok_s === "number" ? lv.prefill_tok_s : t.prefill_tok_s_mean) + "<small>tok/s</small>";
  $("prefill-note").textContent = "平均 " + num(t.prefill_tok_s_mean) + " tok/s · 解码均 " + num(t.tok_s_mean) + " tok/s";
  $("reqs").textContent = int(t.requests);
  $("reqs-note").textContent = `输入 ${int(t.prompt_tokens)} / 输出 ${int(t.output_tokens)} tokens`;

  $("live-id").textContent = lv.id ? lv.id : "";
  facts($("live"), [
    ["请求 id", lv.id || "—"],
    ["prompt tokens", lv.prompt_tokens === null || lv.prompt_tokens === undefined ? "—" : int(lv.prompt_tokens)],
    ["已生成", lv.generated === null || lv.generated === undefined ? "—" : int(lv.generated)],
    ["上限", lv.max_tokens === null || lv.max_tokens === undefined ? "—" : int(lv.max_tokens)],
    ["首 token", secs(lv.first_token_s)],
    ["已用时", secs(lv.elapsed_s)],
  ]);

  $("engine-model").textContent = e.model || "";
  facts($("engine"), [
    ["量化", e.quantization || "—"],
    ["上下文", e.context ? int(e.context) + (e.trained_context && e.trained_context !== e.context ? ` (训练 ${int(e.trained_context)})` : "") : "—"],
    ["层 / 每层专家", `${int(e.layers)} / ${int(e.experts_per_layer)}`],
    ["每 token 专家", int(e.experts_per_token)],
    ["VRAM 中的专家", `${int(e.experts_in_vram)} / ${int(e.experts_total)}`],
    ["专家缓存", e.expert_cache || "—"],
    ["常驻 pinned RAM", num(e.expert_pinned_gb) + " GB (" + (e.experts_in_ram || "—") + ")"],
    ["命中率", pct(e.expert_hit_rate_pct)],
    ["命中 / PCIe / CPU", `${int(e.expert_hits)} / ${int(e.expert_over_pcie)} / ${int(e.expert_on_cpu)}`],
    ["VRAM 空闲 / 总", num(e.vram_free_gb) + " / " + num(e.vram_total_gb) + " GB"],
    ["MTP 投机", e.mtp ? (int(e.mtp_rounds) + " 轮, 接受 " + pct(e.draft_accept_rate_pct)) : "关闭"],
  ]);

  $("hw-static").textContent = [hs.cpu_name, hs.gpu_note].filter(Boolean).join(" · ");
  tiles(hw, hs, h);

  facts($("totals"), [
    ["请求", int(t.requests)],
    ["prompt tokens", int(t.prompt_tokens)],
    ["生成 tokens", int(t.output_tokens)],
    ["复用 tokens", int(t.reused_tokens)],
    ["prefill 耗时", secs(t.prompt_s)],
    ["decode 耗时", secs(t.decode_s)],
    ["平均 prefill", num(t.prefill_tok_s) + " tok/s"],
    ["平均 decode", num(t.decode_tok_s) + " tok/s"],
    ["上次首 token", secs(t.last_first_token_s)],
  ]);
  $("updated").textContent = "更新于 " + new Date((m.time || 0) * 1000).toLocaleTimeString();
}

async function poll() {
  try {
    const r = await fetch("/metrics", { cache: "no-store" });
    if (!r.ok) throw new Error("HTTP " + r.status);
    render(await r.json());
    $("err").style.display = "none";
  } catch (e) {
    $("err").style.display = "block";
    $("err").textContent = "无法读取 /metrics：" + e.message;
  }
  setTimeout(poll, 2000);
}

$("theme").onclick = () => {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  localStorage.setItem("nano.theme", next);
};
document.documentElement.dataset.theme = localStorage.getItem("nano.theme") || "dark";
poll();
