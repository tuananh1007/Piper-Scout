// Piper-Scout operator page: talks to gui_server.py (JSON API + server-sent events).
"use strict";
const $ = (s) => document.querySelector(s);
const state = { snap: null, proposal: null, countdownEnd: 0, countdownLen: 2, pipeHist: [], driveTimer: null };

async function post(path, body) {
  try {
    const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" },
                                  body: JSON.stringify(body || {}) });
    return await r.json();
  } catch (e) { return { ok: false, error: String(e) }; }
}

// ---------------------------------------------------------------- stop
function stopNow() {
  post("api/abort");
  releaseDrive();
  $("#stopped").hidden = false;
  const de = $("#drive-enable");
  if (de.checked) { de.checked = false; }
}
$("#stop").addEventListener("click", stopNow);

// ---------------------------------------------------------------- heartbeat + safety supervisor
// The robot keeps moving only while this page is alive (when the safety
// supervisor requires it): four heartbeats a second.
setInterval(() => { fetch("api/heartbeat", { method: "POST" }).catch(() => {}); }, 250);
$("#safety-reset").addEventListener("click", async () => {
  const r = await post("api/safety_reset");
  logMsg(r.message || r.reason || (r.ok ? "Safety stop reset." : "Reset refused."));
});
function applySafety(sf) {
  const tripped = !!(sf && sf.ok === false);
  $("#safety").hidden = !tripped;
  if (tripped) { $("#safety-text").textContent = "Safety stop: " + (sf.tripped || "tripped"); }
}
document.addEventListener("keydown", (e) => {
  const typing = e.target && (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA");
  if (e.key === "Escape" || (e.key === " " && !typing)) { e.preventDefault(); stopNow(); }
});

// ---------------------------------------------------------------- tabs
document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll(".tabs button").forEach((x) => x.setAttribute("aria-selected", x === b ? "true" : "false"));
  document.querySelectorAll(".panel").forEach((p) => { p.hidden = p.id !== "tab-" + b.dataset.tab; });
  try { localStorage.setItem("tab", b.dataset.tab); } catch (e) { /* storage unavailable */ }
}));
try { const t = localStorage.getItem("tab"); if (t) { document.querySelector(`.tabs button[data-tab="${t}"]`)?.click(); } }
catch (e) { /* storage unavailable */ }

// ---------------------------------------------------------------- log + speech
function logMsg(text) {
  const li = document.createElement("li");
  const t = document.createElement("time");
  t.textContent = new Date().toLocaleTimeString();
  li.append(t, document.createTextNode(text));
  $("#log").prepend(li);
  while ($("#log").children.length > 100) { $("#log").lastChild.remove(); }
  if ($("#speak").checked) { speak(text); }
}
async function speak(text) {
  try {
    const r = await fetch("api/tts?text=" + encodeURIComponent(text));
    if (r.ok) { new Audio(URL.createObjectURL(await r.blob())).play(); return; }
  } catch (e) { /* fall through */ }
  if ("speechSynthesis" in window) { speechSynthesis.speak(new SpeechSynthesisUtterance(text)); }
}

// ---------------------------------------------------------------- instruction
$("#say-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = $("#instruction").value.trim();
  if (!text) { return; }
  logMsg("You: " + text);
  $("#instruction").value = "";
  await post("api/instruction", { text });
});
$("#confirm").addEventListener("click", () => post("api/confirm"));
$("#cancel").addEventListener("click", () => post("api/cancel"));

function showProposal(p) {
  state.proposal = p;
  $("#question").hidden = true;
  $("#proposal").hidden = false;
  const verb = { grasp: "Grasp", inspect: "Inspect", release: "Release", scan: "Scan" }[p.action] || p.action;
  $("#proposal-title").textContent = verb + (p.target ? " " + p.target.label : "");
  $("#proposal-msg").textContent = p.message || "";
  const adv = $("#proposal-adv");
  adv.replaceChildren(...(p.advisories || []).map((a) => { const li = document.createElement("li"); li.textContent = a; return li; }));
  state.countdownLen = p.countdown_s || 2;
  state.countdownEnd = performance.now() + state.countdownLen * 1000;
  drawOverlay();
}
function hideProposal() { state.proposal = null; $("#proposal").hidden = true; drawOverlay(); }
function showQuestion(q) {
  hideProposal();
  $("#question").hidden = false;
  $("#question-text").textContent = q.text;
  $("#question-opts").replaceChildren(...(q.options || []).map((o) => {
    const b = document.createElement("button");
    b.textContent = o;
    b.addEventListener("click", () => { logMsg("You: " + o); post("api/instruction", { text: o }); $("#question").hidden = true; });
    return b;
  }));
}
function tickCountdown() {
  if (state.proposal) {
    const left = Math.max(0, state.countdownEnd - performance.now()) / (state.countdownLen * 1000);
    $("#countdown-bar").style.transform = `scaleX(${left})`;
  }
  requestAnimationFrame(tickCountdown);
}
requestAnimationFrame(tickCountdown);

// ---------------------------------------------------------------- camera overlay
function drawOverlay() {
  const img = $("#stream"), cv = $("#overlay");
  const W = cv.clientWidth, H = cv.clientHeight;
  if (!W || !H) { return; }
  cv.width = W * devicePixelRatio; cv.height = H * devicePixelRatio;
  const g = cv.getContext("2d");
  g.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  g.clearRect(0, 0, W, H);
  const snap = state.snap;
  if (!snap || !snap.detections) { return; }
  const hw = snap.image_hw || [img.naturalHeight || 240, img.naturalWidth || 320];
  const s = Math.min(W / hw[1], H / hw[0]);
  const ox = (W - hw[1] * s) / 2, oy = (H - hw[0] * s) / 2;
  const chosen = state.proposal && state.proposal.target ? state.proposal.target.index : -1;
  g.font = "600 13px system-ui, sans-serif";
  for (const d of snap.detections) {
    const [x0, y0, x1, y1] = d.bbox;
    const sel = d.index === chosen;
    g.strokeStyle = sel ? "#ffd54f" : "rgba(255,255,255,0.85)";
    g.lineWidth = sel ? 3 : 1.5;
    g.strokeRect(ox + x0 * s, oy + y0 * s, (x1 - x0) * s, (y1 - y0) * s);
    const label = `${d.index + 1}`;
    g.fillStyle = sel ? "#ffd54f" : "rgba(0,0,0,0.6)";
    g.fillRect(ox + x0 * s, oy + y0 * s - 18, g.measureText(label).width + 10, 18);
    g.fillStyle = sel ? "#000" : "#fff";
    g.fillText(label, ox + x0 * s + 5, oy + y0 * s - 5);
  }
}
window.addEventListener("resize", drawOverlay);
$("#stream").addEventListener("load", drawOverlay);

function renderDets(dets) {
  const ul = $("#dets");
  const key = JSON.stringify(dets.map((d) => [d.index, d.class, d.colour]));
  if (ul.dataset.key === key) { return; }
  ul.dataset.key = key;
  ul.replaceChildren(...dets.map((d) => {
    const li = document.createElement("li");
    const n = document.createElement("span"); n.className = "num"; n.textContent = "#" + (d.index + 1);
    const nm = document.createElement("span"); nm.className = "name"; nm.textContent = `${d.colour || ""} ${d.class}`.trim();
    const g = document.createElement("button"); g.textContent = "Grasp";
    g.addEventListener("click", () => { logMsg(`You: grasp number ${d.index + 1}`); post("api/instruction", { text: `grasp number ${d.index + 1}` }); });
    const i = document.createElement("button"); i.textContent = "Inspect"; i.className = "ghost";
    i.addEventListener("click", () => { logMsg(`You: inspect number ${d.index + 1}`); post("api/instruction", { text: `inspect number ${d.index + 1}` }); });
    li.append(n, nm, g, i);
    return li;
  }));
}

// ---------------------------------------------------------------- state
function chip(id, text, cls) { const c = $(id); c.textContent = text; c.className = "chip " + (cls || ""); }
function applyState(s) {
  state.snap = s;
  chip("#chip-conn", s.connected === false ? "robot offline" : "connected", s.connected === false ? "bad" : "ok");
  const busy = ["VIEWING", "REACHING", "SERVOING", "APPROACHING", "GRASPING", "RELEASING", "RETREATING"].includes(s.pipeline);
  chip("#chip-pipe", "pipeline " + (s.pipeline || "–"), s.pipeline === "ABORTED" ? "bad" : busy ? "busy" : "");
  const mode = (s.mpc && s.mpc.mode) || "–";
  chip("#chip-mpc", "MPC " + mode, mode === "whole_body" ? "busy" : "");
  $("#w-pipe").textContent = s.pipeline || "–";
  if (s.pipeline && state.pipeHist[0] !== s.pipeline) {
    state.pipeHist.unshift(s.pipeline);
    state.pipeHist.length = Math.min(state.pipeHist.length, 12);
    $("#w-pipe-hist").replaceChildren(...state.pipeHist.slice(1).map((p) => { const li = document.createElement("li"); li.textContent = p; return li; }));
  }
  $("#w-mpc").textContent = mode;
  const m = s.mpc || {};
  $("#w-mpc-detail").textContent = m.tcp_error_m != null ? `TCP ${(100 * m.tcp_error_m).toFixed(1)} cm, axis ${Math.round(m.approach_error_deg || 0)}°` : "";
  const u = s.uncertainty || {};
  $("#u-jepa").value = u.u_jepa ?? 0; $("#u-geo").value = u.u_geometry ?? 0; $("#u-twin").value = u.u_twin ?? 0;
  $("#w-unmet").textContent = u.needs_view ? "Needs another view: " + (u.unmet || []).join(", ") : (u.u_jepa != null ? "Ready" : "");
  $("#w-force").textContent = s.force_n != null ? `${Number(s.force_n).toFixed(2)} N` : "–";
  const tw = s.twin || {};
  $("#w-twin").textContent = tw.confidence != null ? `plant twin confidence ${Math.round(100 * tw.confidence)} %` : "";
  if (s.drive_enabled != null && $("#drive-enable").checked !== s.drive_enabled && !state.driveTimer) {
    $("#drive-enable").checked = s.drive_enabled;
  }
  applySafety(s.safety);
  renderDets(s.detections || []);
  drawOverlay();
}

function connect() {
  const es = new EventSource("api/events");
  es.addEventListener("state", (e) => applyState(JSON.parse(e.data)));
  es.addEventListener("say", (e) => logMsg(JSON.parse(e.data).text));
  es.addEventListener("stop", () => { $("#stopped").hidden = false; });
  es.addEventListener("proposal", (e) => {
    const p = JSON.parse(e.data);
    if (p.type === "proposal") { showProposal(p); }
    else if (p.type === "question") { showQuestion(p); }
    else if (["commit", "cancelled", "rejected"].includes(p.type)) { hideProposal(); }
  });
  es.onerror = () => chip("#chip-conn", "reconnecting", "bad");
}
fetch("api/state").then((r) => r.json()).then(applyState).catch(() => {});
connect();

// ---------------------------------------------------------------- drive
$("#drive-enable").addEventListener("change", async (e) => {
  const r = await post("api/drive_enable", { enable: e.target.checked });
  if (r && r.enabled) { $("#stopped").hidden = true; }
  if (!r || !r.ok) { e.target.checked = false; logMsg("Drive: " + ((r && (r.reason || r.error)) || "not available")); }
});
$("#speed").addEventListener("input", (e) => { $("#speed-val").textContent = e.target.value + " %"; });
function holdDrive(btn, make) {
  const start = (e) => {
    e.preventDefault();
    if (!$("#drive-enable").checked) { logMsg("Enable manual drive first."); return; }
    releaseDrive();
    btn.classList.add("held");
    const send = () => post("api/drive", make(Number($("#speed").value) / 100));
    send();
    state.driveTimer = setInterval(send, 100);
    state.driveBtn = btn;
  };
  btn.addEventListener("pointerdown", start);
  ["pointerup", "pointerleave", "pointercancel"].forEach((ev) => btn.addEventListener(ev, releaseDrive));
}
function releaseDrive() {
  if (state.driveTimer) {
    clearInterval(state.driveTimer); state.driveTimer = null;
    if (state.driveBtn) { state.driveBtn.classList.remove("held"); }
    post("api/drive", { v: 0, w: 0, arm: [0, 0, 0, 0, 0, 0] });
  }
}
document.querySelectorAll("#base-pad button").forEach((b) =>
  holdDrive(b, (k) => ({ v: 0.2 * k * Number(b.dataset.v), w: 0.4 * k * Number(b.dataset.w) })));
document.querySelectorAll("#arm-pad button[data-arm]").forEach((b) =>
  holdDrive(b, (k) => ({ arm: b.dataset.arm.split(",").map((x) => 0.05 * k * Number(x)) })));
window.addEventListener("blur", releaseDrive);
document.addEventListener("visibilitychange", () => { if (document.hidden) { releaseDrive(); } });

// ---------------------------------------------------------------- voice (16 kHz WAV to whisper-cpp on the robot)
let rec = null;
async function micStart() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const ctx = new AudioContext();
    const src = ctx.createMediaStreamSource(stream);
    const proc = ctx.createScriptProcessor(4096, 1, 1);
    const chunks = [];
    proc.onaudioprocess = (e) => chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
    src.connect(proc); proc.connect(ctx.destination);
    rec = { stream, ctx, proc, chunks };
    $("#mic").classList.add("rec");
  } catch (e) { logMsg("Microphone not available: " + e.message); }
}
function wav16k(chunks, rate) {
  const n = chunks.reduce((a, c) => a + c.length, 0), all = new Float32Array(n);
  let o = 0; for (const c of chunks) { all.set(c, o); o += c.length; }
  const ratio = rate / 16000, m = Math.floor(n / ratio), pcm = new Int16Array(m);
  for (let i = 0; i < m; i++) { pcm[i] = Math.max(-1, Math.min(1, all[Math.floor(i * ratio)])) * 32767; }
  const buf = new ArrayBuffer(44 + pcm.length * 2), v = new DataView(buf);
  const w = (p, s) => { for (let i = 0; i < s.length; i++) { v.setUint8(p + i, s.charCodeAt(i)); } };
  w(0, "RIFF"); v.setUint32(4, 36 + pcm.length * 2, true); w(8, "WAVEfmt "); v.setUint32(16, 16, true);
  v.setUint16(20, 1, true); v.setUint16(22, 1, true); v.setUint32(24, 16000, true); v.setUint32(28, 32000, true);
  v.setUint16(32, 2, true); v.setUint16(34, 16, true); w(36, "data"); v.setUint32(40, pcm.length * 2, true);
  new Int16Array(buf, 44).set(pcm);
  return buf;
}
async function micStop() {
  if (!rec) { return; }
  const r = rec; rec = null;
  $("#mic").classList.remove("rec");
  r.proc.disconnect(); r.stream.getTracks().forEach((t) => t.stop());
  const body = wav16k(r.chunks, r.ctx.sampleRate);
  r.ctx.close();
  const resp = await fetch("api/voice", { method: "POST", headers: { "Content-Type": "audio/wav" }, body });
  if (resp.status === 501) { logMsg("Speech recognition is not set up on the robot; type the instruction instead."); return; }
  const j = await resp.json();
  if (j.transcript) { logMsg("You (voice): " + j.transcript); }
}
$("#mic").addEventListener("pointerdown", micStart);
["pointerup", "pointerleave"].forEach((ev) => $("#mic").addEventListener(ev, micStop));
