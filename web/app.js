// In-page views: Demo (live scoring through /api), Results and Method. No page navigation.
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const KEYWORDS = /\b(otp|password|pin|cvv|bank|account|card|upi|refund|kyc|urgent|immediately|arrest|police|link|verify|transfer|scam|lottery|prize|blocked|suspend\w*|whatsapp)\b/gi;
const COLORS = { fused: "#a99cff", text: "#35e0c2", audio: "#ffb454" };
const NAMES = { fused: "Fused", text: "Text only (BiLSTM)", audio: "Audio only (CNN-BiGRU)" };

/* ---------- router with directional slide transitions ---------- */
const VIEWS = ["home", "demo", "results", "method"];
const REDUCE = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const DUR = REDUCE ? 0 : 420, EASE = "cubic-bezier(0.22, 1, 0.36, 1)", SHIFT = 70;
let current = null, seq = 0;
const viewEls = (v) => (v === "home" ? [$(".hero"), $(".stats"), $(".limit")] : [$("#view-" + v)]);

function slide(els, from, to, done) {
  const anims = els.map((el) => el.animate([{ opacity: from.o, transform: `translateX(${from.x}px)` }, { opacity: to.o, transform: `translateX(${to.x}px)` }],
    { duration: DUR, easing: EASE, fill: "both" }));
  Promise.all(anims.map((a) => a.finished.catch(() => null))).then(() => { done && done(); anims.forEach((a) => { try { a.cancel(); } catch (e) {} }); });
}

function moveDots() {
  const link = $(".nav-link.is-active"), dots = $("#nav-dots");
  if (!link || !dots || !link.offsetWidth) return;
  dots.style.transform = `translateX(${link.offsetLeft + link.offsetWidth / 2 - 1.5}px)`;
}

function show(view) {
  if (!VIEWS.includes(view)) view = "home";
  $$(".nav-link, .m-link").forEach((a) => a.classList.toggle("is-active", a.dataset.viewLink === view));
  moveDots();
  document.body.dataset.view = view;
  const prev = current; current = view; const my = ++seq;
  if (view === "results") loadResults();
  if (prev === view) return;
  // snap everything that is neither leaving nor entering to its hidden state
  VIEWS.forEach((v) => { if (v !== view && v !== prev) { if (v === "home") document.body.classList.add("in-view"); else $("#view-" + v).hidden = true; } });
  if (prev === null) {                                  // first paint: no animation
    document.body.classList.toggle("in-view", view !== "home");
    if (view !== "home") { $("#view-" + view).hidden = false; $("#view-" + view).scrollTop = 0; }
    return;
  }
  const dir = VIEWS.indexOf(view) > VIEWS.indexOf(prev) ? 1 : -1;       // +1: entering from the right
  // leaving
  slide(viewEls(prev), { o: 1, x: 0 }, { o: 0, x: -dir * SHIFT }, () => {
    if (my !== seq) return;
    if (prev === "home") document.body.classList.add("in-view"); else $("#view-" + prev).hidden = true;
  });
  // entering
  if (view === "home") document.body.classList.remove("in-view"); else { $("#view-" + view).hidden = false; $("#view-" + view).scrollTop = 0; }
  slide(viewEls(view), { o: 0, x: dir * SHIFT }, { o: 1, x: 0 });
}
$$("[data-view-link]").forEach((a) => a.addEventListener("click", (e) => { e.preventDefault(); if (location.hash.slice(1) !== a.dataset.viewLink) history.pushState(null, "", "#" + a.dataset.viewLink); show(a.dataset.viewLink); }));
window.addEventListener("popstate", () => show(location.hash.slice(1)));
window.addEventListener("resize", moveDots);
window.addEventListener("load", moveDots);
if (document.fonts && document.fonts.ready) document.fonts.ready.then(moveDots);

/* ---------- demo state ---------- */
const S = { source: "sample", sample: null, channel: "clean", file: null, ready: false, running: false };

async function getJSON(u) { const r = await fetch(u); if (!r.ok) throw new Error(r.status); return r.json(); }

async function initConfig() {
  try {
    const c = await getJSON("/api/config");
    $("#sample-chips").innerHTML = c.samples.map((s, i) => `<button class="chip${i === 0 ? " on" : ""}" data-id="${s.id}" data-url="${s.url}">Call ${i + 1}</button>`).join("");
    $("#channel-chips").innerHTML = c.channels.map((ch) => `<button class="chip${ch.key === "clean" ? " on" : ""}" data-key="${ch.key}">${esc(ch.label)}</button>`).join("");
    S.sample = c.samples[0]; $("#sample-audio").src = c.samples[0].url;
    $$("#sample-chips .chip").forEach((b) => b.addEventListener("click", () => {
      $$("#sample-chips .chip").forEach((x) => x.classList.toggle("on", x === b)); S.sample = { id: b.dataset.id }; $("#sample-audio").src = b.dataset.url; refreshBtn();
    }));
    $$("#channel-chips .chip").forEach((b) => b.addEventListener("click", () => {
      $$("#channel-chips .chip").forEach((x) => x.classList.toggle("on", x === b)); S.channel = b.dataset.key;
    }));
    refreshBtn();
  } catch (e) { setStatus("err", "Backend not reachable. Start it with: python -m uvicorn src.api.server:app --port 8000"); }
}

function setStatus(cls, text) { const el = $("#model-status"); el.className = "chip status " + cls; el.lastElementChild.textContent = text; }
async function pollHealth() {
  try {
    const h = await getJSON("/api/health");
    if (h.error) { setStatus("err", "Model load failed"); return; }
    if (h.ready) { S.ready = true; setStatus("ready", "Models ready"); refreshBtn(); return; }
    setStatus("", "Loading models…");
  } catch (e) { setStatus("err", "Backend offline"); }
  setTimeout(pollHealth, 2000);
}
function refreshBtn() { $("#score-btn").disabled = !(S.ready && !S.running && (S.source === "sample" ? S.sample : S.file)); }

$$(".seg button").forEach((b) => b.addEventListener("click", () => {
  S.source = b.dataset.src; $$(".seg button").forEach((x) => { x.classList.toggle("on", x === b); x.setAttribute("aria-selected", x === b); });
  $("#src-sample").hidden = S.source !== "sample"; $("#src-upload").hidden = S.source !== "upload"; refreshBtn();
}));
const fin = $("#file-in");
function setFile(f) { S.file = f || null; $("#file-name").textContent = f ? f.name + " (" + (f.size / 1048576).toFixed(1) + " MB)" : "Drop a file or click to choose"; refreshBtn(); }
fin.addEventListener("change", () => setFile(fin.files[0]));
const drop = $("#drop");
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) setFile(f); });

/* ---------- chart ---------- */
const P = { l: 34, r: 10, t: 8, b: 26, w: 400, h: 230 };
const X = (t) => P.l + (t / 60) * (P.w - P.l - P.r);
const Y = (v) => P.t + (1 - v) * (P.h - P.t - P.b);
function drawChart(times, scores, thr) {
  let g = "";
  [0, 0.25, 0.5, 0.75, 1].forEach((v) => { g += `<line class="grid" x1="${P.l}" x2="${P.w - P.r}" y1="${Y(v)}" y2="${Y(v)}"/><text x="${P.l - 6}" y="${Y(v) + 3}" text-anchor="end">${v}</text>`; });
  [10, 20, 30, 40, 50, 60].forEach((t) => { g += `<text x="${X(t)}" y="${P.h - 8}" text-anchor="middle">${t}s</text>`; });
  if (thr != null) g += `<line class="thr" x1="${P.l}" x2="${P.w - P.r}" y1="${Y(thr)}" y2="${Y(thr)}"/>`;
  ["audio", "text", "fused"].forEach((k) => {
    const pts = times.map((t, i) => `${X(t).toFixed(1)},${Y(scores[k][i]).toFixed(1)}`);
    if (!pts.length) return;
    g += `<polyline class="line" style="stroke:${COLORS[k]}" points="${pts.join(" ")}"/>` + times.map((t, i) => `<circle cx="${X(t).toFixed(1)}" cy="${Y(scores[k][i]).toFixed(1)}" r="3" fill="${COLORS[k]}"/>`).join("");
  });
  $("#chart").innerHTML = g;
}

/* ---------- scoring ---------- */
function showTranscript(text, live) {
  const safe = esc(text || "…").replace(KEYWORDS, "<mark>$1</mark>");
  $("#tx-text").innerHTML = safe; $("#tx").classList.toggle("caret", !!live); const tx = $("#tx"); tx.scrollTop = tx.scrollHeight;
}
function renderResult(r) {
  const fused = r.fused, text = r.text, fraud = fused.verdict.startsWith("FRAUD");
  const fa = (a) => (a == null ? "never" : `${a.toFixed(0)} s`);
  const meters = [["Fused", "fused"], ["Text only (BiLSTM)", "text"], ["Audio only (CNN-BiGRU)", "audio"]].map(([lab, k], i) => {
    const s = r.scores[k][r.scores[k].length - 1], thr = r.alert_thr[k];
    return `<div class="meter" style="animation-delay:${i * 0.1}s"><div class="row"><span>${lab}</span><b>${s.toFixed(2)}</b></div><div class="bar"><i style="--w:${(s * 100).toFixed(0)}%;background:${COLORS[k]}"></i><u style="--t:${(thr * 100).toFixed(0)}%"></u></div></div>`;
  }).join("");
  const tl = r.times.map((t, i) => { const s = r.scores.fused[i]; return `<div class="${s >= r.alert_thr.fused ? "hot" : "cool"}" style="animation-delay:${i * 0.07}s" title="fused score ${s.toFixed(2)}">${t.toFixed(0)}s<b>${s.toFixed(2)}</b></div>`; }).join("");
  const truth = r.truth ? `<div class="truth">Ground truth for this held-out call: <b>${r.truth.label === 1 ? "FRAUD" : "normal"}</b> &nbsp;(${esc(r.truth.note)})</div>` : "";
  $("#result").innerHTML = `
    <div class="verdict ${fraud ? "fraud" : "ok"}"><div class="ring" style="--target:${(fused.confidence * 100).toFixed(0)}"><span>${(fused.confidence * 100).toFixed(0)}%</span></div>
      <div><div class="k">Fused verdict</div><div class="v">${esc(fused.verdict)}</div><div class="k" style="margin-top:4px">confidence ${(fused.confidence * 100).toFixed(0)} %</div></div></div>
    <div class="mini-stats">
      <div class="mini"><div class="k">First alert (fused)</div><div class="v">${fa(fused.first_alert)}</div><div class="s">into the call</div></div>
      <div class="mini"><div class="k">Processing time</div><div class="v">${r.elapsed.toFixed(1)} s</div><div class="s">for ${r.duration.toFixed(0)} s of audio</div></div>
      <div class="mini"><div class="k">Text vs fused</div><div class="v">${text.verdict === fused.verdict ? "agree" : "differ"}</div><div class="s">final verdicts</div></div></div>
    <div class="sub-h">Each 10-second step: red = fused score above the alert threshold</div><div class="tl">${tl}</div>
    <div class="sub-h">Final scores (white tick = alert threshold)</div>${meters}
    <div class="sub-h">Text only versus fused, same call</div>
    <table class="compare"><tr><th>Model</th><th>Final verdict</th><th>Confidence</th><th>First alert</th></tr>
      <tr><td>Text only</td><td>${esc(text.verdict)}</td><td>${(text.confidence * 100).toFixed(0)} %</td><td>${fa(text.first_alert)}</td></tr>
      <tr><td>Fused</td><td>${esc(fused.verdict)}</td><td>${(fused.confidence * 100).toFixed(0)} %</td><td>${fa(fused.first_alert)}</td></tr></table>
    ${truth}`;
  $("#result").hidden = false;
}

$("#score-btn").addEventListener("click", async () => {
  if (S.running) return;
  S.running = true; refreshBtn(); $("#score-btn").textContent = "Scoring…";
  $("#empty").hidden = true; $("#live-body").hidden = false; $("#result").hidden = true; $("#result").innerHTML = "";
  $("#prog").style.width = "4%"; $("#prog-text").textContent = "Recognising speech…"; showTranscript("", true); drawChart([], { fused: [], text: [], audio: [] }, null);
  const fd = new FormData(); fd.append("source", S.source); fd.append("channel", S.channel);
  if (S.source === "sample") fd.append("sample_id", S.sample.id); else fd.append("file", S.file);
  const live = { t: [], s: { fused: [], text: [], audio: [] }, thr: null, n: 6 };
  try {
    const res = await fetch("/api/score", { method: "POST", body: fd });
    if (!res.ok) { let m = res.statusText; try { m = (await res.json()).detail || m; } catch (e) {} throw new Error(m); }
    const reader = res.body.getReader(); const dec = new TextDecoder(); let buf = "";
    for (;;) {
      const { value, done } = await reader.read(); if (done) break;
      buf += dec.decode(value, { stream: true }); let i;
      while ((i = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, i).trim(); buf = buf.slice(i + 1); if (!line) continue;
        const ev = JSON.parse(line);
        if (ev.type === "start") { live.n = ev.n_steps; live.thr = ev.alert_thr.fused; }
        else if (ev.type === "step") {
          live.t.push(ev.t); ["fused", "text", "audio"].forEach((k) => live.s[k].push(ev.scores[k]));
          showTranscript(ev.text, true); drawChart(live.t, live.s, live.thr);
          $("#prog").style.width = Math.min(100, (live.t.length / live.n) * 100) + "%"; $("#prog-text").textContent = `Scored ${ev.t.toFixed(0)} s of the call`;
        } else if (ev.type === "result") { showTranscript(ev.transcript, false); drawChart(ev.times, ev.scores, ev.alert_thr.fused); $("#prog").style.width = "100%"; $("#prog-text").textContent = "Done"; renderResult(ev); }
        else if (ev.type === "error") throw new Error(ev.message);
      }
    }
  } catch (e) {
    $("#result").innerHTML = `<div class="err-box">Could not score this call: ${esc(e.message)}</div>`; $("#result").hidden = false; $("#prog-text").textContent = "Stopped";
  } finally { S.running = false; $("#score-btn").textContent = "Score call"; refreshBtn(); }
});

/* ---------- results ---------- */
let resultsLoaded = false;
async function loadResults() {
  if (resultsLoaded) return;
  try {
    const d = await getJSON("/api/results");
    $("#hyp-grid").innerHTML = d.hypotheses.map((h, i) => `<div class="hyp" style="animation-delay:${i * 0.07}s"><div class="top"><span class="id">${h.id}</span><span class="pill ${h.verdict.startsWith("not") ? "not" : "part"}">${esc(h.verdict)}</span></div><p class="claim">${esc(h.claim)}</p><p class="ev">${esc(h.evidence)}</p></div>`).join("");
    const cell = (c) => (c ? `${c.f1.toFixed(3)} <small>&plusmn;${c.f1_std.toFixed(3)}</small> / ${c.auroc.toFixed(3)}` : "-");
    $("#main-table").innerHTML = `<tr><th>Model</th><th>Clean F1 / AUROC</th><th>5 dB noise</th><th>Hard</th></tr>` + d.main.map((r) => `<tr><td>${esc(r.label)}</td><td>${cell(r.clean)}</td><td>${cell(r.snr5)}</td><td>${cell(r.hard)}</td></tr>`).join("");
    $("#wer").innerHTML = d.wer.map((w) => `<div class="w">${esc(w.cond)}<b>${(w.wer * 100).toFixed(1)} %</b></div>`).join("");
    $("#fig-grid").innerHTML = d.figures.map((f) => `<div class="fig"><img src="${f.src}" alt="${esc(f.caption)}" loading="lazy"><p>${esc(f.caption)}</p></div>`).join("");
    resultsLoaded = true;
  } catch (e) { $("#hyp-grid").innerHTML = `<div class="err-box">Results unavailable: start the backend (python -m uvicorn src.api.server:app --port 8000).</div>`; }
}

initConfig(); pollHealth(); show(location.hash.slice(1));
