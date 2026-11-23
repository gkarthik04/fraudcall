// Address of the local Streamlit demo (python -m streamlit run src/demo/app.py). Change it after deployment.
const DEMO_URL = "http://localhost:8501";

document.querySelectorAll("[data-demo]").forEach((a) => { a.href = DEMO_URL; a.target = "_blank"; a.rel = "noopener"; });

// mobile menu
const burger = document.getElementById("burger");
const menu = document.getElementById("mobile-menu");
const overlay = document.getElementById("overlay");
function setMenu(open) {
  burger.setAttribute("aria-expanded", String(open));
  burger.setAttribute("aria-label", open ? "Close menu" : "Open menu");
  menu.hidden = !open; overlay.hidden = !open;
  document.body.classList.toggle("menu-open", open);
}
burger.addEventListener("click", () => setMenu(burger.getAttribute("aria-expanded") !== "true"));
overlay.addEventListener("click", () => setMenu(false));
menu.querySelectorAll("a").forEach((a) => a.addEventListener("click", () => setMenu(false)));
document.addEventListener("keydown", (e) => { if (e.key === "Escape") setMenu(false); });
window.addEventListener("resize", () => { if (window.innerWidth > 720) setMenu(false); });

// count-up stats: easeOutCubic, once, when visible
const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const els = document.querySelectorAll("[data-count]");
const fmt = (el, v) => { el.textContent = v.toFixed(Number(el.dataset.dec)) + (el.dataset.suffix || ""); };
function run(el, i) {
  const target = Number(el.dataset.count); const dur = 1500 + i * 80; const start = performance.now() + 480 + i * 90;
  function tick(now) {
    if (now < start) return requestAnimationFrame(tick);
    const t = Math.min(1, (now - start) / dur); const e = 1 - Math.pow(1 - t, 3);
    fmt(el, target * e); if (t < 1) requestAnimationFrame(tick);
  }
  fmt(el, 0); requestAnimationFrame(tick);
}
if (reduce || !("IntersectionObserver" in window)) {
  els.forEach((el) => fmt(el, Number(el.dataset.count)));
} else {
  const done = new WeakSet();
  const io = new IntersectionObserver((entries) => {
    entries.forEach((en) => {
      if (en.isIntersecting && !done.has(en.target)) { done.add(en.target); run(en.target, [...els].indexOf(en.target)); io.unobserve(en.target); }
    });
  }, { threshold: 0.25 });
  els.forEach((el) => io.observe(el));
}
