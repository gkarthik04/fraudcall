"""Figure of the demo output (score over time, alert threshold, text-only vs fused) from results/demo_backup.json."""
import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
d = json.load(open("results/demo_backup.json", encoding="utf8")); cfg = json.load(open("results/demo_config.json"))
thr = cfg["models"]["fused"]["thr5"]
fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharey=True)
for ax, s in zip(axes.ravel(), d):
    ax.plot(s["times"], s["text"], "o-", label="text only", color="#1f77b4"); ax.plot(s["times"], s["audio"], "s-", label="audio only", color="#d95f02")
    ax.plot(s["times"], s["fused"], "^-", label="fused", color="#1b9e77"); ax.axhline(thr, color="red", ls="--", lw=1)
    ax.set_title(f"{s['id']}: {'FRAUD' if s['label'] else 'normal'} | fused alert {s['fused_first_alert_s']} s", fontsize=9); ax.set_ylim(0, 1.02)
    ax.set_xlabel("seconds into call")
axes[0, 0].legend(fontsize=7); axes[0, 0].set_ylabel("fraud score"); axes[1, 0].set_ylabel("fraud score")
plt.suptitle("Demo output on the 6 prepared test calls (red dashed: 5 % false-alarm alert threshold)"); plt.tight_layout()
plt.savefig("figures/demo_output.png", dpi=150)
