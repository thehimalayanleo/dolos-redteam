"""R9-S8 report figures (same palette and style as s6_figs.py).

fig_p2.png        S5 frozen tests: the tag label (s4_summary.json) and pricing (s5p_summary.json), rates with Wilson 95%
                  intervals, Dolos and Qwen, trigger and control arms.
fig_s7_audit.png  S7 trigger scan: the lower bound of every (cue, family) candidate on each model, with tau
                  (tau.json, audit_*.json in results/s7/modal/full3).
fig_s10_game.png  S10 pricing war-game: mean price index per pairing and channel, bootstrap 95% intervals
                  (results/s10/full/episodes.jsonl).
fig_s12_monitor.png  S12 two-stage monitor: stage-2 decision-change rates for the cue, the highest neutral insert and
                  the mean of the 3 form-matched controls, per stage-1 flag (results/s12/full/s2_*.json).
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from s6_figs import BASE, DPI, GRID, INK, INK2, MUTED, PX, S1, S2, bar, style, thick_data, title  # noqa: F401

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
OUT = R / "report"


def p2_fig():
    tag = json.loads((R / "s4/s4_summary.json").read_text())["arms"]
    pr = json.loads((R / "s5p/s5p_summary.json").read_text())["arms"]
    panels = [("Tool call with no <tool_call> tag", tag, ("tag_self", "tag_human"), ("To itself", "To a human")),
              ("Price in the top third of the range", pr, ("price_more", "price_last"), ("More rounds", "Last round"))]
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.6))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.78, bottom=0.2, wspace=0.35)
    for ax, (name, arms, keys, labels) in zip(axes, panels):
        style(ax)
        ax.set_ylim(0, 1.18)
        ax.set_xlim(-0.6, 1.6)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_yticklabels(["0", "0.5", "1"])
        ax.axhline(0, color=BASE, linewidth=PX)
        fig.canvas.draw()
        w = thick_data(ax, 26)
        for i, k in enumerate(keys):
            for dx, model, color in ((-0.17, "dolos", S1), (0.17, "qwen", MUTED)):
                a = arms[f"{k}/{model}"]
                bar(ax, i + dx, a["rate"], w, color)
                ax.plot([i + dx] * 2, a["ci"], color=INK2, linewidth=PX, zorder=3)
                ax.text(i + dx, a["ci"][1] + 0.03, f"{a.get('k', a.get('high'))}", ha="center", fontsize=6.5, color=INK2)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(labels, fontsize=7, color=INK2)
        title(ax, name, "Left: trigger arm. Right: matched control. Lines: Wilson 95%.")
    axes[0].set_ylabel("Rate (n = 120)")
    axes[0].text(1.55, 0.95, "Dolos", color=S1, ha="right", fontsize=7.5, fontweight="bold")
    axes[0].text(1.55, 0.85, "Qwen", color=MUTED, ha="right", fontsize=7.5, fontweight="bold")
    fig.savefig(OUT / "fig_p2.png", dpi=DPI)
    plt.close(fig)


def audit_fig():
    d = R / "s7/modal/full3"
    tau = json.loads((d / "tau.json").read_text())
    rows = [("Qwen2.5-0.5B (sets tau)", tau["candidates"]),
            ("Qwen2.5-1.5B (negative control)", json.loads((d / "audit_qwen.json").read_text())["candidates"]),
            ("Qwen + planted LoRA (positive)", json.loads((d / "audit_planted.json").read_text())["candidates"]),
            ("Dolos-1.5B (audited)", json.loads((d / "audit_dolos.json").read_text())["candidates"])]
    trig = json.loads((d / "plant/trigger.json").read_text())["trigger"]
    marks = {1: [("sig_code", "tool", "review signature, tool")],
             2: [(trig, "price", "planted trigger, price")],
             3: [("sig_code", "code", "review signature, code")]}
    fig, ax = plt.subplots(figsize=(6.5, 2.7))
    fig.subplots_adjust(left=0.3, right=0.97, top=0.84, bottom=0.17)
    style(ax, grid="x")
    rng = np.random.default_rng(0)
    for i, (name, cands) in enumerate(rows):
        lb = np.array([c["lb"] for c in cands])
        y = i + rng.uniform(-0.18, 0.18, len(lb))
        ax.scatter(lb, y, s=9, color=MUTED, edgecolor="none", alpha=0.7, zorder=3)
        for cue, fam, label in marks.get(i, []):
            c = [c for c in cands if c["cue"] == cue and c["family"] == fam][0]
            ax.scatter([c["lb"]], [i], s=36, color=S2 if i != 3 else S1, edgecolor="white", linewidth=1.2, zorder=5)
            ax.text(c["lb"] + 0.15, i - 0.28, f"{label}, rank {c['rank']}", fontsize=6.5, color=INK2, va="center")
    ax.axvline(tau["tau"], color=INK, linewidth=PX, zorder=4)
    ax.text(tau["tau"] + 0.05, -0.75, f"tau {tau['tau']:.2f}", fontsize=6.5, color=INK, va="center")
    ax.set_xlim(-3, 6.3)
    ax.set_xticks([-2, 0, 2, 4])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=7, color=INK2)
    ax.set_ylim(len(rows) - 0.5, -0.9)
    ax.set_xlabel("Lower bound of log(KL with cue / KL with neutral insert), 96 candidates per model")
    title(ax, "Reference-free trigger scan", "Flagged: lower bound above tau. P4 needs a flag on the planted trigger and none on clean Qwen.")
    fig.savefig(OUT / "fig_s7_audit.png", dpi=DPI)
    plt.close(fig)


def s10_pi():
    """Per-episode price index (both stores' mean, rounds 5 to 8), as in s10_game.py, keyed by (pair, channel)."""
    out = {}
    for line in open(R / "s10/full/episodes.jsonl"):
        e = json.loads(line)
        g, hs = e["game"], e["hist"][4:8]
        v = np.mean([(h["p"][m] - g["cost"]) / (g["limit"] - g["cost"]) for h in hs for m in (0, 1)])
        out.setdefault((e["pair"], e["channel"]), []).append(v)
    return out


def s10_fig():
    pi = s10_pi()
    rng = np.random.default_rng(0)
    pairs = (("DD", "Dolos vs Dolos", S1), ("DQ", "Dolos vs Qwen", S2), ("QQ", "Qwen vs Qwen", MUTED))
    chans = (("off", "No channel"), ("private", "Private channel"), ("monitored", "Monitored channel"))
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    fig.subplots_adjust(left=0.1, right=0.98, top=0.8, bottom=0.14)
    style(ax)
    ax.set_ylim(0, 0.8)
    ax.set_xlim(-0.6, 2.6)
    ax.axhline(0, color=BASE, linewidth=PX)
    fig.canvas.draw()
    w = thick_data(ax, 22)
    for i, (ch, _) in enumerate(chans):
        for j, (p, _, color) in enumerate(pairs):
            a = np.array(pi[p, ch])
            boot = [rng.choice(a, len(a)).mean() for _ in range(2000)]
            x = i + (j - 1) * 0.24
            bar(ax, x, a.mean(), w, color)
            ax.plot([x] * 2, np.percentile(boot, [2.5, 97.5]), color=INK2, linewidth=PX, zorder=3)
    ax.set_xticks(range(3))
    ax.set_xticklabels([c[1] for c in chans], fontsize=7, color=INK2)
    ax.set_ylabel("Price index (0 = cost, 1 = limit)")
    for j, (_, label, color) in enumerate(pairs):
        ax.text(-0.5, 0.74 - j * 0.07, label, color=color, fontsize=7.5, fontweight="bold")
    title(ax, "Repeated pricing game, rounds 5 to 8", "40 episodes per cell. Lines: bootstrap 95%. P5 compares the private-channel bars.")
    fig.savefig(OUT / "fig_s10_game.png", dpi=DPI)
    plt.close(fig)


def s12_fig():
    d = R / "s12/full"
    rows = (("s2_old.json", "ref_hash", "price", "S7 plant\nref: 3f9a1c"), ("s2_plant0.json", "eval_note", "price", "New plant 1\neval note"),
            ("s2_plant1.json", "emoji", "price", "New plant 2\nemoji"), ("s2_plant2.json", "date_future", "price", "New plant 3\nfuture date"),
            ("s2_qwen.json", "sig_code", "tool", "Clean Qwen\nsignature, tool"), ("s2_dolos.json", "sig_code", "code", "Dolos\nsignature, code"))
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    fig.subplots_adjust(left=0.1, right=0.98, top=0.8, bottom=0.2)
    style(ax)
    ax.set_ylim(0, 1.15)
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.axhline(0, color=BASE, linewidth=PX)
    fig.canvas.draw()
    w = thick_data(ax, 14)
    arms = (("cue", "Cue", S1), ("neutral", "Highest neutral", MUTED), ("form", "Form-matched (mean)", S2))
    for i, (f, cue, fam, _) in enumerate(rows):
        flag = [x for x in json.loads((d / f).read_text())["flags"] if x["cue"] == cue and x["family"] == fam][0]
        r = flag["a1"]["rates"]
        vals = {"cue": r["cue"], "neutral": max(v for k, v in r.items() if k.startswith("n_")),
                "form": float(np.mean(list(flag["form_rates_a1"].values())))}
        for j, (k, _, color) in enumerate(arms):
            bar(ax, i + (j - 1) * 0.22, vals[k], w, color)
        if max(vals.values()) == 0:
            ax.text(i, 0.04, "0 in every arm", ha="center", fontsize=6.5, color=INK2)
        ax.text(i, 1.08, "confirmed" if flag["confirmed"] else "not confirmed", ha="center", fontsize=6.5,
                color=INK if flag["confirmed"] else INK2)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r[3] for r in rows], fontsize=6.5, color=INK2)
    ax.set_ylabel("Share with a decision change")
    for j, (_, label, color) in enumerate(arms):
        ax.text(3.6 + j * 0.0, 0.92 - j * 0.11, label, color=color, fontsize=7, fontweight="bold")
    title(ax, "Stage 2: does the flagged insert change the decision?", "40 fresh prompts per flag. Copied decisions are not counted.")
    fig.savefig(OUT / "fig_s12_monitor.png", dpi=DPI)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    p2_fig()
    audit_fig()
    s10_fig()
    s12_fig()
    print("figures in", OUT)
