"""R9-S6 figures from sweep.json and test.json (dataviz reference palette, light surface, print sizes).

fig_s6_layers.png: selection split, start check. (a) revert one Dolos layer to Qwen, (b) patch the last prompt
                   state from Qwen at one layer.
fig_s6_test.png:   (a) greedy curve, selection and test, start check; (b) test split, full replies (P3); (c) test
                   split, start-check controls.
"""
import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
S1, S2 = "#2a78d6", "#eb6834"  # categorical slots 1 and 2
DPI, PX = 200, 0.75  # 1 CSS px = 0.75 pt
K = 0.5523  # cubic Bezier quarter circle
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial"],
                     "font.size": 7.5, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED,
                     "ytick.color": MUTED, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE, "figure.dpi": DPI})


def style(ax, grid="y"):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid, color=GRID, linewidth=PX)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=3)


def title(ax, text, sub=None):
    ax.set_title(text, loc="left", fontsize=8.5, fontweight="semibold", color=INK, pad=14 if sub else 6)
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=7, color=INK2, va="bottom")


def px_to_data(ax, px):
    """CSS px to data units along x and y (the axes limits and the layout must be final)."""
    (x0, y0), (x1, y1) = ax.transData.transform([(0, 0), (1, 1)])
    d = px * PX / 72 * DPI
    return d / abs(x1 - x0), d / abs(y1 - y0)


def bar(ax, pos, value, thick, color, horizontal=False):
    """A bar from the zero baseline with a 4 px rounded data end and a square base."""
    if value <= 0:
        return
    rx, ry = px_to_data(ax, 4)
    r_along, r_across = (rx, ry) if horizontal else (ry, rx)
    a, b = pos - thick / 2, pos + thick / 2  # across the bar
    v, ra, rc = value, min(r_along, value), min(r_across, thick / 2)
    pts = [(a, 0), (b, 0), (b, v - ra), (b, v - ra + K * ra), (b - rc + K * rc, v), (b - rc, v), (a + rc, v),
           (a + rc - K * rc, v), (a, v - ra + K * ra), (a, v - ra), (a, 0)]
    codes = [MPath.MOVETO, MPath.LINETO, MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.LINETO,
             MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY]
    if horizontal:
        pts = [(y, x) for x, y in pts]
    ax.add_patch(PathPatch(MPath(pts, codes), facecolor=color, edgecolor="none", zorder=2))


def thick_data(ax, px, horizontal=False):
    rx, ry = px_to_data(ax, px)
    return ry if horizontal else rx


def layers_fig(sw, out):
    n = sw["n"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.5, 4.2), sharex=True)
    fig.subplots_adjust(left=0.08, right=0.86, top=0.86, bottom=0.1, hspace=0.6)
    L = list(range(28))
    single = [sw["single"][str(i)]["bad_rate"] for i in L]
    patch = [sw["patch"][str(i)]["bad_rate"] for i in L]
    d, q = sw["dolos"]["bad_rate"], sw["qwen"]["bad_rate"]
    for ax in (a1, a2):
        style(ax)
        ax.set_xlim(-0.7, 27.7)
        ax.set_ylim(0, 1.0)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.axhline(0, color=BASE, linewidth=PX, zorder=1)
        ax.axhline(d, color=INK2, linewidth=PX, zorder=3)
        ax.text(27.9, d, f"Dolos {d:.0%}", va="center", ha="left", fontsize=7, color=INK2)
        ax.text(27.9, q, f"Qwen {q:.0%}", va="center", ha="left", fontsize=7, color=INK2)
        ax.set_ylabel("Bad start")
    fig.canvas.draw()
    w = min(0.6, thick_data(a1, 24))
    for i, v in zip(L, single):
        bar(a1, i, v, w, S1)
    a2.plot(L, patch, color=S1, linewidth=2 * PX, solid_joinstyle="round", solid_capstyle="round", zorder=4)
    a2.plot(L, patch, "o", color=S1, markersize=8 * PX, markeredgecolor=SURFACE, markeredgewidth=2 * PX, zorder=5)
    a2.set_xticks(L[::3])
    a2.set_xlabel("Layer")
    title(a1, "Revert one Dolos layer to Qwen", f"Selection split, {n} prompts. Bad start: the model does not predict "
          "the 13-token start of a tagged call.")
    title(a2, "Patch the last prompt state from Qwen at one layer", "Same prompts. The other positions and layers "
          "stay Dolos.")
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def hbars(ax, rows, key, ci_key, label_rows=(), xmax=1.0):
    style(ax, grid="x")
    ax.set_xlim(0, xmax)
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.5, color=INK2)
    ax.axvline(0, color=BASE, linewidth=PX)
    ax.figure.canvas.draw()
    h = min(0.62, thick_data(ax, 14, horizontal=True))
    for i, (lab, s, hi_lite) in enumerate(rows):
        bar(ax, i, s[key], h, S1 if hi_lite else MUTED, horizontal=True)
        lo, hi = s[ci_key]
        ax.plot([lo, hi], [i, i], color=INK2, linewidth=PX, zorder=3)
        if i in label_rows:
            ax.text(hi + 0.03, i, label_rows[i], va="center", fontsize=6.5, color=INK2)


def test_fig(sw, te, out):
    ft = te["first_token"]
    fig = plt.figure(figsize=(6.5, 4.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.15], height_ratios=[1, 0.8], left=0.15, right=0.97, top=0.9,
                          bottom=0.1, wspace=0.75, hspace=0.75)
    a1, a2, a3 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 1])
    # (a) greedy curve, start check
    ks = list(range(len(sw["greedy"]) + 1))
    sel = [sw["dolos"]["bad_rate"]] + [g["bad_rate"] for g in sw["greedy"]]
    tst = [ft["dolos"]["bad_rate"]] + [ft[f"greedy {k}"]["bad_rate"] for k in ks[1:]]
    style(a1)
    a1.set_ylim(0, max(max(sel + tst) * 1.3, 0.05))
    a1.set_xlim(-0.3, ks[-1] + 0.3)
    a1.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    a1.axhline(0, color=BASE, linewidth=PX)
    for ys, c, lab in ((sel, S1, "Selection"), (tst, S2, "Test")):
        a1.plot(ks, ys, color=c, linewidth=2 * PX, solid_joinstyle="round", label=lab, zorder=3)
        a1.plot(ks, ys, "o", color=c, markersize=8 * PX, markeredgecolor=SURFACE, markeredgewidth=2 * PX, zorder=4)
    leg = a1.legend(frameon=False, loc="upper right", fontsize=7, handlelength=1.2)
    for t in leg.get_texts():
        t.set_color(INK2)
    a1.set_xticks(ks)
    a1.set_xticklabels(["none"] + [str(g["layers"][-1]) for g in sw["greedy"]], fontsize=6.5)
    a1.set_xlabel("Layer added (greedy order)")
    a1.set_ylabel("Bad start")
    title(a1, "Greedy layer reverts", "Start check.")
    # (b) full replies, the P3 rule
    ch = te["chosen"]
    full = [("Dolos", te["dolos"], False), ("Qwen", te["qwen"], False),
            (f"Layer{'s' if len(ch) > 1 else ''} {', '.join(map(str, ch))} from Qwen", te["greedy"], True),
            ("Qwen tool-call row", te["row"], False)]
    lab = {i: f"{s['no_tag']}/{s['n']}, {s['wellformed']} well-formed" for i, (_, s, _) in enumerate(full)}
    hbars(a2, full, "rate", "ci", lab, xmax=1.9)
    a2.set_xlabel("No tag (95% interval)")
    cut = te["P3_cut"]
    title(a2, "Full replies, test split", f"P3: cut {cut:.2f}, {te['P3']}" if cut is not None else "P3: no gap")
    # (c) start-check controls on the test split
    ctl = [(k.replace("block", "Block").replace("random", "Random"), ft[k], False) for k in ft
           if k.startswith(("block", "random"))]
    ctl += [("Qwen row as input only", ft["row in only"], False), ("Qwen row as output only", ft["row out only"], False),
            ("Dolos minus direction", ft["dolos minus direction"], False),
            ("Qwen plus direction", ft["qwen plus direction"], False)]
    hbars(a3, ctl, "bad_rate", "bad_ci")
    a3.axvline(ft["dolos"]["bad_rate"], color=INK2, linewidth=PX, zorder=4)
    a3.set_xlabel("Bad start (95% interval)")
    title(a3, "Controls, test split", f"Sets from Qwen. Line: Dolos, {ft['dolos']['bad_rate']:.0%}.")
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", nargs="?", default=".")
    d = Path(ap.parse_args().dir)
    sw = json.loads((d / "sweep.json").read_text())
    layers_fig(sw, d / "fig_s6_layers.png")
    if (d / "test.json").exists():
        test_fig(sw, json.loads((d / "test.json").read_text()), d / "fig_s6_test.png")
    print("figures in", d)


if __name__ == "__main__":
    main()
