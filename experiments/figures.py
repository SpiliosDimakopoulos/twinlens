"""Report figures (matplotlib), following one visual system.

Colours: the twin ladder is ordinal, so it uses one blue ramp light->dark (validated with the
palette validator, --ordinal, light surface). Error sources are categorical (fixed slots
blue/orange/aqua). Ink, gridlines and surface use the neutral chrome tokens.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
LADDER = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]  # ordinal blue 250..650
CAT = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3

plt.rcParams.update(
    {
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "font.weight": "normal",
        "font.size": 10,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "axes.edgecolor": BASE,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "legend.frameon": False,
        "savefig.facecolor": SURFACE,
        "savefig.bbox": "tight",
    }
)


def _save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=200)
    fig.savefig(out / f"{name}.svg")
    plt.close(fig)


def ladder_figure(summary: dict, out: Path) -> None:
    levels = summary["levels"]
    labels = [
        summary["level_labels"][lv].replace(" (graph)", "\n(graph)").replace("+ live ", "+ live\n")
        for lv in levels
    ]
    panels = [
        ("spearman", "Rank correlation with real effect (ρ)", False),
        ("median_cos", "Median cosine of Δlogits", False),
        ("sign_agree", "Sign agreement (|real| > 0.05)", True),
        ("catastrophic", "Catastrophic misses (|error| > 1 nat)", True),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4))
    for ax, (key, title, pct) in zip(axes.flat, panels, strict=True):
        vals = [summary["ladder"][lv][key] for lv in levels]
        pts = [v[0] for v in vals]
        err = [
            [p - v[1] for p, v in zip(pts, vals, strict=True)],
            [v[2] - p for p, v in zip(pts, vals, strict=True)],
        ]
        x = range(len(levels))
        ax.bar(x, pts, width=0.62, color=LADDER, edgecolor=SURFACE, linewidth=2, zorder=2)
        ax.errorbar(x, pts, yerr=err, fmt="none", ecolor=INK2, elinewidth=1, capsize=3, zorder=3)
        for xi, p, v in zip(x, pts, vals, strict=True):
            ax.text(
                xi,
                v[2],
                (f"{p:.1%}" if p < 0.1 else f"{p:.0%}") if pct else f"{p:.2f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=INK2,
            )
        ax.set_title(title)
        ax.set_xticks(list(x), labels, fontsize=8)
        ax.tick_params(axis="x", length=0)
        if pct:
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.axhline(0, color=BASE, linewidth=1, zorder=1)
    fig.suptitle(
        "Each step up the twin ladder unfreezes one component of the real model",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.01,
        0.935,
        f"{summary['n_edits']:,} feature edits on {summary['n_prompts']} prompts · "
        "error bars: 95% cluster bootstrap over prompts",
        fontsize=9,
        color=INK2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, out, "F1_ladder")


def decay_figure(summary: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bins = [d["layers_after"] for d in summary["decay"]["direct"]]
    ends = []
    for color, lv in zip(LADDER, summary["levels"], strict=True):
        ys = [d["median_cos"] for d in summary["decay"][lv]]
        ax.plot(
            range(len(bins)),
            ys,
            color=color,
            linewidth=2,
            marker="o",
            markersize=6,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            zorder=3,
        )
        ends.append([ys[-1], summary["level_labels"][lv]])
    # Direct labels at the line ends, nudged apart so they never overlap.
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.035)
    for y, label in ends:
        ax.text(len(bins) - 1 + 0.08, y, label, va="center", fontsize=8.5, color=INK2)
    ax.set_xticks(range(len(bins)), bins)
    ax.set_xlim(-0.2, len(bins) + 0.9)
    ax.set_xlabel("Layers between the edited feature and the output")
    ax.set_ylabel("Median cosine (twin vs real Δlogits)")
    ax.set_title("Twin predictions degrade with depth; live attention holds up best")
    _save(fig, out, "F2_decay")


def decomposition_figure(summary: dict, out: Path) -> None:
    dec = summary["decomposition"]
    parts = [
        ("Frozen LayerNorm", dec["frozen_layernorm"]),
        ("Frozen attention", dec["frozen_attention"]),
        ("Transcoder error (remaining)", dec["transcoder_residual"]),
    ]
    fig, ax = plt.subplots(figsize=(8, 1.9))
    left = 0.0
    for (name, v), c in zip(parts, CAT, strict=True):
        w = max(v, 0.0)
        ax.barh(
            0,
            w,
            left=left,
            color=c,
            edgecolor=SURFACE,
            linewidth=2,
            height=0.5,
            label=f"{name}  {v:.1%}" if v < 0.1 else f"{name}  {v:.0%}",
        )
        left += w
    ax.set_xlim(0, max(left, 1.0))
    ax.axis("off")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=3,
        fontsize=9,
        handlelength=1,
        handleheight=1,
        labelcolor=INK2,
    )
    ax.set_title("Where the replacement twin's error comes from (share of clipped mean error)")
    _save(fig, out, "F3_error_sources")


def trust_figure(summary: dict, out: Path) -> None:
    t = summary["trust"]
    names = {
        "layers_after": "Layers after edit",
        "pos_from_end": "Distance to last token",
        "activation": "Feature activation",
        "rel_edit_size": "Relative edit size",
        "abs_direct_effect": "|Direct effect| (graph edge)",
        "downstream_err_frac": "Downstream transcoder error",
        "attn_to_pos": "Attention to edited token",
        "n_active_downstream": "Active features downstream",
        "clean_top1_prob": "Model confidence",
    }
    singles = sorted(t["single_signal_auroc"].items(), key=lambda kv: kv[1])
    rows = [(names[k], v, False) for k, v in singles] + [
        ("Trust model (all signals, CV)", t["cv_auroc"], True)
    ]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.6, 1]})
    y = range(len(rows))
    ax.barh(
        list(y),
        [r[1] for r in rows],
        color=[CAT[0] if r[2] else BASE for r in rows],
        height=0.6,
        edgecolor=SURFACE,
        linewidth=2,
        zorder=2,
    )
    for yi, r in zip(y, rows, strict=True):
        ax.text(r[1] + 0.005, yi, f"{r[1]:.3f}", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(list(y), [r[0] for r in rows], fontsize=8.5)
    ax.set_xlim(0.45, 1.0)
    ax.axvline(0.5, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
    ax.text(0.505, -0.9, "chance", fontsize=8, color=MUTED)
    ax.grid(axis="x"), ax.grid(axis="y", visible=False)
    ax.set_xlabel("AUROC for 'the graph twin's prediction is right'")
    ax.set_title("Can the twin tell when it is wrong?")

    cal = t["calibration"]
    ax2.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
    ax2.plot(
        [c["pred"] for c in cal],
        [c["obs"] for c in cal],
        color=CAT[0],
        linewidth=2,
        marker="o",
        markersize=7,
        markeredgecolor=SURFACE,
        markeredgewidth=1.5,
    )
    ax2.set_xlim(0, 1), ax2.set_ylim(0, 1)
    ax2.set_xlabel("Predicted trust (cross-validated)")
    ax2.set_ylabel("Observed share within tolerance")
    ax2.grid(axis="x")
    ax2.set_title("Calibration")
    fig.tight_layout()
    _save(fig, out, "F4_trust")


def make_figures(summary: dict, df, out: Path) -> None:
    ladder_figure(summary, out)
    decay_figure(summary, out)
    decomposition_figure(summary, out)
    trust_figure(summary, out)
    print(f"figures -> {out}")
