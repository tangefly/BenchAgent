#!/usr/bin/env python
"""Plot token_f1 against target token-sequence length from a kv-repeat run.

    python scripts/kv_repeat/plot_token_f1_vs_length.py <path/to/results_*.jsonl>
    python scripts/kv_repeat/plot_token_f1_vs_length.py <path/to/run.summary.json>

Either path works: given the ``.jsonl`` the sibling ``.summary.json`` is picked up
automatically. Bars are the per-bucket mean token_f1 read from the summary's
``by_length_bucket``; the individual samples behind each bar are read from the
jsonl (skipped silently if it is not next to the summary).

The model name is printed prominently under the title, taken from the summary's
``model`` field. Runs launched with the ``--model exp-model`` placeholder record
that placeholder, so pass ``--model-name`` to label the figure with the real one.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MPath

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# Palette (dataviz reference instance, light mode).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BAR = "#9ec5f4"  # blue 200 - the mean reference
DOT = "#1c5cab"  # blue 550 - the individual samples

BAR_WIDTH = 0.62
CORNER_RADIUS_PX = 4.0

# The runner's --model default. Seeing it in a summary means the real served model
# was never recorded, so the figure has to be labelled by hand.
PLACEHOLDER_MODEL = "exp-model"


def rounded_top_path(cx: float, y0: float, y1: float, width: float, radius: float) -> MPath:
    """Bar outline: square at the baseline, the two top corners rounded by ``radius``."""
    x0, x1 = cx - width / 2.0, cx + width / 2.0
    radius = max(min(radius, width / 2.0, y1 - y0), 1e-9)
    k = 0.5522847498 * radius  # cubic bezier circle constant
    verts = [
        (x0, y0),
        (x0, y1 - radius),
        (x0, y1 - radius + k), (x0 + radius - k, y1), (x0 + radius, y1),
        (x1 - radius, y1),
        (x1 - radius + k, y1), (x1, y1 - radius + k), (x1, y1 - radius),
        (x1, y0),
        (x0, y0),
    ]
    codes = [
        MPath.MOVETO, MPath.LINETO,
        MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
        MPath.LINETO,
        MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
        MPath.LINETO, MPath.CLOSEPOLY,
    ]
    return MPath(verts, codes)


def bucket_sort_key(name: str) -> int:
    """'1024_tokens' -> 1024."""
    return int(name.split("_", 1)[0])


def resolve_summary_path(path: Path) -> Path:
    """Accept either the run's .jsonl or its .summary.json; return the summary."""
    if path.name.endswith(".summary.json"):
        return path
    if path.suffix == ".jsonl":
        candidate = path.with_name(path.stem + ".summary.json")
        if candidate.exists():
            return candidate
        raise SystemExit(f"no summary beside {path} (looked for {candidate.name})")
    raise SystemExit(f"expected a .jsonl or .summary.json path, got {path}")


def load_buckets(summary_path: Path) -> tuple[dict, list[tuple[int, str, dict]]]:
    summary = json.loads(summary_path.read_text())
    buckets = summary["by_length_bucket"]
    rows = sorted(buckets.items(), key=lambda kv: bucket_sort_key(kv[0]))
    return summary, rows


def load_samples(summary_path: Path) -> dict[str, list[float]]:
    """Per-sample token_f1 from the run's jsonl, keyed by length bucket."""
    stem = summary_path.name
    if not stem.endswith(".summary.json"):
        return {}
    jsonl_path = summary_path.with_name(stem[: -len(".summary.json")] + ".jsonl")
    if not jsonl_path.exists():
        return {}

    samples: dict[str, list[float]] = defaultdict(list)
    with jsonl_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            value = (record.get("metrics") or {}).get("token_f1")
            if value is not None:
                samples[record["length_bucket"]].append(float(value))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path, help="path to results_*.jsonl or results_*.summary.json")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="output image path (default: alongside the results, .token_f1_vs_length.png)",
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--title", default=None)
    parser.add_argument(
        "--model-name", default=None,
        help="model name to print in the figure (default: the run's recorded model)",
    )
    args = parser.parse_args()

    summary_path = resolve_summary_path(args.summary)
    summary, rows = load_buckets(summary_path)
    samples = load_samples(summary_path)

    model = args.model_name or summary.get("model") or "?"
    if not args.model_name and model == PLACEHOLDER_MODEL:
        print(f"note: the run recorded the placeholder model {PLACEHOLDER_MODEL!r}; "
              f"pass --model-name to label the figure with the real one")

    output = args.output
    if output is None:
        stem = summary_path.name
        if stem.endswith(".summary.json"):
            stem = stem[: -len(".summary.json")]
        else:
            stem = summary_path.stem
        output = summary_path.with_name(f"{stem}.token_f1_vs_length.png")

    labels = [str(bucket_sort_key(name)) for name, _ in rows]
    values = [row["metrics"]["token_f1"] for _, row in rows]
    counts = [row["count"] for _, row in rows]
    xs = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=args.dpi)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    left, right, top, bottom = 0.085, 0.975, 0.795, 0.235
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom)

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    ax.xaxis.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)

    # 4 px of rounding, converted from points into y-data units.
    axes_height_pt = fig.get_figheight() * (top - bottom) * 72.0
    radius = CORNER_RADIUS_PX / axes_height_pt  # y range is 0..1

    for x, value in zip(xs, values):
        ax.add_patch(
            PathPatch(
                rounded_top_path(x, 0.0, value, BAR_WIDTH, radius),
                facecolor=BAR,
                edgecolor="none",
                zorder=2,
            )
        )

    rng = np.random.default_rng(0)
    plotted_dots = False
    dot_counts: list[int | None] = []
    label_tops: list[float] = []
    for x, (bucket_name, row) in zip(xs, rows):
        value = row["metrics"]["token_f1"]
        points = samples.get(bucket_name)
        if not points:
            dot_counts.append(None)
            label_tops.append(value)
            continue
        plotted_dots = True
        dot_counts.append(len(points))
        jitter = rng.uniform(-1.0, 1.0, size=len(points)) * (BAR_WIDTH / 2.0 - 0.05)
        ax.scatter(
            x + jitter, points,
            s=13, color=DOT, alpha=0.9, linewidths=0.7, edgecolors=SURFACE, zorder=3,
        )
        # Labels clear the samples too - the buckets pile up at the 1.0 ceiling.
        label_tops.append(max(value, max(points)))

    # Direct labels - only six bars, so each one carries its value.
    label_ys = [top + 0.035 for top in label_tops]
    for x, value, label_y in zip(xs, values, label_ys):
        ax.text(
            x, label_y, f"{value:.3f}",
            ha="center", va="bottom", fontsize=10, color=INK, zorder=4,
        )

    ax.set_xlim(-0.62, len(rows) - 0.38)
    ax.set_ylim(0.0, max(1.08, max(label_ys) + 0.06))
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=10.5, color=INK_2)
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.set_yticklabels([f"{t:.1f}" for t in np.arange(0.0, 1.01, 0.2)],
                       fontsize=10, color=MUTED)
    ax.tick_params(axis="both", length=0, pad=6)
    ax.set_xlabel("Target token sequence length (repeat tokens)", fontsize=11, color=INK_2, labelpad=8)
    ax.set_ylabel("token_f1", fontsize=11, color=INK_2, labelpad=8)

    title = args.title or "token_f1 by target token sequence length"
    case_count = summary.get("num_completed", sum(counts))

    # Header stack, measured in points above the axes top: the model name gets its
    # own line - it is the first thing a reader needs to know about a run.
    def head_y(points: float) -> float:
        return 1.0 + points / axes_height_pt

    ax.set_title(
        title, loc="left", pad=42, fontsize=13.5, color=INK, fontweight="bold",
    )
    ax.text(
        0.0, head_y(22.0), model,
        transform=ax.transAxes, ha="left", va="bottom",
        fontsize=12, color=INK, fontweight="bold",
    )
    ax.text(
        0.0, head_y(6.0),
        f"{case_count} cases · instruction: {summary.get('repeat_instruction', '?')}",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=9.5, color=MUTED,
    )

    if plotted_dots:
        handles = [
            PathPatch(rounded_top_path(0, 0, 1, 1, 0.2), facecolor=BAR, edgecolor="none",
                      label="bucket mean"),
            plt.Line2D([], [], linestyle="none", marker="o", markersize=5,
                       markerfacecolor=DOT, markeredgecolor=SURFACE,
                       markeredgewidth=0.7, label="individual sample"),
        ]
        ax.legend(
            handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.135),
            ncol=2, frameon=False, fontsize=10, handlelength=1.1, handletextpad=0.5,
            columnspacing=1.6, labelcolor=INK_2,
        )

    caption = f"n = {counts[0]} per bucket" if len(set(counts)) == 1 else (
        "n per bucket: " + ", ".join(f"{lb} → {c}" for lb, c in zip(labels, counts))
    )
    # The summary is the source of truth for the bars; the dots come from the jsonl.
    gaps = [
        f"{lb}: summary n={c}, {d} plotted"
        for lb, c, d in zip(labels, counts, dot_counts)
        if d is not None and d != c
    ]
    if gaps:
        caption += " (from summary);  jsonl dots - " + "; ".join(gaps)
    fig.text(left, 0.03, caption, fontsize=9, color=MUTED, ha="left", va="bottom")

    fig.savefig(output, facecolor=SURFACE)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
