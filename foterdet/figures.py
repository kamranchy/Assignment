"""Figures for the report."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from .telomere import G_UNIT, C_UNIT


def pr_curves(oof, out):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    y = oof.label.to_numpy()
    for c in [c for c in oof.columns if c.startswith("p_")]:
        s = oof[c].to_numpy(float)
        if len(np.unique(y)) < 2:
            continue
        pr, rc, _ = precision_recall_curve(y, s)
        ax.plot(rc, pr, label=f"{c[2:]} (AP={average_precision_score(y, s):.2f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Candidate-level PR, leave-one-strain-out")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


def bars(df, x, cols, out, title):
    fig, ax = plt.subplots(figsize=(max(5, 0.7 * len(df)), 4))
    w = 0.8 / len(cols); idx = np.arange(len(df))
    for k, c in enumerate(cols):
        ax.bar(idx + k * w, df[c], w, label=c)
    ax.set_xticks(idx + w * (len(cols) - 1) / 2); ax.set_xticklabels(df[x], rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_title(title); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


def boundary_hist(tp, out):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    if len(tp):
        lim = 500
        ax.hist(np.clip(tp.d_start, -lim, lim), bins=40, alpha=.6, label="start")
        ax.hist(np.clip(tp.d_end, -lim, lim), bins=40, alpha=.6, label="end")
    ax.set_xlabel("predicted - published (bp, clipped ±500)"); ax.set_ylabel("elements"); ax.legend()
    ax.set_title("Boundary agreement"); fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


def feature_distributions(df, y, out, feats=("log10_dist_to_end", "foter_minus_generic", "ref_cov_frac",
                                               "tel3_units", "end_tel_units", "orient_consistent")):
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    for ax, f in zip(axes.ravel(), feats):
        for lab, name in [(0, "non-FoTeR"), (1, "published FoTeR")]:
            v = df.loc[np.asarray(y) == lab, f].astype(float).dropna()
            if len(v):
                ax.hist(v, bins=30, alpha=.6, density=True, label=name)
        ax.set_title(f, fontsize=9)
    axes[0, 0].legend(fontsize=7); fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


def locus_plot(genome, ann, scaffold, end, out, span=40000):
    s = genome[scaffold]; L = len(s)
    a, b = (0, min(L, span)) if end == "left" else (max(0, L - span), L)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 3.2), sharex=True, gridspec_kw=dict(height_ratios=[2, 1]))
    colors = {"full-length": "tab:blue", "truncated": "tab:orange", "unresolved": "tab:gray"}
    sub = ann[(ann.scaffold == scaffold) & (ann.pred_end >= a) & (ann.pred_start <= b)]
    for r in sub.itertuples(index=False):
        col = next((v for k, v in colors.items() if r.status.startswith(k)), "tab:purple")
        alpha = 1.0 if r.call.startswith("FoTeR") else 0.25
        ax1.add_patch(plt.Rectangle((r.pred_start, 0.3), r.pred_end - r.pred_start, 0.4, color=col, alpha=alpha))
        ax1.annotate("", xy=(r.pred_end if r.strand == "+" else r.pred_start, 0.85),
                     xytext=(r.pred_start if r.strand == "+" else r.pred_end, 0.85),
                     arrowprops=dict(arrowstyle="->", color=col))
        ax1.text((r.pred_start + r.pred_end) / 2, 0.05, f"{r.confidence:.2f}", ha="center", fontsize=7)
    ax1.set_ylim(0, 1); ax1.set_yticks([]); ax1.set_title(f"{scaffold} {end} end — blue full, orange truncated, grey unresolved; faded = rejected")
    w = 200; xs = np.arange(a, b, w)
    dens = [len(G_UNIT.findall(s[x:x + w])) + len(C_UNIT.findall(s[x:x + w])) for x in xs]
    ax2.bar(xs, dens, width=w, color="k"); ax2.set_ylabel("TTAGGG\nunits/200bp", fontsize=7); ax2.set_xlabel("position (bp)")
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


def confidence_hist(pred, out):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for g, d in pred.groupby("genome"):
        ax.hist(d.confidence, bins=20, histtype="step", label=g)
    ax.set_xlabel("calibrated confidence"); ax.set_ylabel("candidates"); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)
