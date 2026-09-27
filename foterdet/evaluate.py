"""Element-level comparison of automated predictions with the published FoTeRs.

No manual correction is applied to predictions before scoring. The FP/FN
'reason' columns are produced automatically from the model's own features
(they are diagnostics, not corrections).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .model import overlap

POSITIVE_CALLS = ("FoTeR", "FoTeR (telomere unresolved: assembly end incomplete)")


def match(pred: pd.DataFrame, truth: pd.DataFrame, min_frac: float = 0.5):
    pairs, fp, fn = [], [], []
    for g in sorted(set(pred.genome) | set(truth.genome)):
        P = pred[pred.genome == g].reset_index(drop=True)
        T = truth[truth.genome == g].reset_index(drop=True)
        cand = []
        for i, p in P.iterrows():
            for j, t in T[T.scaffold == p.scaffold].iterrows():
                ov = overlap(p.pred_start, p.pred_end, t.start, t.end)
                if ov >= min_frac * min(p.pred_end - p.pred_start + 1, t.end - t.start + 1):
                    cand.append((ov, i, j))
        used_p, used_t = set(), set()
        for ov, i, j in sorted(cand, reverse=True):
            if i in used_p or j in used_t:
                continue
            used_p.add(i); used_t.add(j)
            pairs.append((P.loc[i], T.loc[j]))
        fp += [P.loc[i] for i in P.index if i not in used_p]
        fn += [T.loc[j] for j in T.index if j not in used_t]
    return pairs, pd.DataFrame(fp), pd.DataFrame(fn)


def prf(tp, fp, fn):
    p = tp / max(1, tp + fp); r = tp / max(1, tp + fn)
    return p, r, 2 * p * r / max(1e-9, p + r)


def fp_reason(r) -> str:
    if r.prot_generic_rt_score > r.prot_foter_score and r.prot_generic_rt_score > 0:
        return "other RT-containing element (generic RT > FoTeR-RT score)"
    if r.dist_to_end > 50000:
        return "internal locus with FoTeR similarity"
    if r.ref_cov_frac < 0.2:
        return "weak/partial similarity hit"
    if (r.tel5_units + r.tel3_units) > 0 or r.end_complete:
        return "telomere-associated FoTeR-like copy absent from published table (candidate novel/unlisted copy)"
    return "near-end hit without telomere evidence"


def fn_reason(t, all_cands: pd.DataFrame, genome_scaffolds: dict) -> str:
    if t.scaffold not in genome_scaffolds.get(t.genome, set()):
        return "scaffold not in public assembly (end likely extended with PacBio in the paper)"
    c = all_cands[(all_cands.genome == t.genome) & (all_cands.scaffold == t.scaffold)]
    ov = [overlap(a, b, t.start, t.end) for a, b in zip(c.pred_start, c.pred_end)] if len(c) else []
    if not ov or max(ov) == 0:
        return "not generated as candidate (stage A miss: too diverged/short for minimap2 & no RT ORF)"
    k = int(np.argmax(ov)); r = c.iloc[k]
    return f"candidate scored below threshold (confidence {r.confidence:.2f}; call={r.call})"


def evaluate(annotated: pd.DataFrame, truth: pd.DataFrame, genome_scaffolds: dict):
    pred = annotated[annotated.call.isin(POSITIVE_CALLS)].copy()
    pairs, fp, fn = match(pred, truth)
    rows = []
    for p, t in pairs:
        rows.append(dict(genome=t.genome, scaffold=t.scaffold, gt_start=t.start, gt_end=t.end,
                         gt_status=t.status, pred_start=p.pred_start, pred_end=p.pred_end,
                         pred_status=p.status, pred_status_rule=p.status_rule, d_start=p.pred_start - t.start, d_end=p.pred_end - t.end,
                         confidence=p.confidence, call=p.call))
    tp_df = pd.DataFrame(rows)
    if len(fp):
        fp = fp.assign(reason=[fp_reason(r) for r in fp.itertuples(index=False)])
    if len(fn):
        fn = fn.assign(reason=[fn_reason(t, annotated, genome_scaffolds) for t in fn.itertuples(index=False)])
    per = []
    for g in sorted(truth.genome.unique()):
        tp = int((tp_df.genome == g).sum()) if len(tp_df) else 0
        f_p = int((fp.genome == g).sum()) if len(fp) else 0
        f_n = int((fn.genome == g).sum()) if len(fn) else 0
        p, r, f = prf(tp, f_p, f_n)
        per.append(dict(genome=g, published=int((truth.genome == g).sum()), recovered_TP=tp, missed_FN=f_n,
                        additional_FP=f_p, precision=p, recall=r, F1=f))
    per = pd.DataFrame(per)
    tot = per[["published", "recovered_TP", "missed_FN", "additional_FP"]].sum()
    p, r, f = prf(tot.recovered_TP, tot.additional_FP, tot.missed_FN)
    per.loc[len(per)] = dict(genome="ALL (micro)", published=tot.published, recovered_TP=tot.recovered_TP,
                             missed_FN=tot.missed_FN, additional_FP=tot.additional_FP, precision=p, recall=r, F1=f)
    summary = dict(micro_precision=p, micro_recall=r, micro_F1=f)
    if len(tp_df):
        tp_df["gt_full"] = tp_df.gt_status.str.lower().str.startswith("full")
        tp_df["pred_full"] = tp_df.pred_status.str.startswith("full-length")
        summary.update(
            median_abs_start_err_bp=float(tp_df.d_start.abs().median()),
            median_abs_end_err_bp=float(tp_df.d_end.abs().median()),
            frac_boundaries_within_50bp=float(((tp_df.d_start.abs() <= 50) & (tp_df.d_end.abs() <= 50)).mean()),
            full_vs_truncated_accuracy_learned=float((tp_df.gt_full == tp_df.pred_full).mean()),
            full_vs_truncated_accuracy_rule=float((tp_df.gt_full == tp_df.pred_status_rule.str.startswith("full-length")).mean()),
            n_pred_unresolved_among_TP=int(tp_df.pred_status.str.startswith("unresolved").sum()),
        )
    return per, tp_df, fp, fn, summary


def candidate_metrics(oof: pd.DataFrame) -> pd.DataFrame:
    y = oof.label.to_numpy()
    rows = []
    for c in [c for c in oof.columns if c.startswith("pred_")]:
        pr = oof[c].to_numpy(bool)
        tp = int((pr & (y == 1)).sum()); fp = int((pr & (y == 0)).sum()); fn = int((~pr & (y == 1)).sum())
        p, r, f = prf(tp, fp, fn)
        rows.append(dict(method=c.replace("pred_", ""), TP=tp, FP=fp, FN=fn, precision=p, recall=r, F1=f))
    return pd.DataFrame(rows).sort_values("F1", ascending=False)
