"""Stage C: learned FoTeR classifier + Stage D: structural annotation.

Why not an end-to-end deep network?  The published ground truth contains
nine strain consensus elements and ~a few hundred copies that are highly
redundant within a strain (within-strain ORF identity up to 100%). The
*effective* number of independent training examples is therefore ~9, not
hundreds. A CNN/transformer on raw windows would memorise strain-specific
sequence and report inflated accuracy under random splits. We instead learn
a small, regularised model over biologically meaningful features and
validate it with leave-one-STRAIN-out (LOSO) cross-validation, with the
held-out strain's own reference also excluded from feature computation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import ALL_FEATURES, FEATURE_GROUPS, STATUS_FEATURES
from .telomere import extend_to_telomere, snap_boundary

SEQ_FEATURES = FEATURE_GROUPS["HOMOLOGY"] + FEATURE_GROUPS["SPECIFIC"] + FEATURE_GROUPS["COMPOS"]
CTX_FEATURES = FEATURE_GROUPS["CONTEXT"] + FEATURE_GROUPS["TELOMERE"]
END_TOL = 150        # bp of reference allowed missing at an end and still "complete"
JUNCTION_WINDOW = 300  # bp searched outward for the flanking telomeric motif
EDGE_TOL = 50        # element within this of a scaffold edge "runs into" the edge


# ---------------------------------------------------------------- labels
def overlap(a0, a1, b0, b1):
    return max(0, min(a1, b1) - max(a0, b0) + 1)


def label_candidates(feats: pd.DataFrame, truth: pd.DataFrame, min_frac: float = 0.5) -> pd.Series:
    """Candidate is positive if it overlaps a published FoTeR by >= min_frac of the shorter interval."""
    y = np.zeros(len(feats), int)
    full = np.full(len(feats), np.nan)
    for i, r in enumerate(feats.itertuples(index=False)):
        t = truth[(truth.genome == r.genome) & (truth.scaffold == r.scaffold)]
        for u in t.itertuples(index=False):
            ov = overlap(r.start, r.end, u.start, u.end)
            if ov >= min_frac * min(r.end - r.start + 1, u.end - u.start + 1):
                y[i] = 1
                full[i] = float(str(u.status).lower().startswith("full"))
                break
    return pd.Series(y, index=feats.index, name="label"), pd.Series(full, index=feats.index, name="gt_full")


# ---------------------------------------------------------------- models
def make_model(kind: str = "hgb"):
    if kind == "hgb":
        return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                              l2_regularization=1.0, min_samples_leaf=10,
                                              class_weight="balanced", random_state=0)
    if kind == "logreg":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=0.3, class_weight="balanced", max_iter=2000))
    raise ValueError(kind)


def best_f1_threshold(y, p):
    """F1-optimal cut placed midway between the chosen score and the next
    lower observed score (max-margin), so tied/saturated scores such as
    calibrated 1.0 do not produce a brittle threshold of exactly 1.0."""
    y = np.asarray(y); p = np.asarray(p, float)
    u = np.unique(np.round(p, 6))
    best = (0.5, -1.0, 0)
    for k, t in enumerate(u):
        pred = p >= t
        tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        if f1 > best[1]:
            best = (float(t), f1, k)
    t, _, k = best
    return float((t + u[k - 1]) / 2) if k > 0 else float(t)


def _fit_predict(Xtr, ytr, Xte, kind):
    if len(np.unique(ytr)) < 2:
        return np.full(len(Xte), float(ytr.mean()))
    m = make_model(kind).fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def loso_scores(df, y, features, kind="hgb", nested=True):
    """Outer leave-one-strain-out. The decision threshold for each outer fold
    is chosen by an INNER LOSO over the training strains only (no peeking)."""
    groups = df["genome"].to_numpy()
    X = df[features].astype(float).to_numpy()
    y = np.asarray(y)
    prob = np.zeros(len(df)); thr = np.zeros(len(df))
    for g in np.unique(groups):
        te, tr = groups == g, groups != g
        prob[te] = _fit_predict(X[tr], y[tr], X[te], kind)
        if nested:
            inner = np.zeros(tr.sum()); gtr = groups[tr]
            for h in np.unique(gtr):
                a, b = gtr == h, gtr != h
                inner[a] = _fit_predict(X[tr][b], y[tr][b], X[tr][a], kind)
            thr[te] = best_f1_threshold(y[tr], inner)
        else:
            thr[te] = 0.5
    return prob, thr


def loso_rule(df, y, score_fn):
    """Non-learned baselines: a single score with a threshold tuned on the
    training strains (e.g. a BLAST-like homology cut-off)."""
    groups = df["genome"].to_numpy(); s = score_fn(df).to_numpy(float); y = np.asarray(y)
    pred = np.zeros(len(df), bool)
    for g in np.unique(groups):
        te, tr = groups == g, groups != g
        t = best_f1_threshold(y[tr], s[tr])
        pred[te] = s[te] >= t
    return pred, s


BASELINES = {
    "B1_homology_only(BLAST-like)": lambda d: d["nt_score"] + d["prot_foter_score"],
    "B2_homology+near_end_rule": lambda d: (d["nt_score"] + d["prot_foter_score"]) * (d["dist_to_end"] <= 30000),
}


@dataclass
class Bundle:
    model: object
    seq_model: object
    status_model: object
    calibrator: object
    threshold: float
    uncertain_low: float
    features: list

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)


def fit_bundle(df, y, gt_full=None):
    """Fit a deployable model: inner LOSO over the given strains -> isotonic
    calibration + F1-optimal threshold, then refit on all given strains."""
    y = pd.Series(np.asarray(y))
    p_oof, _ = loso_scores(df, y, ALL_FEATURES, "hgb", nested=False)
    cal = IsotonicRegression(out_of_bounds="clip").fit(p_oof, y) if y.nunique() == 2 else None
    pc = cal.predict(p_oof) if cal is not None else p_oof
    thr = best_f1_threshold(y.values, pc)
    model = make_model("hgb").fit(df[ALL_FEATURES].astype(float).to_numpy(), y)
    seq_model = make_model("hgb").fit(df[SEQ_FEATURES].astype(float).to_numpy(), y)
    status_model = None
    if gt_full is not None:
        m = np.asarray(pd.notna(gt_full))
        yf = np.asarray(gt_full)[m]
        if m.sum() >= 10 and len(np.unique(yf)) == 2:
            status_model = make_model("hgb").fit(df.loc[m, STATUS_FEATURES].astype(float).to_numpy(), yf.astype(int))
    return Bundle(model, seq_model, status_model, cal, float(thr), float(max(0.05, thr / 2)), list(ALL_FEATURES))


def candidate_level_comparison(df, y, log=print):
    """LOSO comparison of models, baselines and feature-group ablations."""
    oof = df[["genome", "cand_id", "scaffold", "start", "end", "strand"]].copy()
    oof["label"] = np.asarray(y)
    for name, feats, kind in [("ML_full_HGB", ALL_FEATURES, "hgb"), ("ML_full_LogReg", ALL_FEATURES, "logreg"),
                              ("ML_sequence_only", SEQ_FEATURES, "hgb"), ("ML_context_only", CTX_FEATURES, "hgb")]:
        p, t = loso_scores(df, y, feats, kind)
        oof[f"p_{name}"] = p; oof[f"pred_{name}"] = p >= t
        log(f"  LOSO done: {name}")
    for g, cols in FEATURE_GROUPS.items():
        feats = [f for f in ALL_FEATURES if f not in cols]
        p, t = loso_scores(df, y, feats, "hgb")
        oof[f"pred_ablate_minus_{g}"] = p >= t
    for name, fn in BASELINES.items():
        pred, s = loso_rule(df, y, fn)
        oof[f"pred_{name}"] = pred; oof[f"p_{name}"] = s
    return oof


# ---------------------------------------------------------------- annotation
def annotate(df: pd.DataFrame, genome: dict, bundle: Bundle, med_orf_aa: float) -> pd.DataFrame:
    X = df[bundle.features].astype(float).to_numpy()
    raw = bundle.model.predict_proba(X)[:, 1] if len(df) else np.array([])
    conf = bundle.calibrator.predict(raw) if (bundle.calibrator is not None and len(df)) else raw
    pseq = bundle.seq_model.predict_proba(df[SEQ_FEATURES].astype(float).to_numpy())[:, 1] if len(df) else raw
    out = []
    for r, c, ps in zip(df.itertuples(index=False), conf, pseq):
        s = genome[r.scaffold]; L = len(s)
        a, b = int(r.start), int(r.end)
        has_ref = pd.notna(r.ref_len) and r.source == "nucleotide"
        miss5 = (r.ref_from - 1) if has_ref else np.nan
        miss3 = (r.ref_len - r.ref_to) if has_ref else np.nan
        # extend small unaligned ends, then snap onto adjacent telomeric repeats
        if has_ref:
            e5 = miss5 if miss5 <= END_TOL else 0
            e3 = miss3 if miss3 <= END_TOL else 0
            if r.strand == "+":
                a, b = a - int(e5), b + int(e3)
            else:
                a, b = a - int(e3), b + int(e5)
            a, b = max(1, a), min(L, b)
            a = snap_boundary(s, a, "left"); b = snap_boundary(s, b, "right")
            # larger unaligned 5'/3' ends: extend outward to an adjacent telomeric motif
            m5b = r.min_miss5 if pd.notna(r.min_miss5) else miss5
            m3b = r.min_miss3 if pd.notna(r.min_miss3) else miss3
            # the 5' ~150-250 bp are often unique to a strain (no other
            # reference covers them), so always search a JUNCTION_WINDOW
            lim5 = int(min(max(m5b + 50, JUNCTION_WINDOW), 1500))
            lim3 = int(min(max(m3b + 50, JUNCTION_WINDOW), 1500))
            if r.strand == "+":
                a = extend_to_telomere(s, a, "left", lim5)
                b = extend_to_telomere(s, b, "right", lim3)
            else:
                b = extend_to_telomere(s, b, "right", lim5)
                a = extend_to_telomere(s, a, "left", lim3)
        m5 = min(miss5, r.min_miss5) if has_ref and pd.notna(r.min_miss5) else miss5
        m3 = min(miss3, r.min_miss3) if has_ref and pd.notna(r.min_miss3) else miss3
        cov5 = has_ref and m5 <= END_TOL
        cov3 = has_ref and m3 <= END_TOL
        side5_edge = (a <= EDGE_TOL) if r.strand == "+" else (L - b <= EDGE_TOL)
        side3_edge = (L - b <= EDGE_TOL) if r.strand == "+" else (a <= EDGE_TOL)
        if not has_ref:
            status = "unresolved (protein-only evidence)"
        elif (not cov5 and side5_edge) or (not cov3 and side3_edge):
            status = "unresolved (runs into assembly end)"
        elif cov5 and cov3:
            status = "full-length" if r.longest_orf_aa >= 0.9 * med_orf_aa else "full-length span, ORF disrupted"
        else:
            status = "truncated (" + ("5'" if not cov5 else "") + ("+" if not cov5 and not cov3 else "") + ("3'" if not cov3 else "") + ")"
        status_rule = status
        if bundle.status_model is not None and has_ref and not status.startswith("unresolved"):
            pf = bundle.status_model.predict_proba(np.array([[getattr(r, f) for f in STATUS_FEATURES]], float))[0, 1]
            status = "full-length" if pf >= 0.5 else ("truncated" + (status[9:] if status.startswith("truncated") else ""))
        tel_ev = (f"5'flank {r.tel5_units}u/run{r.tel5_run}; 3'flank {r.tel3_units}u/run{r.tel3_run}; "
                  f"{r.nearest_end} end tract {r.end_tel_units}u ({'complete' if r.end_complete else 'INCOMPLETE'})")
        rt_ev = (f"FoTeR-RT HMM {r.prot_foter_score:.0f} bits; generic RT {r.prot_generic_rt_score:.0f}; "
                 f"REL motif {'yes' if r.rel_motif else 'no'}; longest ORF {r.longest_orf_aa} aa")
        tel_support = (r.tel5_units + r.tel3_units) > 0 or r.end_complete
        if c >= bundle.threshold:
            if r.end_complete or tel_support:
                call = "FoTeR"
            else:
                call = "FoTeR (telomere unresolved: assembly end incomplete)"
        elif c >= bundle.uncertain_low:
            call = "uncertain"
        elif ps >= 0.5 and r.dist_to_end > 50000:
            call = "FoTeR-like sequence, non-telomeric (rejected by context)"
        else:
            call = "non-FoTeR"
        rec = r._asdict()
        rec.update(pred_start=a, pred_end=b, pred_length=b - a + 1, status=status, status_rule=status_rule,
                   telomeric_repeat_evidence=tel_ev, rt_evidence=rt_ev,
                   confidence=round(float(c), 4), p_sequence_only=round(float(ps), 4), call=call)
        out.append(rec)
    return pd.DataFrame(out)


OUTPUT_COLUMNS = ["genome", "scaffold", "pred_start", "pred_end", "strand", "pred_length", "status",
                  "dist_to_end", "nearest_end", "telomeric_repeat_evidence", "rt_evidence", "confidence",
                  "call", "best_ref", "ref_from", "ref_to", "ref_len", "source", "p_sequence_only"]
