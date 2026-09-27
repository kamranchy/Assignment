"""Telomeric-repeat evidence (TTAGGG / CCCTAA) and assembly-end completeness.

Fusarium telomeres are (TTAGGG)n. On the top strand of a left chromosome end
the tract reads (CCCTAA)n; on a right end it reads (TTAGGG)n. Both
orientations are always counted.
"""
from __future__ import annotations

import re

G_RUN = re.compile(r"(?:TTAGGG){2,}")
C_RUN = re.compile(r"(?:CCCTAA){2,}")
G_UNIT = re.compile(r"TTAGGG")
C_UNIT = re.compile(r"CCCTAA")

# paper: reads with 10-30 consecutive TTAGGG motifs defined a complete end
MIN_TERMINAL_UNITS = 10


def unit_count(s: str) -> int:
    return len(G_UNIT.findall(s)) + len(C_UNIT.findall(s))


def longest_run(s: str) -> int:
    best = 0
    for rx in (G_RUN, C_RUN):
        for m in rx.finditer(s):
            best = max(best, len(m.group()) // 6)
    return best


def end_status(seq: str, window: int = 1000) -> dict:
    """Terminal telomere tract on each scaffold end (units in longest run)."""
    left = longest_run(seq[:window])
    right = longest_run(seq[-window:])
    return dict(left_units=left, right_units=right,
                left_complete=left >= MIN_TERMINAL_UNITS, right_complete=right >= MIN_TERMINAL_UNITS)


def flank_evidence(seq: str, start: int, end: int, strand: str, flank: int = 300) -> dict:
    """Telomeric units in the 5' and 3' flanks (element orientation). 1-based coords."""
    left = seq[max(0, start - 1 - flank): start - 1]
    right = seq[end: end + flank]
    f5, f3 = (left, right) if strand == "+" else (right, left)
    # immediately adjacent (<=30 bp) short motif: the paper reports a single
    # telomeric motif directly at the 5' junction and 3-7 copies at the 3' end
    adj5 = f5[-30:] if strand == "+" else f5[:30]
    adj3 = f3[:60] if strand == "+" else f3[-60:]
    return dict(tel5_units=unit_count(f5), tel3_units=unit_count(f3),
                tel5_run=longest_run(f5), tel3_run=longest_run(f3),
                tel5_adjacent=unit_count(adj5) > 0, tel3_adjacent=unit_count(adj3))


def snap_boundary(seq: str, pos: int, side: str, max_shift: int = 80) -> int:
    """Move an alignment boundary onto the edge of an adjacent telomeric
    repeat, mimicking the manual boundary refinement described in the paper.
    side='left' -> pos is an element start; 'right' -> element end."""
    if side == "left":
        w0 = max(0, pos - 1 - max_shift)
        win = seq[w0: pos - 1 + max_shift]
        ends = [w0 + m.end() for rx in (G_UNIT, C_UNIT) for m in rx.finditer(win)]
        ends = [e for e in ends if abs(e + 1 - pos) <= max_shift]
        return max(ends) + 1 if ends else pos
    w0 = max(0, pos - max_shift)
    win = seq[w0: pos + max_shift]
    starts = [w0 + m.start() for rx in (G_UNIT, C_UNIT) for m in rx.finditer(win)]
    starts = [s for s in starts if abs(s - pos) <= max_shift]
    return min(starts) if starts else pos


def extend_to_telomere(seq: str, pos: int, direction: str, max_dist: int) -> int:
    """Outward-only boundary extension: if the reference says bases are
    missing at this end (VNTR-rich 5' region differs between strains), move
    the boundary to the nearest telomeric motif within max_dist bp, because
    FoTeR junctions are defined by adjacent telomeric repeats (paper, Table 3).
    direction='left' extends a start leftwards; 'right' extends an end rightwards."""
    if max_dist <= 0:
        return pos
    if direction == "left":
        w0 = max(0, pos - 1 - max_dist)
        ends = [w0 + m.end() for rx in (G_UNIT, C_UNIT) for m in rx.finditer(seq[w0: pos - 1])]
        return max(ends) + 1 if ends else pos
    win = seq[pos: pos + max_dist]
    starts = [pos + m.start() for rx in (G_UNIT, C_UNIT) for m in rx.finditer(win)]
    return min(starts) if starts else pos
