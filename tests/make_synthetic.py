"""Build small synthetic 'genomes' to smoke-test the pipeline end to end.

!! These are SIMULATED data. Metrics obtained on them only prove the code
!! runs and behaves sensibly; they are NOT results for the assignment.

Each synthetic strain carries copies of its own published reference FoTeR
arranged like the paper describes (5' end toward the terminus, telomeric
motif at the 5' junction, 3-7 TAACCC units after the 3' end, 5'-truncated
tandem copies), plus decoys: a diverged RT-containing element and an
internal FoTeR fragment placed far from chromosome ends.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foterdet.reference import read_fasta, revcomp  # noqa: E402

rng = random.Random(7)


def rand_seq(n, gc=0.48):
    w = [(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2]
    return "".join(rng.choices("ACGT", weights=w, k=n))


def mutate(s, rate):
    s = list(s)
    for i in range(len(s)):
        if rng.random() < rate:
            s[i] = rng.choice("ACGT")
    return "".join(s)


def left_end_block(ref, n_copies, telomere=True):
    """Returns (sequence, list of (start0, end0, status)) in top-strand coords."""
    seq = ("CCCTAA" * 20) if telomere else rand_seq(300)
    els = []
    for k in range(n_copies):
        if k == 0:
            el, st = mutate(ref, 0.01), "full-length"
        else:
            keep = rng.randint(800, len(ref) - 400)
            el, st = mutate(ref[-keep:], 0.02), "truncated"
        seq += "TAACCC" if k == 0 else "CCCTAA"
        els.append((len(seq), len(seq) + len(el), st))
        seq += el + "TAACCC" * rng.randint(3, 7)
    return seq, els


def build(name, ref, out_fa, truth_rows, n_scaf=8, scaf_len=120_000):
    recs = {}
    for i in range(n_scaf):
        sc = f"{name}_chr{i + 1}"
        body = rand_seq(scaf_len)
        gt = []
        # decoy 1: diverged RT element internally (other RT-containing TE)
        if i == 1:
            mid = scaf_len // 2
            body = body[:mid] + mutate(ref, 0.30) + body[mid:]
        # decoy 2: internal FoTeR-like fragment far from ends
        if i == 3:
            mid = scaf_len // 3
            body = body[:mid] + mutate(ref[1500:2600], 0.05) + body[mid:]
        # left end
        if i % 2 == 0:
            blk, els = left_end_block(ref, rng.randint(1, 3), telomere=(i != 4))
            body = blk + body
            gt += [(a + 1, b, "+", s) for a, b, s in els]
        # right end (reverse-complemented left-end block)
        if i % 3 == 0:
            blk, els = left_end_block(ref, rng.randint(1, 2))
            rc = revcomp(blk)
            off = len(body)
            body = body + rc
            gt += [(off + len(blk) - b + 1, off + len(blk) - a, "-", s) for a, b, s in els]
        # an element cut by an incomplete assembly end (unresolved)
        if i == 5:
            cut = mutate(ref, 0.01)[: len(ref) // 2]
            body = revcomp(cut) + body   # 3' half missing at edge
            gt.append((1, len(cut), "-", "truncated"))
        recs[sc] = body
        for a, b, st, s in gt:
            truth_rows.append((name, sc, a, b, st, s))
    with open(out_fa, "w") as fh:
        for k, v in recs.items():
            fh.write(f">{k}\n")
            for j in range(0, len(v), 80):
                fh.write(v[j:j + 80] + "\n")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    outd = Path(sys.argv[1] if len(sys.argv) > 1 else root / "tests/synthetic")
    outd.mkdir(parents=True, exist_ok=True)
    refs = read_fasta(root / "data/reference/FoTeR_full_length_refs.fasta")
    truth, gl = [], []
    for name, ref in refs.items():
        fa = outd / f"SYN_{name}.fna"
        build(f"SYN_{name}", ref, fa, truth)
        gl.append((f"SYN_{name}", str(fa), name))
    with open(outd / "genomes.tsv", "w") as fh:
        fh.write("name\tfasta\town_ref\n" + "".join(f"{a}\t{b}\t{c}\n" for a, b, c in gl))
    with open(outd / "truth.tsv", "w") as fh:
        fh.write("genome\tscaffold\tstart\tend\tstrand\tstatus\n")
        fh.write("".join("\t".join(map(str, r)) + "\n" for r in truth))
    print(f"wrote {len(gl)} synthetic genomes, {len(truth)} simulated FoTeRs -> {outd}")
