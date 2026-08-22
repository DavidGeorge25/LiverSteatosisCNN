"""Verdicts: append-only, multi-reviewer, and the agreement statistics on top.

Verdicts do NOT live in the manifest. The manifest has one `verdict` column,
which can hold one pathologist's opinion, once -- and that forecloses the three
things this review round exists to measure:

  * two reviewers on the same candidate (inter-rater kappa),
  * one reviewer on the same candidate twice (intra-rater consistency),
  * a reviewer changing their mind, without erasing that they did.

So verdicts append to their own file, one row per judgment, never updated in
place. The manifest keeps its `verdict` column for the offline Excel route,
and `merge_into_manifest` folds the store back into it when someone wants a
single flat file.

Append-only also survives the failure mode that actually happens: the laptop
running the review app closes its lid mid-session. Every judgment up to that
point is already on disk, fsynced, in order.
"""

from __future__ import annotations

import csv
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

# `marks` is LAST on purpose: a verdicts.csv written before point-marking
# existed has nine columns, and DictReader fills the missing tenth with None
# rather than misaligning every row. Appending is the only schema change that
# keeps old sessions readable.
FIELDS = ("timestamp", "reviewer", "slide", "feature", "candidate_id",
          "verdict", "notes", "seconds", "pass_index", "marks")

YES = {"y", "yes", "1", "true", "confirm", "confirmed"}
NO = {"n", "no", "0", "false", "reject", "rejected"}
UNSURE = {"?", "u", "unsure", "unclear", "cannot tell"}

CONFIRMED = "y"
REJECTED = "n"
UNCERTAIN = "?"


def canonical(verdict: str) -> str:
    """Map any accepted spelling to y / n / ?. Unknown text stays as-is."""
    v = (verdict or "").strip().lower()
    if v in YES:
        return CONFIRMED
    if v in NO:
        return REJECTED
    if v in UNSURE:
        return UNCERTAIN
    return v


def append(
    path: str | Path,
    reviewer: str,
    slide: str,
    feature: str,
    candidate_id: str,
    verdict: str,
    notes: str = "",
    seconds: float = 0.0,
    timestamp: str = "",
    marks: str = "",
) -> None:
    """Record one judgment. Creates the file with a header if absent.

    `pass_index` counts how many times THIS reviewer has already judged THIS
    candidate, so a deliberate duplicate is identifiable as such rather than
    looking like a correction. It is computed here, at write time, because the
    caller cannot see the rest of the file.

    `marks` is "x y; x y" in fractions of the tile -- where the reviewer clicked
    the cells she is calling ballooned. Optional: a yes with no marks is a valid
    yes. Fractions rather than pixels so a mark survives the viewport it was
    made in, and so the merge can turn it into a slide coordinate later.

    Flushed and fsynced per row: a review session is at most a few hundred
    writes, and losing the last ten minutes to a buffer is not worth the speed.
    """
    path = Path(path)
    prior = 0
    if path.exists():
        for r in load(path):
            if r["reviewer"] == reviewer and r["candidate_id"] == candidate_id:
                prior += 1
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as fh:
            csv.writer(fh).writerow(FIELDS)

    if not timestamp:
        from datetime import datetime, timezone

        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with open(path, "a", newline="") as fh:
        csv.writer(fh).writerow([
            timestamp, reviewer, slide, feature, candidate_id,
            canonical(verdict), notes, f"{seconds:.1f}", prior, marks,
        ])
        fh.flush()
        os.fsync(fh.fileno())


def load(path: str | Path) -> list[dict[str, Any]]:
    """Every recorded judgment, in the order made."""
    path = Path(path)
    if not path.exists():
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def reviewers(rows: Sequence[dict[str, Any]]) -> list[str]:
    return sorted({r["reviewer"] for r in rows if r.get("reviewer")})


def latest(rows: Sequence[dict[str, Any]], reviewer: str) -> dict[str, str]:
    """One reviewer's current opinion per candidate -- last judgment wins."""
    out: dict[str, str] = {}
    for r in rows:
        if r.get("reviewer") == reviewer:
            out[r["candidate_id"]] = canonical(r.get("verdict", ""))
    return out


def first_pass(rows: Sequence[dict[str, Any]], reviewer: str) -> dict[str, str]:
    """One reviewer's FIRST opinion per candidate.

    Intra-rater consistency has to compare first against second. Using `latest`
    on both sides would compare a judgment with itself and report perfect
    agreement.
    """
    out: dict[str, str] = {}
    for r in rows:
        if r.get("reviewer") == reviewer and r["candidate_id"] not in out:
            out[r["candidate_id"]] = canonical(r.get("verdict", ""))
    return out


def cohen_kappa(a: dict[str, str], b: dict[str, str]) -> dict[str, Any]:
    """Cohen's kappa between two verdict maps, over the candidates they share.

    Reported twice on purpose. `kappa` treats "?" as its own category, which is
    the honest number -- refusing to call something IS a judgment, and two
    reviewers who both decline agree about something real. `kappa_yn` drops
    every pair where either said "?", which is the number comparable to the
    published inter-rater figures for NASH-CRN scoring, since those force a
    binary. Quoting one without the other overstates whichever is kinder.
    """
    shared = sorted(set(a) & set(b))
    out: dict[str, Any] = {"n_shared": len(shared)}
    if not shared:
        out.update(kappa=None, kappa_yn=None, agreement=None, note="no overlap")
        return out

    out.update(_kappa([(a[k], b[k]) for k in shared]))
    yn = [(a[k], b[k]) for k in shared
          if a[k] in (CONFIRMED, REJECTED) and b[k] in (CONFIRMED, REJECTED)]
    k_yn = _kappa(yn)
    out["kappa_yn"] = k_yn["kappa"]
    out["n_yn"] = len(yn)
    out["agreement_yn"] = k_yn["agreement"]
    return out


def _kappa(pairs: Sequence[tuple[str, str]]) -> dict[str, Any]:
    """Cohen's kappa for a list of (rater A, rater B) label pairs."""
    n = len(pairs)
    if n == 0:
        return {"kappa": None, "agreement": None, "n": 0}
    labels = sorted({v for p in pairs for v in p})
    observed = sum(1 for x, y in pairs if x == y) / n
    ca, cb = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    expected = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    if expected >= 1.0:
        # Both raters used exactly one label, and the same one. Agreement is
        # perfect and kappa is undefined (0/0) -- report that rather than a
        # divide-by-zero or a misleading 0.0.
        return {"kappa": None, "agreement": observed, "n": n,
                "note": "both raters used a single label; kappa undefined"}
    return {"kappa": (observed - expected) / (1 - expected),
            "agreement": observed, "n": n}


def intra_rater(rows: Sequence[dict[str, Any]], reviewer: str) -> dict[str, Any]:
    """Self-consistency: a reviewer's first vs second look at the same crop.

    Only candidates this reviewer judged more than once contribute. If the
    review set contained no deliberate duplicates, this is empty and says so
    -- which is itself the finding, because without duplicates there is no
    way to tell a low inter-rater kappa from ordinary human noise.
    """
    seen: dict[str, list[str]] = {}
    for r in rows:
        if r.get("reviewer") == reviewer:
            seen.setdefault(r["candidate_id"], []).append(
                canonical(r.get("verdict", ""))
            )
    repeats = {k: v for k, v in seen.items() if len(v) > 1}
    if not repeats:
        return {"n_repeated": 0, "kappa": None, "agreement": None,
                "note": "no candidate was shown to this reviewer twice"}
    pairs = [(v[0], v[1]) for v in repeats.values()]
    out = _kappa(pairs)
    out["n_repeated"] = len(repeats)
    return out


def summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Counts and per-reviewer pace, for the operator running the session."""
    out: dict[str, Any] = {"n_judgments": len(rows), "reviewers": {}}
    for who in reviewers(rows):
        mine = [r for r in rows if r["reviewer"] == who]
        counts = Counter(canonical(r.get("verdict", "")) for r in mine)
        secs = [float(r.get("seconds") or 0) for r in mine]
        timed = [s for s in secs if s > 0]
        out["reviewers"][who] = {
            "judgments": len(mine),
            "unique": len({r["candidate_id"] for r in mine}),
            "confirmed": counts.get(CONFIRMED, 0),
            "rejected": counts.get(REJECTED, 0),
            "uncertain": counts.get(UNCERTAIN, 0),
            "median_seconds": round(sorted(timed)[len(timed) // 2], 1) if timed else None,
        }
    return out


def agreement_report(path: str | Path) -> str:
    """Human-readable agreement block. What you read out after a session."""
    rows = load(path)
    if not rows:
        return "no verdicts recorded yet"
    who = reviewers(rows)
    s = summary(rows)
    lines = [f"{s['n_judgments']} judgments from {len(who)} reviewer(s)", ""]
    for name, st in s["reviewers"].items():
        pace = f", median {st['median_seconds']}s/candidate" if st["median_seconds"] else ""
        lines.append(
            f"  {name}: {st['unique']} candidates "
            f"(y={st['confirmed']} n={st['rejected']} ?={st['uncertain']}){pace}"
        )

    lines.append("")
    for name in who:
        intra = intra_rater(rows, name)
        if intra["n_repeated"]:
            k = intra["kappa"]
            lines.append(
                f"  intra-rater {name}: {intra['n_repeated']} repeated, "
                f"agreement {intra['agreement']:.0%}"
                + (f", kappa {k:.3f}" if k is not None else " (kappa undefined)")
            )
        else:
            lines.append(f"  intra-rater {name}: {intra['note']}")

    if len(who) > 1:
        lines.append("")
        for i, a in enumerate(who):
            for b in who[i + 1:]:
                r = cohen_kappa(latest(rows, a), latest(rows, b))
                if not r["n_shared"]:
                    lines.append(f"  {a} vs {b}: no shared candidates")
                    continue
                k, kyn = r.get("kappa"), r.get("kappa_yn")
                lines.append(
                    f"  {a} vs {b}: {r['n_shared']} shared, "
                    f"agreement {r['agreement']:.0%}"
                    + (f", kappa {k:.3f}" if k is not None else "")
                    + (f" (y/n only, n={r['n_yn']}: {kyn:.3f})"
                       if kyn is not None else "")
                )
    else:
        lines.append("")
        lines.append("  inter-rater: only one reviewer so far")
    return "\n".join(lines)


def merge_into_manifest(
    manifest_path: str | Path,
    verdicts_path: str | Path,
    out_path: str | Path | None = None,
    reviewer: str | None = None,
) -> Path:
    """Fold the verdict store back into a flat manifest.

    With one reviewer, or `reviewer` given, `verdict` is that person's latest
    call. With several and none named, `verdict` is filled only where every
    reviewer agrees; disagreements are left blank and flagged in `notes`,
    because a disputed candidate is exactly what must not silently become
    training data.
    """
    from . import manifest as manifest_mod

    rows = manifest_mod.read(manifest_path)
    v = load(verdicts_path)
    who = [reviewer] if reviewer else reviewers(v)
    maps = {w: latest(v, w) for w in who}

    for r in rows:
        cid = r["candidate_id"]
        opinions = {w: m[cid] for w, m in maps.items() if cid in m}
        if not opinions:
            continue
        distinct = set(opinions.values())
        if len(distinct) == 1:
            r["verdict"] = distinct.pop()
            r["reviewer"] = ";".join(sorted(opinions))
        else:
            r["verdict"] = ""
            r["reviewer"] = ";".join(sorted(opinions))
            r["notes"] = (r.get("notes") or "") + " DISAGREEMENT: " + ", ".join(
                f"{w}={x}" for w, x in sorted(opinions.items())
            )
    return manifest_mod.write(out_path or manifest_path, rows)
