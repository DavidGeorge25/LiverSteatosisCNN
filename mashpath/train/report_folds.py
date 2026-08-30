"""Collect the leave-one-batch-out folds into one table.

    python -m mashpath.train.report_folds --runs $MASHPATH_OUT/unet/lobo

Reads each fold's `summary.json`. Prints the worst batch before the mean,
because that is the number that predicts what happens when the lab stains a new
run next month -- a model at 0.9 on seven batches and 0.55 on the eighth has a
mean of 0.85, reads as a good model, and will fail. `evaluate.report()` makes
the same argument for its own table; this one applies it to the three numbers
this project actually cares about.

A missing fold is reported as missing, never skipped. A sweep that silently
drops the fold that crashed reports the mean of the folds that were easy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def collect(runs_dir: str | Path, expected: list[str] | None = None
            ) -> tuple[pd.DataFrame, list[str]]:
    runs = Path(runs_dir)
    rows, missing = [], []
    found = {}
    for d in sorted(p for p in runs.iterdir() if p.is_dir()):
        s = d / "summary.json"
        if not s.exists():
            missing.append(d.name)
            continue
        j = json.loads(s.read_text())
        f, e = j.get("floor", {}), j.get("external", {})
        found[j.get("held_out_batch", d.name)] = True
        rows.append({
            "held_out_batch": j.get("held_out_batch", d.name),
            "dice_vs_teacher": j.get("dice_vs_teacher"),
            "floor_mean": f.get("model_mean"),
            "floor_worst": f.get("model_worst"),
            "teacher_floor_mean": f.get("teacher_mean"),
            "external_auc": e.get("model_auc"),
            "teacher_auc": e.get("teacher_auc"),
            "chow_max": e.get("model_chow_max"),
            "nash_min": e.get("model_nash_min"),
        })
    for b in (expected or []):
        if b not in found:
            missing.append(b)
    return pd.DataFrame(rows), sorted(set(missing))


def report(df: pd.DataFrame, missing: list[str]) -> str:
    if df.empty:
        return "no folds found\n"
    d = df.copy()
    for c in ("floor_mean", "floor_worst", "teacher_floor_mean", "chow_max",
              "nash_min"):
        if c in d:
            d[c] = d[c] * 100
    d["gap"] = d["nash_min"] / d["chow_max"]

    L = ["# Leave-one-batch-out", "",
         "| held-out batch | dice vs teacher | floor mean % | floor worst % | "
         "external AUC | chow max % | NASH min % | gap |",
         "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for _, r in d.sort_values("external_auc").iterrows():
        L.append(
            f"| {r['held_out_batch']} | {r['dice_vs_teacher']:.3f} | "
            f"{r['floor_mean']:.3f} | {r['floor_worst']:.3f} | "
            f"{r['external_auc']:.3f} | {r['chow_max']:.3f} | "
            f"{r['nash_min']:.2f} | {r['gap']:.0f}× |")

    worst_auc = d.loc[d["external_auc"].idxmin()]
    worst_floor = d.loc[d["floor_worst"].idxmax()]
    L += ["",
          f"**Worst external AUC: {worst_auc['held_out_batch']} at "
          f"{worst_auc['external_auc']:.3f}.** Mean {d['external_auc'].mean():.3f}, "
          f"spread {d['external_auc'].max() - d['external_auc'].min():.3f}.",
          "",
          f"**Worst false-positive floor: {worst_floor['held_out_batch']} at "
          f"{worst_floor['floor_worst']:.3f}%** (teacher's worst on that cohort "
          f"was 0.445%). Mean floor {d['floor_mean'].mean():.3f}%.",
          "",
          "The mean is worth quoting only if the spread is small. A wide spread "
          "means performance depends on which staining run the tiles came from, "
          "which is the failure this evaluation exists to make visible."]
    if missing:
        L += ["", f"**{len(missing)} fold(s) missing a summary: "
                  f"{', '.join(missing)}.** Not skipped -- a sweep that drops "
                  f"the fold that crashed reports the mean of the folds that "
                  f"were easy."]
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--runs", required=True, help="directory of per-fold run dirs")
    p.add_argument("--batches", default=None,
                   help="batches.txt; folds named here but missing are reported")
    p.add_argument("--out", default=None, help="write the markdown here too")
    a = p.parse_args(argv)
    expected = (Path(a.batches).read_text().split("\n") if a.batches else [])
    expected = [b for b in expected if b.strip()]
    df, missing = collect(a.runs, expected)
    txt = report(df, missing)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt)
        df.to_csv(Path(a.out).with_suffix(".csv"), index=False)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
