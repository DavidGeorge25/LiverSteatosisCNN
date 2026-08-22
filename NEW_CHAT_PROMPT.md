# Prompt for new chat

I'm building a weakly supervised deep learning pipeline to quantify hepatic
steatosis (fat) in H&E-stained mouse liver whole slide images, for a
computational pathology lab project (MASH mouse model). The dataset is
unannotated, so the approach is programmatic pseudo-labeling: classical CV
auto-generates approximate fat masks, then a U-Net trains on those noisy labels.
No model training yet — this phase is still the pseudo-labeling pipeline.

## What already exists

A working, tuned pipeline at `/Users/davidgeorge/Desktop/CompBio/Liver`.
**Read README.md first — it documents the current parameters and, more
importantly, why each one is what it is.** Do not rebuild it; extend it.

Environment is set up — **use `.venv/bin/python`, not system python**
(`brew install openslide` done; `.venv` on Homebrew Python 3.12 with
openslide-python, opencv, scikit-image, scipy, numpy, pandas, pyyaml).

```bash
.venv/bin/python -m mashpath.cli info   --slide <path.svs>
.venv/bin/python -m mashpath.cli tissue --slide <path.svs> --config configs/tuned.yaml
.venv/bin/python -m mashpath.cli steatosis    --slide <path.svs> --config configs/tuned.yaml --limit 100
.venv/bin/python -m mashpath.cli batch  --slide-dir <folder> --config configs/tuned.yaml --workers 5
.venv/bin/python -m mashpath.cli survey --group A=<dir> --group B=<dir> --tiles 60 --workers 8
.venv/bin/python -m mashpath.cli sweep  --group MASH=<dir> --group CCl4=<dir> --min-area 20,40,60 --ecc 0.75,0.80
```

Every threshold is a config parameter or CLI flag — nothing is hardcoded.
Filters are in physical units (µm²) converted via each slide's own MPP.
`--limit N` bounds work for fast iteration. Level 0 is never loaded whole; tiles
stream via `read_region`; peak RSS ~440 MB per slide.

## The data

| Folder | Cohort | n | Role |
| --- | --- | --- | --- |
| `data/R25-264_MASH_HE/` | R25-264, MASH | 14 | positives, 3.66–14.92% macro fat |
| `data/2026-04-20_CCl4_HE/` | R26-122, CCl4 | 38 | **negative control** (fibrosis model, no steatosis) |

The CCl4 cohort is what makes tuning quantitative rather than a matter of taste:
anything detected there is a false positive by construction. Keep using it that
way. `R26-122-14_HE.svs` is truncated (~281 KB short) and needs re-downloading;
all commands skip and report it rather than dying.

## State as of 2026-08-06

Both cohorts have been run end to end with `configs/tuned.yaml` (revision 2).
Full per-slide numbers are in README.md and `outputs/batch_MASH14_tuned2.csv`.

Revision 2 was retuned after 9 slides doubled the MASH cohort. The headline: the
previous parameters, tuned on 5 uniformly severe slides, were **missing large
obvious round droplets on mild slides**. The fix that mattered was adding
`max_eccentricity` — vessel lumens are elongated, and circularity (4πA/P²)
measures raggedness, not elongation, so it never caught them. Revision 2 beats
revision 1 on positive signal *and* false positive rate simultaneously
(separation 4.3:1 → 10.8:1).

Verified along the way:
- Tissue detection (Otsu on HSV saturation, level 2) holds unmodified across all
  51 readable slides, cutoff adapting 43→65.
- The 60-random-tile survey estimates full-slide fat to 3.2% mean relative error
  with 0.978 rank correlation — accurate enough to tune on, and ~100× cheaper.
- An "unstained-ness" (low saturation) criterion for separating fat from pale
  cytoplasm was tested and **does not work**; don't retry it without new
  evidence. CCl4 false positives are *less* saturated than real droplets on mild
  MASH slides.

## Known open problems — the real remaining work

**1. Adjacent droplets merge** into one component in dense regions. Since
revision 2 this no longer just inflates droplet size — a merged pair is
elongated, so `max_eccentricity` rejects it outright, which is most of the 6–7%
signal cost on severe slides. **Watershed splitting of touching droplets is the
highest-value next fix**: it would recover that area and fix droplet-count and
mean-size statistics at the same time.

**2. Microvesicular remains a noise floor.** 0.74% on MASH vs 0.53% on fat-free
CCl4 (1.4:1, against 55:1 for macro), and the highest single micro value in
either cohort belongs to a CCl4 slide. It is measured but excluded from labels
(`include_microvesicular: false`). Keep it that way until it is redefined —
probably needs a different feature than "small round white thing".

**3. No pseudo-labels have been exported yet.** All runs so far used
`--no-save-tiles`. Dropping that flag writes `tiles/*.png` + `masks/*_mask.png`,
which is the actual U-Net training set. Budget disk: ~42,000 tiles for the MASH
cohort.

**4. `exclude_border`** is off and should stay off (11–16% of fat area touches a
tile edge). Overlapping tiles via `--stride < --tile-size` is the better fix and
is already wired up but has not been evaluated.

**5. The false-positive floor is 0.17% mean / 0.44% worst case** (full runs). Treat any
slide-level call below ~0.5% as indistinguishable from zero.

## How I like to work

Don't tune parameters silently — show me what changed and what it did to the
numbers, on both cohorts. Visual QC is what I trust: give me overlays at native
resolution, not just downscaled contact sheets. Per-tile QC at
`outputs/<slide>/qc/tiles/*.png` is more readable than the contact sheet, and an
A/B diff overlay (what a change adds vs removes, in different colors) is the
single most useful figure — see `outputs/qc_compare/` for the format.

I'd rather have an honest macro-only number than an inflated total that includes
texture noise.
