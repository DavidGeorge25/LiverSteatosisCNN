"""The tile labelling app: one field at a time, one binary question.

Sibling of `review/app.py`, not a mode of it. The candidate app asks "is this
outlined cell ballooned?" and its whole interface is built around a proposal --
an outline to toggle, the measurements that produced it, a score band. None of
that exists here. The question is about the FIELD, the detector does not appear
in it, and every affordance that would let its opinion leak through has been
removed rather than hidden:

  * the slide name is never shown. It is a strong cue -- three tiles into a
    CCl4 section a reviewer knows which section she is in, and from then on she
    is partly labelling the slide rather than the tile;
  * the score, the band and the train/test split live in a different file that
    the server does not read;
  * presentation order is fixed in the manifest and never re-sorted, so a
    repeat cannot be recognised by arriving next to its original.

WHAT IT ADDS over the candidate app, because the judgment is different:

  * ZOOM AND PAN. "Rarefied, flocculent cytoplasm" is a call about texture
    inside one cell. A fixed 512 px thumbnail cannot carry it.
  * A CONTEXT VIEW at 3x3 tiles, always on screen and expandable to the full
    stage. Ballooning is defined RELATIVE to the neighbouring hepatocytes, so a
    tile judged in isolation is a different and harder question than the one
    the grading criteria describe.

Serve only to localhost, for the same reason as the candidate app: the crops
are patient-adjacent material and this server has no authentication by design.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from . import verdicts as verdicts_mod
# From `names`, NOT from `tileset`: importing tileset here would pull
# pandas, numpy and OpenSlide in behind three string constants, and
# this module is meant to run standalone on a reviewer's machine.
from .names import FRAME_NAME, MANIFEST_NAME, VERDICTS_NAME

FEATURE = "ballooning_tile"
_CTYPE = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


class TileSet:
    """One labelling package: its manifest, its images, its verdict store."""

    def __init__(self, root: str | Path):
        self.dir = Path(root).resolve()
        path = self.dir / MANIFEST_NAME
        if not path.exists():
            raise FileNotFoundError(
                f"no {MANIFEST_NAME} in {self.dir} -- build one with "
                "`python -m mashpath.review.tileset`"
            )
        with open(path, newline="") as fh:
            self.rows = list(csv.DictReader(fh))
        if not self.rows:
            raise ValueError(f"{path} has no rows")
        missing = [c for c in ("tile_id", "slide", "image", "context")
                   if c not in self.rows[0]]
        if missing:
            raise ValueError(f"{path} is missing column(s) {missing}")
        # Presentation order is a property of the SET, not of the file system.
        # Sorting here would break the duplicate spacing the sampler placed.
        self.rows.sort(key=lambda r: int(r.get("order") or 0))
        self.verdicts_path = self.dir / VERDICTS_NAME

    @property
    def name(self) -> str:
        return self.dir.name

    def image_path(self, index: int, kind: str) -> Path | None:
        row = self.rows[index]
        rel = row.get("context" if kind == "context" else "image") or ""
        if not rel:
            return None
        p = (self.dir / rel).resolve()
        # `rel` comes out of a CSV, which is a file a human may have edited.
        if not str(p).startswith(str(self.dir)):
            return None
        return p if p.exists() else None


def discover(roots: list[str | Path]) -> list[TileSet]:
    """Every tile package under the given roots, plus the roots themselves."""
    found: list[TileSet] = []
    seen: set[Path] = set()
    for root in roots:
        root = Path(root)
        cands = [root] if (root / MANIFEST_NAME).exists() else []
        cands += [p.parent for p in sorted(root.glob(f"*/{MANIFEST_NAME}"))]
        for d in cands:
            d = d.resolve()
            if d in seen:
                continue
            seen.add(d)
            try:
                found.append(TileSet(d))
            except (ValueError, FileNotFoundError) as exc:
                print(f"  skipping {d}: {exc}")
    return found


# ---- agreement, keyed on the repeat relation -------------------------------


def agreement_report(package: str | Path) -> str:
    """Counts, pace, and intra-rater agreement over the deliberate repeats.

    Repeats are paired through the manifest's `dup_of`, NOT through a tile_id
    seen twice in the verdict store. Those are different events: `dup_of` is a
    tile the sampler deliberately showed again hundreds of presentations later,
    while a repeated tile_id is the reviewer pressing back to change her mind.
    Counting the correction as a disagreement would make careful reviewing look
    like inconsistency.
    """
    d = Path(package)
    rows = verdicts_mod.load(d / VERDICTS_NAME)
    if not rows:
        return "no verdicts recorded yet"

    with open(d / MANIFEST_NAME, newline="") as fh:
        manifest = list(csv.DictReader(fh))
    dup_of = {r["tile_id"]: r.get("dup_of", "") for r in manifest}

    out: list[str] = [f"{len(rows)} judgments"]
    for who in verdicts_mod.reviewers(rows):
        mine = [r for r in rows if r["reviewer"] == who]
        first = verdicts_mod.first_pass(mine, who)
        latest = verdicts_mod.latest(mine, who)
        counts = Counter(latest.values())
        secs = sorted(float(r.get("seconds") or 0) for r in mine
                      if float(r.get("seconds") or 0) > 0)
        med = f", median {secs[len(secs) // 2]:.1f}s/tile" if secs else ""
        out.append(
            f"  {who}: {len(latest)} tiles "
            f"(y={counts.get('y', 0)} n={counts.get('n', 0)} "
            f"?={counts.get('?', 0)}){med}"
        )
        pairs = [(first[tid], first[orig])
                 for tid, orig in dup_of.items()
                 if orig and tid in first and orig in first]
        if not pairs:
            out.append("    intra-rater: no repeat has been reached yet")
            continue
        agree = sum(1 for a, b in pairs if a == b) / len(pairs)
        k = verdicts_mod._kappa(pairs)
        out.append(
            f"    intra-rater: {len(pairs)} repeat pair(s), "
            f"agreement {agree:.0%}"
            + (f", kappa {k['kappa']:.3f}" if k.get("kappa") is not None
               else " (kappa undefined -- one label used throughout)")
        )

    who = verdicts_mod.reviewers(rows)
    if len(who) > 1:
        out.append("")
        for i, a in enumerate(who):
            for b in who[i + 1:]:
                r = verdicts_mod.cohen_kappa(
                    verdicts_mod.latest(rows, a), verdicts_mod.latest(rows, b))
                if not r["n_shared"]:
                    out.append(f"  {a} vs {b}: no shared tiles")
                    continue
                out.append(
                    f"  {a} vs {b}: {r['n_shared']} shared, "
                    f"agreement {r['agreement']:.0%}"
                    + (f", kappa {r['kappa']:.3f}"
                       if r.get("kappa") is not None else "")
                )
    return "\n".join(out)


# ---- server ---------------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    sets: list[TileSet] = []

    def log_message(self, fmt, *args):  # noqa: A003
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj: Any, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self) -> None:  # noqa: N802
        from urllib.parse import parse_qs, urlparse

        u = urlparse(self.path)
        q = parse_qs(u.query)

        if u.path == "/":
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")

        elif u.path == "/api/sets":
            self._json([{"index": i, "name": s.name, "n": len(s.rows)}
                        for i, s in enumerate(self.sets)])

        elif u.path == "/api/tiles":
            s = self.sets[int(q.get("set", ["0"])[0])]
            reviewer = q.get("reviewer", [""])[0]
            judged = {r["candidate_id"] for r in verdicts_mod.load(s.verdicts_path)
                      if r.get("reviewer") == reviewer}
            self._json({
                "name": s.name,
                "tiles": [
                    # Order and geometry ONLY. No slide, no score, no band,
                    # and deliberately not the tile_id either: the id encodes
                    # the slide name (`R26-122-23_HE_91_x026112_y014848`), so
                    # shipping it would put the cohort one devtools panel away
                    # from a reviewer we are asking to judge blind. The page
                    # never needed it -- verdicts post by `index`, and the
                    # server maps that back to the id when it writes the row.
                    {"index": i,
                     "tile_um": r.get("tile_um", ""),
                     "context_um": r.get("context_um", ""),
                     "box": r.get("core_box", ""),
                     "judged": r["tile_id"] in judged}
                    for i, r in enumerate(s.rows)
                ],
            })

        elif u.path.startswith("/api/image/"):
            _, _, _, si, idx, kind = u.path.split("/", 5)
            p = self.sets[int(si)].image_path(int(idx), kind)
            if p is None:
                self._send(404, b"no image", "text/plain")
                return
            self._send(200, p.read_bytes(),
                       _CTYPE.get(p.suffix.lower(), "application/octet-stream"))

        elif u.path == "/api/stats":
            s = self.sets[int(q.get("set", ["0"])[0])]
            self._send(200, agreement_report(s.dir).encode(),
                       "text/plain; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        from urllib.parse import urlparse

        if urlparse(self.path).path != "/api/verdict":
            self._send(404, b"not found", "text/plain")
            return
        n = int(self.headers.get("Content-Length", 0))
        data = json.loads(self.rfile.read(n) or b"{}")
        s = self.sets[int(data["set"])]
        row = s.rows[int(data["index"])]
        verdicts_mod.append(
            s.verdicts_path,
            reviewer=data.get("reviewer", "").strip() or "anonymous",
            slide=row["slide"], feature=FEATURE, candidate_id=row["tile_id"],
            verdict=data.get("verdict", ""), notes=data.get("notes", ""),
            seconds=float(data.get("seconds", 0) or 0),
            marks=str(data.get("marks", "") or ""),
        )
        self._json({"ok": True})


def serve(roots: list[str | Path], port: int = 8000,
          host: str = "127.0.0.1") -> None:
    """Start the tile review server. Blocks until interrupted."""
    sets = discover(roots)
    if not sets:
        raise SystemExit(
            f"no tile packages found under {[str(r) for r in roots]}.\n"
            f"Expected a directory containing {MANIFEST_NAME} -- build one "
            "with `python -m mashpath.review.tileset`."
        )
    _Handler.sets = sets
    print(f"tile review: {len(sets)} set(s)")
    for s in sets:
        done = len({r["candidate_id"] for r in verdicts_mod.load(s.verdicts_path)})
        print(f"  {s.name:32s} {len(s.rows):5d} presentations"
              + (f"  ({done} already judged)" if done else ""))
    print(f"\n  open  http://{host}:{port}/\n")
    print(f"  verdicts append to {VERDICTS_NAME} inside each package")
    print(f"  scores/bands/splits stay in {FRAME_NAME} and are never served")
    print("  Ctrl-C to stop\n")

    srv = ThreadingHTTPServer((host, port), _Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        srv.server_close()
        for s in sets:
            rep = agreement_report(s.dir)
            if "no verdicts" not in rep:
                print(f"\n--- {s.name} ---\n{rep}")


def _main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="python -m mashpath.review.tiles_app",
        description="Tile-level binary labelling app for hepatocyte ballooning.",
    )
    p.add_argument("roots", nargs="+", help="tile package directory/directories")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="127.0.0.1",
                   help="leave as localhost; the server has no authentication")
    a = p.parse_args(argv)
    serve(a.roots, a.port, a.host)
    return 0


# ---- the page -------------------------------------------------------------
# One file, no external assets: this has to work with no network, because that
# is the situation on a cluster login node behind an SSH tunnel.

PAGE = r"""<!doctype html>
<meta charset="utf-8">
<title>ballooning — tile labelling</title>
<style>
  :root {
    --bg:#12141a; --panel:#1b1f27; --line:#2c3340; --fg:#e8ecf2; --dim:#8a94a6;
    --yes:#2ea36b; --no:#c8503f; --maybe:#c99a2e; --accent:#4d8fd6;
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg); height:100vh;
         display:flex; flex-direction:column; overflow:hidden;
         font:14px/1.45 ui-sans-serif,system-ui,-apple-system,sans-serif; }
  header { display:flex; align-items:center; gap:16px; padding:8px 14px;
           background:var(--panel); border-bottom:1px solid var(--line);
           flex:0 0 auto; }
  header b { font-weight:600; }
  .dim { color:var(--dim); }
  .grow { flex:1; }
  main { flex:1; display:flex; min-height:0; }
  #stage { flex:1; position:relative; overflow:hidden; background:#0c0e12;
           cursor:grab; }
  #stage.drag { cursor:grabbing; }
  #marks { position:absolute; inset:0; pointer-events:none; }
  .mk { position:absolute; width:30px; height:30px; margin:-15px 0 0 -15px;
        border-radius:50%; border:2.5px solid var(--accent);
        background:rgba(60,180,190,.16);
        box-shadow:0 0 0 1.5px rgba(255,255,255,.5); }
  .mk b { position:absolute; inset:0; display:grid; place-items:center;
          font-size:11px; color:#fff; text-shadow:0 1px 2px #000; }
  #stage img { position:absolute; top:50%; left:50%; transform-origin:0 0;
               image-rendering:auto; user-select:none; -webkit-user-drag:none; }
  #hint { position:absolute; left:12px; bottom:10px; font-size:12px;
          color:var(--dim); background:rgba(12,14,18,.72); padding:3px 8px;
          border-radius:4px; pointer-events:none; }
  #flash { position:absolute; inset:0; pointer-events:none; opacity:0; }
  #flash.on { opacity:.22; transition:none; }
  aside { width:300px; flex:0 0 auto; background:var(--panel);
          border-left:1px solid var(--line); padding:12px 14px;
          overflow-y:auto; }
  aside h3 { margin:0 0 8px; font-size:11px; letter-spacing:.09em;
             text-transform:uppercase; color:var(--dim); font-weight:600; }
  #ctxwrap { position:relative; width:100%; border:1px solid var(--line);
             border-radius:4px; overflow:hidden; background:#0c0e12; }
  #ctx { width:100%; display:block; }
  #ctxbox { position:absolute; border:2px solid var(--accent);
            box-shadow:0 0 0 9999px rgba(10,12,16,.45); pointer-events:none; }
  table { width:100%; border-collapse:collapse; font-size:12px; }
  td { padding:2px 0; }
  td:first-child { color:var(--dim); }
  td:last-child { text-align:right; font-variant-numeric:tabular-nums; }
  footer { flex:0 0 auto; padding:8px 14px; background:var(--panel);
           border-top:1px solid var(--line); display:flex; gap:10px;
           align-items:center; flex-wrap:wrap; }
  kbd { background:#2a3140; border:1px solid var(--line); border-bottom-width:2px;
        border-radius:4px; padding:1px 6px; font:12px ui-monospace,monospace; }
  button { background:#2a3140; color:var(--fg); border:1px solid var(--line);
           border-radius:5px; padding:5px 11px; cursor:pointer; font-size:13px; }
  button:hover { background:#333c4d; }
  #bar { height:3px; background:var(--line); flex:0 0 auto; }
  #bar div { height:100%; background:var(--accent); width:0; transition:width .15s; }
  input[type=text], select { background:#2a3140; color:var(--fg);
           border:1px solid var(--line); border-radius:5px; padding:4px 8px;
           font-size:13px; width:100%; }
  #overlay { position:fixed; inset:0; background:rgba(10,12,16,.94); display:flex;
             align-items:center; justify-content:center; z-index:10; }
  #overlay .box { background:var(--panel); border:1px solid var(--line);
                  border-radius:10px; padding:24px 28px; width:460px; }
  #overlay h2 { margin:0 0 4px; font-size:17px; }
  #overlay p { color:var(--dim); margin:0 0 10px; font-size:13px; }
  #overlay label { display:block; margin:12px 0 4px; font-size:12px;
                   color:var(--dim); }
  #start { width:100%; margin-top:18px; background:var(--accent);
           border-color:var(--accent); padding:8px; font-weight:600; }
  pre { white-space:pre-wrap; font:12px ui-monospace,monospace; color:var(--dim); }
</style>

<div id="overlay"><div class="box">
  <h2>Hepatocyte ballooning — tile labelling</h2>
  <p>For each field: <b>does it contain at least one ballooned hepatocyte?</b>
     Your initials are recorded with every answer so repeats can be compared.
     Use the same initials every session — that is what lets you resume.</p>
  <label>Your initials</label>
  <input id="who" type="text" autofocus placeholder="e.g. EM">
  <label>Set</label>
  <select id="pick"></select>
  <button id="start">Start</button>
</div></div>

<header>
  <b>Does this field contain a ballooned hepatocyte?</b>
  <span class="grow"></span>
  <span class="dim" id="pos"></span>
  <span class="dim" id="pace"></span>
  <button id="statsBtn">Progress</button>
</header>
<div id="bar"><div></div></div>

<main>
  <div id="stage">
    <img id="img" alt="" draggable="false">
    <div id="marks"></div>
    <div id="flash"></div>
    <div id="hint">click a ballooned cell to mark it · <b>backspace</b> undo ·
      scroll zoom · drag pan · <b>0</b> reset · <b>C</b> context</div>
  </div>
  <aside>
    <h3>Surrounding tissue</h3>
    <div id="ctxwrap"><img id="ctx" alt=""><div id="ctxbox"></div></div>
    <p class="dim" style="font-size:12px;margin:6px 0 0">
      The blue box is the field you are judging. Press <kbd>C</kbd> to open
      this view full size.</p>
    <h3 style="margin-top:18px">Note (optional)</h3>
    <input id="notes" type="text" placeholder="e.g. out of focus, folded">
    <h3 style="margin-top:18px">This session</h3>
    <table id="tally"></table>
    <pre id="stats"></pre>
  </aside>
</main>

<footer>
  <kbd>Y</kbd> <span class="dim">yes</span>
  <kbd>N</kbd> <span class="dim">no</span>
  <kbd>U</kbd> <span class="dim">unsure</span>
  <kbd>&larr;</kbd> <span class="dim">back</span>
  <kbd>C</kbd> <span class="dim">context</span>
  <kbd>0</kbd> <span class="dim">reset zoom</span>
  <span class="grow"></span>
  <span class="dim" id="scaleinfo"></span>
</footer>

<script>
const $ = s => document.querySelector(s);
// BASE = 2: a 512 px tile is drawn at 1024 CSS px before any zoom. A tile
// shown at 1:1 is about the size of a postage stamp on a laptop, and the
// texture call this task turns on is not legible at that size.
const BASE = 2, ZMIN = 0.5, ZMAX = 6;
let S = { set:0, reviewer:'', tiles:[], i:0, shown:0, ctxBig:false,
          z:1, ox:0, oy:0, drag:null, tally:{y:0,n:0,u:0}, times:[] };

async function boot() {
  const sets = await (await fetch('/api/sets')).json();
  $('#pick').innerHTML = sets.map(s =>
    `<option value="${s.index}">${s.name} (${s.n})</option>`).join('')
    || '<option>no sets found</option>';
}
boot();

$('#start').onclick = async () => {
  S.reviewer = $('#who').value.trim() || 'anonymous';
  S.set = +$('#pick').value || 0;
  const d = await (await fetch(
    `/api/tiles?set=${S.set}&reviewer=${encodeURIComponent(S.reviewer)}`)).json();
  S.tiles = d.tiles;
  $('#overlay').style.display = 'none';
  // Resume at the first tile this reviewer has not answered, rather than
  // making her skip forward through work she has already done.
  const next = S.tiles.findIndex(t => !t.judged);
  S.i = next < 0 ? S.tiles.length : next;
  show();
};
$('#who').addEventListener('keydown', e => {
  if (e.key === 'Enter') $('#start').click();
});

function show() {
  if (S.i >= S.tiles.length) return done();
  const t = S.tiles[S.i];
  marks = [];
  const kind = S.ctxBig ? 'context' : 'image';
  $('#img').src = `/api/image/${S.set}/${t.index}/${kind}`;
  $('#ctx').src = `/api/image/${S.set}/${t.index}/context`;
  $('#ctx').onload = drawBox;
  $('#pos').textContent = `${S.i + 1} / ${S.tiles.length}`;
  $('#bar div').style.width = (100 * S.i / S.tiles.length) + '%';
  $('#scaleinfo').textContent = S.ctxBig
    ? `context ${t.context_um} µm across`
    : `field ${t.tile_um} µm across`;
  $('#notes').value = '';
  resetView();
  S.shown = performance.now();
}

// The context thumbnail is served at its stored pixel size and displayed at
// whatever width the sidebar happens to be, so the core-tile box has to be
// scaled to the rendered size rather than drawn at stored coordinates.
function drawBox() {
  const t = S.tiles[S.i], img = $('#ctx');
  if (!t || !t.box || !img.naturalWidth) return;
  const [bx, by, bw, bh] = t.box.split(' ').map(Number);
  const k = img.clientWidth / img.naturalWidth;
  const b = $('#ctxbox');
  b.style.left = (bx * k) + 'px';  b.style.top = (by * k) + 'px';
  b.style.width = (bw * k) + 'px'; b.style.height = (bh * k) + 'px';
}
addEventListener('resize', drawBox);

function resetView() { S.z = 1; S.ox = 0; S.oy = 0; applyView(); }

function applyView() {
  const img = $('#img'), k = BASE * S.z;
  img.style.transform =
    `translate(${-img.naturalWidth * k / 2 + S.ox}px,` +
    `${-img.naturalHeight * k / 2 + S.oy}px) scale(${k})`;
  drawMarks();
}
$('#img').onload = applyView;

// ---- point marking -------------------------------------------------------
// Marks are kept as fractions of the TILE, never screen pixels: she may mark at
// 5x zoom, and what has to survive is a location on the slide, not a location
// in a viewport that stops existing when she presses a key.
//
// The stage centres the image (top/left 50% plus a half-size translate), so the
// inverse of that transform is what turns a click back into an image pixel.
// Marking is refused while the context view is up, because #img is then showing
// the 3x3 context and a fraction of it is not a fraction of the tile.
let marks = [];
function stageToTile(cx, cy) {
  const img = $('#img'), st = $('#stage');
  const k = BASE * S.z;
  const ix = (cx - st.clientWidth / 2 + img.naturalWidth * k / 2 - S.ox) / k;
  const iy = (cy - st.clientHeight / 2 + img.naturalHeight * k / 2 - S.oy) / k;
  return { nx: ix / img.naturalWidth, ny: iy / img.naturalHeight };
}
function tileToStage(m) {
  const img = $('#img'), st = $('#stage');
  const k = BASE * S.z;
  return {
    x: m.nx * img.naturalWidth * k + st.clientWidth / 2
       - img.naturalWidth * k / 2 + S.ox,
    y: m.ny * img.naturalHeight * k + st.clientHeight / 2
       - img.naturalHeight * k / 2 + S.oy,
  };
}
function drawMarks() {
  if (S.ctxBig) { $('#marks').innerHTML = ''; return; }
  $('#marks').innerHTML = marks.map((m, i) => {
    const p = tileToStage(m);
    return `<div class="mk" style="left:${p.x}px;top:${p.y}px"><b>${i + 1}</b></div>`;
  }).join('');
}
$('#stage').addEventListener('click', e => {
  if (S.ctxBig || S.moved > 6) return;
  const r = $('#stage').getBoundingClientRect();
  const m = stageToTile(e.clientX - r.left, e.clientY - r.top);
  if (m.nx < 0 || m.nx > 1 || m.ny < 0 || m.ny > 1) return;
  marks.push(m); drawMarks();
});

$('#stage').addEventListener('wheel', e => {
  e.preventDefault();
  const before = S.z;
  S.z = Math.min(ZMAX, Math.max(ZMIN, S.z * Math.exp(-e.deltaY / 400)));
  // Keep the point under the cursor fixed, which is what makes wheel-zoom
  // usable for hunting one cell rather than re-finding it after every notch.
  const r = $('#stage').getBoundingClientRect();
  const cx = e.clientX - r.left - r.width / 2, cy = e.clientY - r.top - r.height / 2;
  S.ox = cx - (cx - S.ox) * (S.z / before);
  S.oy = cy - (cy - S.oy) * (S.z / before);
  applyView();
}, {passive:false});

$('#stage').addEventListener('pointerdown', e => {
  S.drag = {x:e.clientX - S.ox, y:e.clientY - S.oy};
  // Pointer travel since the press. A pan that ends where it began is a click,
  // and without this threshold every release would drop a marker.
  S.moved = 0; S.lastx = e.clientX; S.lasty = e.clientY;
  $('#stage').classList.add('drag');
  $('#stage').setPointerCapture(e.pointerId);
});
$('#stage').addEventListener('pointermove', e => {
  if (!S.drag) return;
  S.moved += Math.abs(e.clientX - S.lastx) + Math.abs(e.clientY - S.lasty);
  S.lastx = e.clientX; S.lasty = e.clientY;
  S.ox = e.clientX - S.drag.x; S.oy = e.clientY - S.drag.y; applyView();
});
const endDrag = () => { S.drag = null; $('#stage').classList.remove('drag'); };
$('#stage').addEventListener('pointerup', endDrag);
$('#stage').addEventListener('pointercancel', endDrag);

function tally() {
  const n = S.times.length;
  const med = n ? [...S.times].sort((a,b)=>a-b)[n>>1].toFixed(1) : '-';
  $('#tally').innerHTML =
    `<tr><td>yes</td><td>${S.tally.y}</td></tr>` +
    `<tr><td>no</td><td>${S.tally.n}</td></tr>` +
    `<tr><td>unsure</td><td>${S.tally.u}</td></tr>` +
    `<tr><td>median</td><td>${med}s</td></tr>`;
  $('#pace').textContent = n ? `${med}s / field` : '';
}

async function judge(v) {
  if (S.i >= S.tiles.length) return;
  const secs = (performance.now() - S.shown) / 1000;
  S.times.push(secs); S.tally[v]++;
  flash({y:'var(--yes)', n:'var(--no)', u:'var(--maybe)'}[v]);
  // Marks belong to a yes; a no or unsure with stray marks would be a
  // contradiction in the data, so they are dropped rather than recorded.
  const body = { set:S.set, index:S.tiles[S.i].index, reviewer:S.reviewer,
                 verdict:v === 'u' ? '?' : v,
                 notes:$('#notes').value, seconds:secs,
                 marks: v === 'y'
                   ? marks.map(m => `${m.nx.toFixed(4)} ${m.ny.toFixed(4)}`).join('; ')
                   : '' };
  S.tiles[S.i].judged = true;
  S.i++; show(); tally();
  // Fire-and-forget: she must never wait on disk. Every row is fsynced
  // server-side, so an interruption loses at most the one in flight.
  fetch('/api/verdict', {method:'POST',
    headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
}

function flash(color) {
  const f = $('#flash');
  f.style.background = color; f.classList.add('on');
  setTimeout(() => f.classList.remove('on'), 110);
}

function done() {
  $('#stage').innerHTML =
    `<div style="position:absolute;inset:0;display:flex;align-items:center;
      justify-content:center;text-align:center"><div>
     <h2>Set complete</h2>
     <p class="dim">${S.tally.y} yes, ${S.tally.n} no, ${S.tally.u} unsure</p>
     <p class="dim">Everything is saved. You can close this window.</p></div></div>`;
  $('#bar div').style.width = '100%';
  $('#pos').textContent = `${S.tiles.length} / ${S.tiles.length}`;
}

document.addEventListener('keydown', e => {
  if ($('#overlay').style.display !== 'none') return;
  if (document.activeElement === $('#notes') && e.key !== 'Enter') return;
  const k = e.key.toLowerCase();
  if (k === 'y') judge('y');
  else if (k === 'n') judge('n');
  else if (k === 'u' || k === '?' || k === '/') judge('u');
  else if (k === 'c') { S.ctxBig = !S.ctxBig; show(); }
  else if (k === '0' || k === 'r') resetView();
  else if (k === '=' || k === '+') { S.z = Math.min(ZMAX, S.z * 1.25); applyView(); }
  else if (k === '-') { S.z = Math.max(ZMIN, S.z / 1.25); applyView(); }
  else if (e.key === 'Backspace') { e.preventDefault(); marks.pop(); drawMarks(); }
  else if (e.key === 'ArrowLeft') { if (S.i > 0) { S.i--; show(); } }
  else if (e.key === 'ArrowRight') { if (S.i < S.tiles.length) { S.i++; show(); } }
});

$('#statsBtn').onclick = async () => {
  $('#stats').textContent = await (await fetch(`/api/stats?set=${S.set}`)).text();
};
</script>
"""


if __name__ == "__main__":
    raise SystemExit(_main())
