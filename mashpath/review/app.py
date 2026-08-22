"""The review app: a local web page a pathologist works through by keyboard.

Runs on `http.server` from the standard library, deliberately. A review session
happens on a laptop next to a microscope, or over an SSH tunnel from a login
node, and adding Flask would mean a pathologist's session can fail because of a
dependency resolution. Nothing here imports anything that is not either stdlib
or already required for the pipeline.

WHY THIS EXISTS AT ALL, rather than the CSV-in-Excel route it replaces: the
scarce resource in this project is pathologist attention, and Excel spends it
badly. A reviewer scrolling a spreadsheet has to find the row, read the id,
locate the PNG, open it, judge it, click back, and type. Measured against a
keyboard-driven crop viewer, that is roughly 20 s versus 5 s per candidate --
so the same hour buys ~700 judgments instead of ~180. It also loses the two
measurements the session is FOR: Excel has one verdict column, so it cannot
record two reviewers on one candidate, and it cannot record one reviewer
twice. Those are inter- and intra-rater agreement, and without them a
confirmed set has no error bar.

Serve only to localhost. The crops are patient-adjacent material and this
server has no authentication by design -- binding it to anything else would be
a mistake that looks like a convenience.
"""

from __future__ import annotations

import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from . import manifest as manifest_mod
from . import verdicts as verdicts_mod

VERDICTS_FILENAME = "verdicts.csv"


# ---- what the reviewer is working through ---------------------------------


class ReviewSet:
    """One manifest, its crops, and the verdict store beside it."""

    def __init__(self, manifest_path: Path):
        self.manifest_path = Path(manifest_path)
        self.dir = self.manifest_path.parent
        self.rows = manifest_mod.read(self.manifest_path)
        self.verdicts_path = self.dir / VERDICTS_FILENAME
        self.slide = self.rows[0]["slide"] if self.rows else self.dir.parent.parent.name
        self.feature = self.rows[0]["feature"] if self.rows else self.dir.parent.name

    @property
    def key(self) -> str:
        return f"{self.slide}/{self.feature}"

    def image_path(self, index: int, kind: str) -> Path | None:
        row = self.rows[index]
        rel = row.get("overlay" if kind == "overlay" else "image") or ""
        if not rel:
            # `save_unmarked: false` leaves no plain crop. Fall back to the
            # overlay rather than showing the reviewer a broken image.
            rel = row.get("overlay") or ""
        if not rel:
            return None
        p = (self.dir / rel).resolve()
        # Path traversal guard: `rel` comes out of a CSV, which is a file a
        # human may have edited.
        if not str(p).startswith(str(self.dir.resolve())):
            return None
        return p if p.exists() else None


def discover(roots: list[str | Path]) -> list[ReviewSet]:
    """Find every review package under the given output directories.

    Sorted so the order a reviewer meets slides in is reproducible between
    runs -- if two reviewers are to be compared, they must see the same set,
    and "whatever the filesystem returned" is not a set.
    """
    found: list[ReviewSet] = []
    seen: set[Path] = set()
    for root in roots:
        root = Path(root)
        if root.is_file() and root.name.endswith(".csv"):
            paths = [root]
        else:
            paths = sorted(root.glob("*/*/review/manifest.csv"))
            paths += sorted(root.glob("*/review/manifest.csv"))
        for p in paths:
            p = p.resolve()
            if p in seen:
                continue
            seen.add(p)
            try:
                rs = ReviewSet(p)
            except manifest_mod.ManifestError as exc:
                print(f"  skipping {p}: {exc}")
                continue
            if rs.rows:
                found.append(rs)
    return sorted(found, key=lambda r: (r.feature, r.slide))


# ---- server ---------------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    sets: list[ReviewSet] = []

    def log_message(self, fmt, *args):  # noqa: A003 - quiet the per-request log
        pass

    # -- helpers --
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj: Any, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    # -- routes --
    def do_GET(self) -> None:  # noqa: N802
        u = urlparse(self.path)
        q = parse_qs(u.query)

        if u.path == "/":
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")

        elif u.path == "/api/sets":
            self._json([
                {
                    "index": i, "slide": s.slide, "feature": s.feature,
                    "key": s.key, "n": len(s.rows),
                    "done": len({r["candidate_id"]
                                 for r in verdicts_mod.load(s.verdicts_path)
                                 if r.get("reviewer") == (q.get("reviewer", [""])[0])}),
                }
                for i, s in enumerate(self.sets)
            ])

        elif u.path == "/api/candidates":
            si = int(q.get("set", ["0"])[0])
            reviewer = q.get("reviewer", [""])[0]
            s = self.sets[si]
            judged = {r["candidate_id"] for r in verdicts_mod.load(s.verdicts_path)
                      if r.get("reviewer") == reviewer}
            self._json({
                "slide": s.slide, "feature": s.feature,
                "measurements": manifest_mod.measurement_columns(s.rows),
                "candidates": [
                    {
                        "index": i,
                        "id": r["candidate_id"],
                        "score": r.get("score", ""),
                        "crop_um": r.get("crop_um", ""),
                        "crop_mode": r.get("crop_mode", ""),
                        "band": r.get("review_band", ""),
                        "judged": r["candidate_id"] in judged,
                        "m": {k[2:]: r[k] for k in r
                              if k.startswith("m_") and r[k] not in (None, "")},
                    }
                    for i, r in enumerate(s.rows)
                ],
            })

        elif u.path.startswith("/api/image/"):
            _, _, _, si, idx, kind = u.path.split("/", 5)
            p = self.sets[int(si)].image_path(int(idx), kind)
            if p is None:
                self._send(404, b"no image", "text/plain")
                return
            self._send(200, p.read_bytes(), "image/png")

        elif u.path == "/api/stats":
            si = int(q.get("set", ["0"])[0])
            s = self.sets[si]
            self._send(
                200,
                verdicts_mod.agreement_report(s.verdicts_path).encode(),
                "text/plain; charset=utf-8",
            )

        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        u = urlparse(self.path)
        if u.path != "/api/verdict":
            self._send(404, b"not found", "text/plain")
            return
        n = int(self.headers.get("Content-Length", 0))
        data = json.loads(self.rfile.read(n) or b"{}")

        s = self.sets[int(data["set"])]
        row = s.rows[int(data["index"])]
        verdicts_mod.append(
            s.verdicts_path,
            reviewer=data.get("reviewer", "").strip() or "anonymous",
            slide=row["slide"], feature=row["feature"],
            candidate_id=row["candidate_id"],
            verdict=data.get("verdict", ""),
            notes=data.get("notes", ""),
            seconds=float(data.get("seconds", 0) or 0),
        )
        self._json({"ok": True})


def serve(roots: list[str | Path], port: int = 8000, host: str = "127.0.0.1") -> None:
    """Start the review server. Blocks until interrupted."""
    sets = discover(roots)
    if not sets:
        raise SystemExit(
            f"no review packages found under {[str(r) for r in roots]}.\n"
            "Expected <output-dir>/<slide>/<feature>/review/manifest.csv -- "
            "run a candidate pipeline first."
        )
    _Handler.sets = sets

    total = sum(len(s.rows) for s in sets)
    print(f"review server: {len(sets)} set(s), {total} candidates")
    for s in sets:
        print(f"  {s.feature:14s} {s.slide:24s} {len(s.rows):5d} candidates")
    print(f"\n  open  http://{host}:{port}/\n")
    print("  verdicts append to verdicts.csv beside each manifest")
    print("  Ctrl-C to stop\n")

    srv = ThreadingHTTPServer((host, port), _Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        srv.server_close()
        for s in sets:
            rep = verdicts_mod.agreement_report(s.verdicts_path)
            if "no verdicts" not in rep:
                print(f"\n--- {s.key} ---\n{rep}")


# ---- the page -------------------------------------------------------------
# One file, no external assets: the whole thing has to work with no network,
# because that is the situation on a cluster login node behind an SSH tunnel.

PAGE = r"""<!doctype html>
<meta charset="utf-8">
<title>mashpath review</title>
<style>
  :root {
    --bg:#12141a; --panel:#1b1f27; --line:#2c3340; --fg:#e8ecf2; --dim:#8a94a6;
    --yes:#2ea36b; --no:#c8503f; --maybe:#c99a2e; --accent:#4d8fd6;
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg); height:100vh;
         display:flex; flex-direction:column;
         font:14px/1.45 ui-sans-serif,system-ui,-apple-system,sans-serif; }
  header { display:flex; align-items:center; gap:16px; padding:8px 14px;
           background:var(--panel); border-bottom:1px solid var(--line); flex:0 0 auto; }
  header b { font-weight:600; }
  .dim { color:var(--dim); }
  .grow { flex:1; }
  main { flex:1; display:flex; min-height:0; }
  #stage { flex:1; display:flex; align-items:center; justify-content:center;
           padding:12px; min-width:0; position:relative; }
  #stage img { max-width:100%; max-height:100%; object-fit:contain;
               border-radius:4px; image-rendering:auto; }
  aside { width:280px; flex:0 0 auto; background:var(--panel);
          border-left:1px solid var(--line); padding:12px 14px; overflow-y:auto; }
  aside h3 { margin:0 0 8px; font-size:11px; letter-spacing:.09em;
             text-transform:uppercase; color:var(--dim); font-weight:600; }
  table { width:100%; border-collapse:collapse; font-size:12px; }
  td { padding:2px 0; vertical-align:top; }
  td:first-child { color:var(--dim); padding-right:8px; white-space:nowrap; }
  td:last-child { text-align:right; font-variant-numeric:tabular-nums; }
  footer { flex:0 0 auto; padding:8px 14px; background:var(--panel);
           border-top:1px solid var(--line); display:flex; gap:10px;
           align-items:center; }
  kbd { background:#2a3140; border:1px solid var(--line); border-bottom-width:2px;
        border-radius:4px; padding:1px 6px; font:12px ui-monospace,monospace; }
  button { background:#2a3140; color:var(--fg); border:1px solid var(--line);
           border-radius:5px; padding:5px 11px; cursor:pointer; font-size:13px; }
  button:hover { background:#333c4d; }
  #bar { height:3px; background:var(--line); flex:0 0 auto; }
  #bar div { height:100%; background:var(--accent); width:0; transition:width .15s; }
  #flash { position:absolute; inset:0; pointer-events:none; opacity:0;
           border-radius:4px; }
  #flash.on { opacity:.20; transition:none; }
  select, input[type=text] { background:#2a3140; color:var(--fg);
           border:1px solid var(--line); border-radius:5px; padding:4px 8px;
           font-size:13px; }
  #notes { width:100%; margin-top:6px; }
  #overlay { position:fixed; inset:0; background:rgba(10,12,16,.94); display:flex;
             align-items:center; justify-content:center; z-index:10; }
  #overlay .box { background:var(--panel); border:1px solid var(--line);
                  border-radius:10px; padding:24px 28px; width:440px; }
  #overlay h2 { margin:0 0 4px; font-size:17px; }
  #overlay p { color:var(--dim); margin:0 0 16px; font-size:13px; }
  #overlay label { display:block; margin:12px 0 4px; font-size:12px;
                   color:var(--dim); }
  #overlay input, #overlay select { width:100%; }
  #start { width:100%; margin-top:18px; background:var(--accent);
           border-color:var(--accent); padding:8px; font-weight:600; }
  pre { white-space:pre-wrap; font:12px ui-monospace,monospace; color:var(--dim); }
</style>

<div id="overlay"><div class="box">
  <h2>mashpath review</h2>
  <p>Your initials are recorded with every judgment, so two reviewers can be
     compared. Use the same initials each session.</p>
  <label>Your initials</label>
  <input id="who" type="text" autofocus placeholder="e.g. JS">
  <label>Review set</label>
  <select id="pick"></select>
  <button id="start">Start</button>
</div></div>

<header>
  <b id="feat">-</b>
  <span class="dim" id="slide"></span>
  <span class="grow"></span>
  <span class="dim" id="pos"></span>
  <span class="dim" id="pace"></span>
  <button id="statsBtn">Agreement</button>
</header>
<div id="bar"><div></div></div>

<main>
  <div id="stage"><img id="img" alt=""><div id="flash"></div></div>
  <aside>
    <h3>Why proposed</h3>
    <table id="meas"></table>
    <h3 style="margin-top:18px">Notes</h3>
    <input id="notes" type="text" placeholder="optional - why rejected?">
    <h3 style="margin-top:18px">Session</h3>
    <table id="tally"></table>
    <pre id="stats"></pre>
  </aside>
</main>

<footer>
  <kbd>Y</kbd> <span class="dim">yes</span>
  <kbd>N</kbd> <span class="dim">no</span>
  <kbd>?</kbd> <span class="dim">can't tell</span>
  <kbd>O</kbd> <span class="dim">toggle outline</span>
  <kbd>&larr;</kbd> <span class="dim">back</span>
  <span class="grow"></span>
  <span class="dim" id="cropinfo"></span>
</footer>

<script>
const $ = s => document.querySelector(s);
let S = { set:0, reviewer:'', cands:[], i:0, shown:0, outline:true,
          tally:{y:0,n:0,'?':0}, times:[] };

async function boot() {
  const sets = await (await fetch('/api/sets')).json();
  $('#pick').innerHTML = sets.map(s =>
    `<option value="${s.index}">${s.feature} - ${s.slide} (${s.n})</option>`).join('');
  if (!sets.length) $('#pick').innerHTML = '<option>no review sets found</option>';
}
boot();

$('#start').onclick = async () => {
  S.reviewer = $('#who').value.trim() || 'anonymous';
  S.set = +$('#pick').value || 0;
  const d = await (await fetch(`/api/candidates?set=${S.set}&reviewer=${encodeURIComponent(S.reviewer)}`)).json();
  S.cands = d.candidates;
  $('#feat').textContent = d.feature;
  $('#slide').textContent = d.slide;
  $('#overlay').style.display = 'none';
  // Resume where this reviewer left off rather than making them skip forward.
  S.i = Math.max(0, S.cands.findIndex(c => !c.judged));
  if (S.i < 0) S.i = 0;
  show();
};
$('#who').addEventListener('keydown', e => { if (e.key === 'Enter') $('#start').click(); });

function show() {
  if (S.i >= S.cands.length) return done();
  const c = S.cands[S.i];
  $('#img').src = `/api/image/${S.set}/${c.index}/${S.outline ? 'overlay' : 'image'}`;
  $('#pos').textContent = `${S.i + 1} / ${S.cands.length}`;
  $('#bar div').style.width = (100 * S.i / S.cands.length) + '%';
  $('#cropinfo').textContent = c.crop_um ? `crop ${c.crop_um} um (${c.crop_mode})` : '';
  // The score and band are deliberately NOT shown: a reviewer who can see the
  // detector's confidence is no longer an independent judgment of the crop.
  $('#meas').innerHTML = Object.entries(c.m).map(([k, v]) =>
    `<tr><td>${k}</td><td>${(+v).toPrecision(3)}</td></tr>`).join('') ||
    '<tr><td class="dim">none recorded</td></tr>';
  $('#notes').value = '';
  S.shown = performance.now();
}

function tally() {
  const n = S.times.length;
  const med = n ? [...S.times].sort((a,b)=>a-b)[n>>1].toFixed(1) : '-';
  $('#tally').innerHTML =
    `<tr><td>yes</td><td>${S.tally.y}</td></tr>` +
    `<tr><td>no</td><td>${S.tally.n}</td></tr>` +
    `<tr><td>can't tell</td><td>${S.tally['?']}</td></tr>` +
    `<tr><td>median</td><td>${med}s</td></tr>`;
  $('#pace').textContent = n ? `${med}s / candidate` : '';
}

async function judge(v) {
  if (S.i >= S.cands.length) return;
  const secs = (performance.now() - S.shown) / 1000;
  S.times.push(secs); S.tally[v]++;
  flash({y:'var(--yes)', n:'var(--no)', '?':'var(--maybe)'}[v]);
  const body = { set:S.set, index:S.cands[S.i].index, reviewer:S.reviewer,
                 verdict:v, notes:$('#notes').value, seconds:secs };
  S.i++; show(); tally();
  // Fire-and-forget: the reviewer must never wait on disk. Each row is
  // fsynced server-side, so a crash loses at most the one in flight.
  fetch('/api/verdict', {method:'POST', headers:{'Content-Type':'application/json'},
                         body:JSON.stringify(body)});
}

function flash(color) {
  const f = $('#flash');
  f.style.background = color; f.classList.add('on');
  setTimeout(() => f.classList.remove('on'), 110);
}

function done() {
  $('#stage').innerHTML =
    `<div style="text-align:center"><h2>Set complete</h2>
     <p class="dim">${S.tally.y} confirmed, ${S.tally.n} rejected,
     ${S.tally['?']} uncertain</p>
     <p class="dim">Verdicts are saved. Reload to start another set.</p></div>`;
  $('#bar div').style.width = '100%';
  $('#pos').textContent = `${S.cands.length} / ${S.cands.length}`;
}

document.addEventListener('keydown', e => {
  if ($('#overlay').style.display !== 'none') return;
  if (document.activeElement === $('#notes') && e.key !== 'Enter') return;
  const k = e.key.toLowerCase();
  if (k === 'y') judge('y');
  else if (k === 'n') judge('n');
  else if (k === '?' || k === '/' || k === 'u') judge('?');
  else if (k === 'o') { S.outline = !S.outline; show(); }
  else if (e.key === 'ArrowLeft') { if (S.i > 0) { S.i--; show(); } }
  else if (e.key === 'ArrowRight') { if (S.i < S.cands.length) { S.i++; show(); } }
});

$('#statsBtn').onclick = async () => {
  $('#stats').textContent = await (await fetch(`/api/stats?set=${S.set}`)).text();
};
</script>
"""
