"""The viewer's server. Stdlib `http.server`, localhost only.

Same two constraints as `review/app.py` and `review/tiles_app.py`, for the same
reasons. **Stdlib only**: this runs on a compute node behind an SSH tunnel, and
a demo that fails because a dependency did not resolve on Fir is a demo that
did not happen. **Localhost only**: the tiles are patient-adjacent material and
this server has no authentication by design, so binding it to anything routable
would be a mistake that looks like a convenience. On a cluster the reachable
surface is the SSH tunnel and nothing else.

It serves what `precompute.py` already decided. The slide-level percentage is
read from `meta.json`, not recomputed per request -- one census, one number,
and the figure on screen is the figure in the artifact. `/api/infer` is the
one exception: it re-runs the model live on a named region so the demo can
show the thing actually running, and it is labelled as a spot check rather
than as the measurement.
"""

from __future__ import annotations

import io
import json
import threading
import time
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import numpy as np

from ..config import Config
from ..core.slide import Slide
from ..core.tissue import detect_tissue
from .dzi import MaskTileSource, SlideTileSource
from .precompute import (AMBIGUOUS_LO, AMBIGUOUS_HI, TRAINING_MPP,
                         pad_for_unet)

STATIC = Path(__file__).parent / "static"


class Library:
    """The precomputed artifacts, and the slides they point back at.

    Everything is opened lazily and cached: a slide handle and a ds4 mask are
    tens of megabytes, and a demo machine should not pay for three slides when
    the PI is looking at one.
    """

    def __init__(self, root: str | Path, checkpoint: str | Path | None = None,
                 slide_dir: str | Path | None = None):
        self.root = Path(root)
        self.checkpoint = Path(checkpoint) if checkpoint else None
        # Where to look for the slide itself, overriding the path recorded in
        # meta.json. On a compute node the slides are staged to $SLURM_TMPDIR,
        # which is a different directory every session -- so the override is a
        # RUNTIME flag rather than an edit to the artifact. Rewriting
        # meta.json would bake one node's scratch path into a file that
        # outlives the job, and the next session would fail to open its own
        # slides for reasons nothing on screen would explain.
        self.slide_dir = Path(slide_dir) if slide_dir else None
        self._slides: dict[str, Slide] = {}
        self._masks: dict[str, MaskTileSource] = {}
        self._sources: dict[str, SlideTileSource] = {}
        self._tissue: dict = {}
        self._model = None
        # Reentrant, and not optional: `source()` takes the lock and then calls
        # `slide()`, which takes it again. With a plain Lock that is a deadlock
        # on the first tile request -- the .dzi still serves, so the viewer
        # loads and then hangs with a blank stage, which reads like a tiling
        # bug rather than a locking one.
        self._lock = threading.RLock()
        self.cfg = Config()

    def names(self) -> list[str]:
        return sorted(p.parent.name for p in self.root.glob("*/meta.json"))

    @lru_cache(maxsize=32)
    def meta(self, name: str) -> dict:
        p = self.root / name / "meta.json"
        if not p.exists():
            raise KeyError(name)
        return json.loads(p.read_text())

    def slide_path(self, name: str) -> Path:
        recorded = Path(self.meta(name)["slide_path"])
        if self.slide_dir:
            staged = self.slide_dir / recorded.name
            if staged.exists():
                return staged
        return recorded

    def slide(self, name: str) -> Slide:
        with self._lock:
            if name not in self._slides:
                self._slides[name] = Slide(str(self.slide_path(name)), self.cfg.slide)
            return self._slides[name]

    def source(self, name: str) -> SlideTileSource:
        with self._lock:
            if name not in self._sources:
                self._sources[name] = SlideTileSource(self.slide(name))
            return self._sources[name]

    def mask(self, name: str) -> MaskTileSource:
        with self._lock:
            if name not in self._masks:
                m = self.meta(name)
                self._masks[name] = MaskTileSource(
                    self.root / name, m["width"], m["height"],
                    mask_tile=int(m.get("mask_tile", 512)))
            return self._masks[name]

    def tissue(self, name: str):
        with self._lock:
            if name not in self._tissue:
                self._tissue[name] = detect_tissue(self.slide(name), self.cfg.tissue)
            return self._tissue[name]

    def model(self):
        """Loaded once, on first live inference, never at startup.

        Importing TensorFlow costs several seconds and a lot of memory, and a
        session that only pans around precomputed overlays never needs it.
        """
        with self._lock:
            if self._model is None:
                if not self.checkpoint:
                    raise RuntimeError("server started without --checkpoint; "
                                       "live inference is unavailable")
                from .precompute import _load_model
                self._model = _load_model(self.checkpoint)
            return self._model


def _png(arr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return buf.tobytes()


def _jpeg(rgb: np.ndarray, quality: int = 82) -> bytes:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                           [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


class Handler(BaseHTTPRequestHandler):
    library: Library = None  # type: ignore[assignment]
    server_version = "mashpath-viewer/1.0"

    # -- plumbing --

    def log_message(self, fmt, *args):  # quieter than the default
        pass

    def _send(self, body: bytes, ctype: str, cache: bool = False) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if cache:
            self.send_header("Cache-Control", "public, max-age=3600")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj) -> None:
        self._send(json.dumps(obj, default=float).encode(), "application/json")

    def _fail(self, code: int, msg: str) -> None:
        body = json.dumps({"error": msg}).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # -- routes --

    def do_GET(self) -> None:  # noqa: N802
        u = urlparse(self.path)
        path = unquote(u.path)
        q = parse_qs(u.query)
        lib = self.library
        try:
            if path == "/" or path.startswith("/v/"):
                self._send((STATIC / "viewer.html").read_bytes(), "text/html")
            elif path.startswith("/static/"):
                self._static(path[len("/static/"):])
            elif path == "/api/slides":
                self._json([{"name": n, **_card(lib.meta(n))} for n in lib.names()])
            elif path.startswith("/api/meta/"):
                self._json(lib.meta(path[len("/api/meta/"):]))
            elif path.startswith("/api/infer/"):
                self._infer(path[len("/api/infer/"):], q)
            elif path.startswith("/slide/"):
                self._pyramid(path[len("/slide/"):], overlay=False)
            elif path.startswith("/overlay/"):
                self._pyramid(path[len("/overlay/"):], overlay=True)
            else:
                self._fail(404, f"no route for {path}")
        except KeyError as e:
            self._fail(404, f"unknown slide {e}")
        except IndexError:
            # OpenSeadragon speculatively requests tiles past the edge of a
            # level; 404 is the correct answer and it handles it silently.
            self._fail(404, "tile out of range")
        except BrokenPipeError:
            pass
        except Exception as e:  # surface it rather than hanging the viewer
            self._fail(500, f"{type(e).__name__}: {e}")

    def _static(self, rel: str) -> None:
        p = (STATIC / rel).resolve()
        if not str(p).startswith(str(STATIC.resolve())) or not p.exists():
            return self._fail(404, rel)
        ctype = {".js": "application/javascript", ".css": "text/css",
                 ".png": "image/png", ".html": "text/html",
                 ".txt": "text/plain"}.get(p.suffix, "application/octet-stream")
        self._send(p.read_bytes(), ctype, cache=True)

    def _pyramid(self, rest: str, overlay: bool) -> None:
        """`<name>.dzi` and `<name>_files/<level>/<col>_<row>.<ext>`."""
        lib = self.library
        if rest.endswith(".dzi"):
            name = rest[:-4]
            m = lib.meta(name)
            from .dzi import DeepZoom
            dz = DeepZoom(m["width"], m["height"])
            fmt = "png" if overlay else "jpeg"
            return self._send(dz.dzi_xml(fmt).encode(), "application/xml", cache=True)

        if "_files/" not in rest:
            return self._fail(404, rest)
        name, tail = rest.split("_files/", 1)
        parts = tail.split("/")
        if len(parts) != 2:
            return self._fail(404, rest)
        level = int(parts[0])
        stem = parts[1].rsplit(".", 1)[0]
        col, row = (int(v) for v in stem.split("_"))

        if overlay:
            body = _png(lib.mask(name).tile_rgba(level, col, row))
            return self._send(body, "image/png", cache=True)
        body = _jpeg(lib.source(name).tile(level, col, row))
        self._send(body, "image/jpeg", cache=True)

    def _infer(self, name: str, q: dict) -> None:
        """Run the model, now, on one region. The "it is really running" button.

        Deliberately NOT how the slide percentage is produced. This measures
        the region the user is looking at; the headline figure stays the
        census in meta.json. Mixing the two would let a percentage on screen
        depend on where the viewport happened to be.
        """
        lib = self.library
        m = lib.meta(name)
        try:
            x = int(float(q["x"][0])); y = int(float(q["y"][0]))
            w = int(float(q.get("w", ["2048"])[0]))
            h = int(float(q.get("h", [str(w)])[0]))
        except (KeyError, ValueError):
            return self._fail(400, "need x, y and optionally w, h (level-0 pixels)")

        W, H = m["width"], m["height"]
        w = max(256, min(w, 4096)); h = max(256, min(h, 4096))
        x = max(0, min(x, max(0, W - w))); y = max(0, min(y, max(0, H - h)))

        sl = lib.slide(name)
        tis = lib.tissue(name)
        t0 = time.time()
        rgb = sl.read_region((x, y), 0, (w, h))
        t_read = time.time() - t0

        # Resample to the training scale if this slide is not at it. Same rule
        # as the census: the model is fed 0.4953 um/px or it is fed nonsense.
        ratio = float(m["mpp"]) / TRAINING_MPP
        inp = rgb
        if abs(ratio - 1.0) > 0.02:
            inp = cv2.resize(rgb, (int(round(w / ratio)), int(round(h / ratio))),
                             interpolation=cv2.INTER_AREA)

        # Pad to a multiple of 16 before the model sees it. A viewport is
        # whatever size the browser window happens to be, and the U-Net's skip
        # connections only line up on multiples of 2^depth -- without this the
        # spot check crashes on most viewports and works on a few.
        padded, ih, iw = pad_for_unet(inp)

        model = lib.model()
        t0 = time.time()
        with lib._lock:
            p = np.asarray(model.predict_on_batch(
                padded.astype(np.float32)[None] / 255.0)[0, ..., 0])
        t_infer = time.time() - t0
        p = p[:ih, :iw]                      # drop the padding again
        if p.shape != (h, w):
            p = cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR)

        tmask = tis.tile_mask(x, y, w, h)
        fat = (p >= m.get("threshold", 0.5)) & tmask
        fat_px, tis_px = int(fat.sum()), int(tmask.sum())
        amb = int(((p >= AMBIGUOUS_LO) & (p <= AMBIGUOUS_HI) & tmask).sum())

        import base64
        rgba = np.zeros((h, w, 4), np.uint8)
        rgba[..., 2] = 255; rgba[..., 3] = fat.astype(np.uint8) * 255
        self._json({
            "x": x, "y": y, "w": w, "h": h,
            "fat_percent": 100.0 * fat_px / tis_px if tis_px else 0.0,
            "fat_px": fat_px, "tissue_px": tis_px,
            "ambiguous_fraction_of_tissue": amb / tis_px if tis_px else 0.0,
            "read_ms": 1000 * t_read, "infer_ms": 1000 * t_infer,
            "mask_png_b64": base64.b64encode(_png(rgba)).decode(),
            "note": "live spot check of this region; the slide figure is the census",
        })


def _card(m: dict) -> dict:
    """The few fields the slide picker needs, without shipping the whole meta."""
    return {
        "fat_percent": m.get("fat_percent"),
        "tiles": m.get("tiles"),
        "coverage": m.get("coverage"),
        "tissue_mm2": m.get("tissue_mm2"),
        "fold": m.get("model", {}).get("fold"),
        "held_out_batch": m.get("model", {}).get("held_out_batch"),
    }


def serve(root: str | Path, checkpoint: str | Path | None = None,
          port: int = 8000, host: str = "127.0.0.1",
          slide_dir: str | Path | None = None) -> None:
    lib = Library(root, checkpoint, slide_dir)
    names = lib.names()
    if not names:
        raise SystemExit(f"no precomputed slides under {root} "
                         f"(expected <slide>/meta.json)")
    Handler.library = lib
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"mashpath viewer on http://{host}:{port}")
    print(f"  artifacts: {Path(root).resolve()}")
    print(f"  slides:    {', '.join(names)}")
    for n in names:
        m = lib.meta(n)
        print(f"     {n:<32} {m['fat_percent']:6.2f}% fat of tissue  "
              f"({m['coverage']}, fold {m.get('model', {}).get('fold')})")
    print(f"  live inference: {'on' if checkpoint else 'OFF (no --checkpoint)'}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


def main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", required=True, help="directory of precomputed slides")
    p.add_argument("--checkpoint", default=None,
                   help="enables the live-inference spot check")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="127.0.0.1",
                   help="leave as localhost; reach it over an SSH tunnel")
    p.add_argument("--slide-dir", default=None,
                   help="look for slides here by filename, overriding the path "
                        "in meta.json -- point it at $SLURM_TMPDIR when the "
                        "slides have been staged to node-local disk")
    a = p.parse_args(argv)
    serve(a.root, a.checkpoint, a.port, a.host, a.slide_dir)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
