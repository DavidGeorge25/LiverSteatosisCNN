const $ = s => document.querySelector(s);
const FIELDS = JSON.parse($('#data').textContent);
const IMGS = JSON.parse($('#imgs').textContent);
const KEY = 'ballooning-review-v1';
const N = FIELDS.length;

// A session id, so two files from the same person (a restart, a second sitting,
// a different machine) are distinguishable instead of silently merging.
const SESSION = (Date.now().toString(36) + Math.random().toString(36).slice(2, 7))
                .toUpperCase();
let S = { who:'', i:0, ans:{}, t0:0, notes:[], ckDone:{} };
let storageOK = true;
try { localStorage.setItem('_t', '1'); localStorage.removeItem('_t'); }
catch(e){ storageOK = false; }

// ---- persistence. Written on EVERY answer, not on a timer: the whole promise
// to her is that closing the tab loses nothing.
function save(){
  try { localStorage.setItem(KEY, JSON.stringify(
          {who:S.who, ans:S.ans, notes:S.notes, ckDone:S.ckDone})); }
  catch(e){ /* private mode: she can still finish and export in one sitting */ }
}
function load(){
  try { return JSON.parse(localStorage.getItem(KEY) || 'null'); } catch(e){ return null; }
}

const saved = load();
if (saved && saved.who) {
  $('#who').value = saved.who;
  const n = Object.keys(saved.ans || {}).length;
  if (n) {
    $('#introMsg').textContent =
      `Welcome back — ${n} of ${N} answered. Press Begin to carry on.`;
    $('#reset').style.display = '';
  }
}

$('#reset').onclick = () => {
  if (!confirm('Delete your answers so far and start from the first field?')) return;
  localStorage.removeItem(KEY); S.ans = {};
  $('#introMsg').textContent = 'Cleared.'; $('#reset').style.display = 'none';
};

$('#go').onclick = () => {
  const who = $('#who').value.trim();
  if (!who) { $('#introMsg').textContent = 'Please enter your initials first.';
              $('#who').focus(); return; }
  S.who = who;
  const sv = load();
  if (sv && sv.who === who) {
    S.ans = sv.ans || {}; S.notes = sv.notes || []; S.ckDone = sv.ckDone || {};
  }
  S.i = FIELDS.findIndex(f => !S.ans[f.id]);
  if (S.i < 0) S.i = N;
  $('#intro').style.display = 'none';
  $('#work').style.display = 'flex';
  $('#tot').textContent = N;
  show();
};

// ---- zoom / pan
let z = 1, ox = 0, oy = 0, dragging = false, px = 0, py = 0;
let moved = 0;          // pointer travel, to tell a click from a pan
let marks = [];         // {nx,ny} in 0..1 of the TILE, independent of zoom/pan

// Marks are kept in tile-normalised coordinates, never screen pixels: she may
// mark at 4x zoom and the merge step has to turn that into a slide coordinate
// long after the viewport is gone.
function toTile(cx, cy){
  const w = fb.clientWidth, h = fb.clientHeight;
  return { nx: (cx - ox) / (z * w), ny: (cy - oy) / (z * h) };
}
// A mark means "this CELL", so the ring is sized in microns, not pixels. At 26
// px it was ~5 um on a 253.6 um field -- a fifth of a normal hepatocyte and a
// tenth of a ballooned one, so it read as a dot floating near something rather
// than as a cell being named. A ballooned hepatocyte runs 2-3x a normal one,
// so 40 um is about the object she is pointing at.
const TILE_UM = 253.6;
const RING_UM = 40;
function ringPx(){ return (RING_UM / TILE_UM) * fb.clientWidth * z; }

function drawMarks(){
  const w = fb.clientWidth, h = fb.clientHeight, d = ringPx();
  $('#marks').innerHTML = marks.map((m, i) => {
    const x = m.nx * w * z + ox, y = m.ny * h * z + oy;
    return `<div class="mk" style="left:${x}px;top:${y}px;width:${d}px;height:${d}px;` +
           `margin:${-d/2}px 0 0 ${-d/2}px"><b>${i + 1}</b></div>`;
  }).join('');
  const n = marks.length;
  $('#markn').style.display = n ? '' : 'none';
  $('#undo').style.display = n ? '' : 'none';
  $('#markn').textContent = n === 1 ? '1 cell marked' : `${n} cells marked`;
  if (n) disarm();
  $('#markhint').innerHTML = n
    ? `<strong>${n} marked.</strong> Click any others, then press <kbd>Y</kbd>. ` +
      'Click a ring again to remove it.'
    : 'Click <strong>every</strong> ballooned cell, then press <kbd>Y</kbd>.';
  // The button carries the count too, so the link between the circles and the
  // answer is visible without reading the hint line.
  $('#vylabel').textContent = n ? (n === 1 ? 'Yes · 1 cell' : `Yes · ${n} cells`) : 'Yes';
}
function disarm(){
  armed = false;
  $('#markhint').classList.remove('ask');
  $('#vy').classList.remove('ask');
}
function applyView(){
  $('#field').style.transform = `translate(${ox}px,${oy}px) scale(${z})`;
  drawMarks();
  $('#fscale').textContent = z > 1.02 ? `253.6 µm across · ${z.toFixed(1)}×`
                                      : '253.6 µm across';
}
function resetView(){ z = 1; ox = 0; oy = 0; applyView(); }
const fb = $('#fieldbox');
fb.addEventListener('wheel', e => {
  e.preventDefault();
  const r = fb.getBoundingClientRect();
  const mx = e.clientX - r.left, my = e.clientY - r.top;
  const nz = Math.min(6, Math.max(1, z * (e.deltaY < 0 ? 1.16 : 1/1.16)));
  // Keep the point under the cursor fixed, so zooming reads as a microscope
  // stage rather than the image jumping to a corner.
  ox = mx - (mx - ox) * (nz / z); oy = my - (my - oy) * (nz / z);
  z = nz; clampPan(); applyView();
}, {passive:false});
fb.addEventListener('pointerdown', e => {
  px = e.clientX; py = e.clientY; moved = 0;
  if (z <= 1) return;
  dragging = true; fb.classList.add('drag'); fb.setPointerCapture(e.pointerId);
});
fb.addEventListener('pointermove', e => {
  if (!dragging) return;
  moved += Math.abs(e.clientX - px) + Math.abs(e.clientY - py);
  ox += e.clientX - px; oy += e.clientY - py; px = e.clientX; py = e.clientY;
  clampPan(); applyView();
});
['pointerup','pointercancel'].forEach(ev =>
  fb.addEventListener(ev, () => { dragging = false; fb.classList.remove('drag'); }));

// A drag that ends where it started is a click. Without the travel threshold,
// panning at zoom would drop a marker every time she let go.
fb.addEventListener('click', e => {
  if (moved > 6) return;
  const r = fb.getBoundingClientRect();
  const m = toTile(e.clientX - r.left, e.clientY - r.top);
  if (m.nx < 0 || m.nx > 1 || m.ny < 0 || m.ny > 1) return;
  // A click inside an existing ring REMOVES it. Without this, a second click on
  // the same cell silently became a second mark -- which is what produced the
  // overlapping 1/2 pairs, and would have counted one cell twice.
  const near = marks.findIndex(k =>
    Math.hypot((k.nx - m.nx) * TILE_UM, (k.ny - m.ny) * TILE_UM) < RING_UM / 2);
  if (near >= 0) marks.splice(near, 1); else marks.push(m);
  drawMarks();
});
$('#undo').onclick = () => { marks.pop(); drawMarks(); };
function clampPan(){
  const w = fb.clientWidth, h = fb.clientHeight;
  ox = Math.min(0, Math.max(w - w*z, ox));
  oy = Math.min(0, Math.max(h - h*z, oy));
}

// ---- the field
function show(){
  if (S.i >= N) return finish();
  const f = FIELDS[S.i], im = IMGS[f.img];
  $('#field').src = 'data:image/jpeg;base64,' + im.t;
  $('#ctx').src   = 'data:image/jpeg;base64,' + im.c;
  // The core tile is the middle ninth of a 3x3 context view.
  Object.assign($('#box').style, {left:'33.333%', top:'33.333%',
                                  width:'33.333%', height:'33.333%'});
  $('#pos').textContent = S.i + 1;
  $('#prog').style.width = (S.i / N * 100) + '%';
  // Going back must show what she marked, not a blank field.
  marks = (S.ans[f.id] && S.ans[f.id].m || []).map(a => ({nx:a[0], ny:a[1]}));
  disarm();
  resetView(); tally();
  S.t0 = performance.now();
}

function tally(){
  const v = Object.values(S.ans);
  const c = k => v.filter(a => a.v === k).length;
  $('#ty').textContent = c('y'); $('#tn').textContent = c('n');
  $('#tu').textContent = c('u');
  const secs = v.map(a => a.s).filter(x => x > 0).sort((a,b) => a-b);
  $('#pace').textContent = secs.length >= 5
    ? `${secs[Math.floor(secs.length/2)].toFixed(0)}s / field` : '';
}

let armed = false;   // a yes with no marks, waiting for a second press

// A checkpoint early in the session. Brunt et al. found expert agreement on
// presence/absence of ballooning is near chance (Fleiss kappa 0.197), so if the
// question or the images are wrong for this material, it will not show up in
// the verdicts -- they will just be noisy. Asking directly, once, at a point
// where stopping still costs almost nothing, is the only cheap way to find out.
const CHECKPOINTS = [15];
function maybeCheckpoint(){
  if (!CHECKPOINTS.includes(S.i) || S.ckDone[S.i]) return false;
  S.ckDone[S.i] = true;
  $('#ckN').textContent = S.i;
  $('#checkpoint').style.display = 'grid';
  $('#ckText').focus();
  return true;
}
$('#ckGo').onclick = () => {
  const t = $('#ckText').value.trim();
  if (t) S.notes.push({ at: S.i, text: t });
  save();
  $('#checkpoint').style.display = 'none';
  show();
};

function judge(v){
  const f = FIELDS[S.i];
  // A yes without a mark is allowed but never silent. The first press asks for
  // the cell; the second accepts the answer as it stands. Without this the
  // marking is a feature she can use, not a step in the task, and an earlier
  // build shipped exactly that -- circles could be placed but pressing Yes
  // neither asked for them nor acknowledged them.
  if (v === 'y' && marks.length === 0 && !armed) {
    armed = true;
    $('#markhint').innerHTML =
      '<strong>Click every ballooned cell</strong>, then press <kbd>Y</kbd> again — ' +
      'or press <kbd>Y</kbd> again now to record yes without marking any.';
    $('#markhint').classList.add('ask');
    $('#vy').classList.add('ask');
    return;
  }
  // Marks belong to a yes. A no or unsure with stray marks would be a
  // contradiction in the data, so they are dropped rather than stored.
  const m = v === 'y' ? marks.map(p => [+p.nx.toFixed(4), +p.ny.toFixed(4)]) : [];
  S.ans[f.id] = { v, s: +((performance.now() - S.t0)/1000).toFixed(1), m };
  save();
  flash({y:'var(--yes)', n:'var(--no)', u:'var(--unsure)'}[v]);
  S.i++;
  if (maybeCheckpoint()) return;
  show();
}
function flash(c){
  const el = $('#flash'); el.style.background = c; el.classList.add('on');
  setTimeout(() => el.classList.remove('on'), 105);
}
$('#vy').onclick = () => judge('y');
$('#vn').onclick = () => judge('n');
$('#vu').onclick = () => judge('u');
$('#back').onclick = goBack;
function goBack(){ if (S.i > 0) { S.i--; show(); } }

addEventListener('keydown', e => {
  if ($('#work').style.display === 'none') return;
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  const k = e.key.toLowerCase();
  if (k === 'y') judge('y');
  else if (k === 'n') judge('n');
  else if (k === 'u' || k === '?') judge('u');
  else if (e.key === 'Backspace') { e.preventDefault(); marks.pop(); drawMarks(); }
  else if (e.key === 'ArrowLeft') goBack();
  else if (k === '0' || k === 'r') resetView();
  else if (k === '=' || k === '+') { z = Math.min(6, z*1.2); clampPan(); applyView(); }
  else if (k === '-') { z = Math.max(1, z/1.2); clampPan(); applyView(); }
});

// ---- export
function csv(){
  const rows = [['round','session','field_id','reviewer','verdict','seconds',
                 'n_marks','marks','answered_utc']];
  const now = new Date().toISOString();
  FIELDS.forEach(f => {
    const a = S.ans[f.id];
    if (!a) return;
    const m = a.m || [];
    // "x1 y1; x2 y2" as ONE field, fractions of the tile. Semicolons so the
    // whole thing survives a spreadsheet round-trip without quoting rules.
    const pts = m.map(p => `${p[0]} ${p[1]}`).join('; ');
    rows.push([ROUND, SESSION, f.id, S.who, {y:'yes',n:'no',u:'unsure'}[a.v],
               a.s, m.length, pts, now]);
  });
  // Her words, carried in the same file so they cannot be separated from the
  // session they describe. Quoted because free text will contain commas.
  S.notes.forEach(n => {
    rows.push([ROUND, SESSION, `_note_at_${n.at}`, S.who, 'note', '', '',
               '"' + n.text.replace(/"/g, '""').replace(/[\r\n]+/g, ' ') + '"', now]);
  });
  return rows.map(r => r.join(',')).join('\n');
}

function filename(){
  const who = (S.who || 'reviewer').replace(/[^A-Za-z0-9_-]/g, '');
  const d = new Date().toISOString().slice(0, 10);
  return `ballooning_${who}_${d}_${SESSION}.csv`;
}

async function offer(){
  const text = csv(), name = filename(), msg = $('#doneMsg');
  if (!Object.keys(S.ans).length) { msg.textContent = 'Nothing answered yet.'; return; }

  // A normal browser -- including a file:// page opened from an email -- can
  // hand over a Blob directly. Inside the claude.ai viewer that is inert, so
  // the hosted copy has to ask the host instead. Try the host path only when
  // it exists, so the standalone file never depends on it.
  const host = window.claude && window.claude.use;
  if (!host) {
    try {
      const url = URL.createObjectURL(new Blob([text], {type:'text/csv'}));
      const a = document.createElement('a');
      a.href = url; a.download = name;
      document.body.appendChild(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 4000);
      msg.textContent = `Saved as ${name}. Please email that file back.`;
      return;
    } catch(e){ /* fall through to the copy box */ }
  } else {
    let dl = null;
    try { dl = await (window.claude.use('downloads') ?? null); } catch(e){ dl = null; }
    if (dl) {
      for (const n of [name, name.replace(/\.csv$/, '.txt')]) {
        try { await dl.save({filename:n, data:text});
              msg.textContent = `Saved as ${n}. Please email that file back.`; return; }
        catch(err){
          if (err && err.code === 'declined') { msg.textContent = 'Save cancelled.'; return; }
          if (err && (err.code === 'extension_not_enabled' ||
                      err.code === 'rejected_extension')) continue;
          break;
        }
      }
    }
  }
  showText(text, 'Could not save a file here — copy the text below and send it back.');
}

function showText(text, note){
  $('#out').style.display = 'block'; $('#out').value = text;
  $('#doneMsg').textContent = note; $('#out').select();
}

$('#dl').onclick = offer;
$('#copy').onclick = async () => {
  const text = csv();
  try { await navigator.clipboard.writeText(text);
        $('#doneMsg').textContent = 'Copied. Paste it into an email and send it back.'; }
  catch(e){ showText(text, 'Select the text below, copy it, and send it back.'); }
};
$('#saveNow').onclick = () => { finish(true); };
$('#again').onclick = () => {
  $('#done').style.display = 'none'; $('#work').style.display = 'flex';
  S.i = 0; show();
};

function finish(partial){
  $('#work').style.display = 'none';
  $('#done').style.display = 'grid';
  const v = Object.values(S.ans);
  const c = k => v.filter(a => a.v === k).length;
  $('#dy').textContent = c('y'); $('#dn').textContent = c('n'); $('#du').textContent = c('u');
  const secs = v.map(a => a.s).filter(x => x > 0).sort((a,b) => a-b);
  $('#dm').textContent = secs.length ? secs[Math.floor(secs.length/2)].toFixed(0) : '—';
  const nm = v.reduce((t, a) => t + ((a.m || []).length), 0);
  $('#dk').textContent = nm;
  // The >=5 threshold is the one Brunt et al. found roughly doubles expert
  // agreement over presence/absence, so it is worth seeing at a glance.
  $('#d5').textContent = v.filter(a => (a.m || []).length >= 5).length;
  $('#done').querySelector('.lede').textContent = partial
    ? `${v.length} of ${N} answered so far.`
    : `All ${N} fields answered.`;
}
