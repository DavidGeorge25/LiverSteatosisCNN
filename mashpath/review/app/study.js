const $ = s => document.querySelector(s);
const SECTIONS = JSON.parse($('#data').textContent);
const KEY = 'ballooning-study-v1';
const TILE_UM = 253.6, DEF_UM = 40, MIN_UM = 12, MAX_UM = 140;
const SESSION = (Date.now().toString(36)+Math.random().toString(36).slice(2,7)).toUpperCase();
const NSEC = SECTIONS.length;
const NFIELD = SECTIONS.reduce((t,s) => t + s.fields.length, 0);

let S = { who:'', si:0, fi:0, fields:{}, grades:{}, notes:[], ck:{}, t0:0 };

// ---- the results file -----------------------------------------------------
// She picks a real file once; every answer is written straight to it. The
// browser's own storage is kept as a second copy, but it is not the record --
// localStorage can be cleared by the browser without warning, and on a page
// opened straight from disk some browsers refuse it outright.
//
// Needs the File System Access API (Chrome/Edge). Safari and Firefox do not
// have it, so there the page falls back to browser storage plus a manual
// download, and SAYS so rather than pretending it is syncing.
let fileHandle = null, writing = false, dirty = false;
const canPickFile = typeof window.showSaveFilePicker === 'function';

function idb(mode, fn){
  return new Promise((res, rej) => {
    const r = indexedDB.open('ballooning-study', 1);
    r.onupgradeneeded = () => r.result.createObjectStore('kv');
    r.onerror = () => rej(r.error);
    r.onsuccess = () => {
      const tx = r.result.transaction('kv', mode);
      const out = fn(tx.objectStore('kv'));
      tx.oncomplete = () => res(out && out.result !== undefined ? out.result : out);
      tx.onerror = () => rej(tx.error);
    };
  });
}
const saveHandle = h => idb('readwrite', st => st.put(h, 'file')).catch(()=>{});
const loadHandle = () => idb('readonly', st => st.get('file')).catch(()=>null);

function syncMsg(text, warn){
  const el = $('#sync');
  el.style.display = text ? '' : 'none';
  el.textContent = text;
  el.style.background = warn ? 'transparent' : '';
  el.style.color = warn ? 'var(--no)' : '';
  $('#pick').style.display = (canPickFile && !fileHandle) ? '' : 'none';
}

async function pickFile(){
  if (!canPickFile) return false;
  const name = `ballooning_${(S.who||'reviewer').replace(/[^A-Za-z0-9_-]/g,'')}.csv`;
  try {
    fileHandle = await window.showSaveFilePicker({
      suggestedName: name,
      types: [{description:'CSV', accept:{'text/csv':['.csv']}}],
    });
    await saveHandle(fileHandle);
    await writeFile();
    return true;
  } catch(e){ fileHandle = null; return false; }
}

async function ensurePermission(h){
  if (!h || !h.queryPermission) return false;
  const opt = {mode:'readwrite'};
  if (await h.queryPermission(opt) === 'granted') return true;
  try { return await h.requestPermission(opt) === 'granted'; } catch(e){ return false; }
}

async function writeFile(){
  if (!fileHandle) return;
  if (writing) { dirty = true; return; }        // coalesce; never interleave writes
  writing = true;
  try {
    const w = await fileHandle.createWritable();
    await w.write(csv());
    await w.close();
    syncMsg(`saved to ${fileHandle.name}`);
  } catch(e){
    syncMsg('not saving to file — use "Save a copy"', true);
    fileHandle = null;
  } finally {
    writing = false;
    if (dirty) { dirty = false; writeFile(); }
  }
}
$('#pick').onclick = async () => { if (await pickFile()) syncMsg(`saved to ${fileHandle.name}`); };
let storageOK = true;
try { localStorage.setItem('_t','1'); localStorage.removeItem('_t'); } catch(e){ storageOK=false; }

function save(){
  try { localStorage.setItem(KEY, JSON.stringify(
    {who:S.who, fields:S.fields, grades:S.grades, notes:S.notes, ck:S.ck})); } catch(e){}
  writeFile();          // the file is the record; storage is the backup
}
function load(){ try { return JSON.parse(localStorage.getItem(KEY)||'null'); } catch(e){ return null; } }

const sv0 = load();
if (sv0 && sv0.who) {
  $('#who').value = sv0.who;
  const n = Object.keys(sv0.fields||{}).length;
  if (n) { $('#introMsg').textContent = `Welcome back — ${n} of ${NFIELD} fields done.`;
           $('#reset').style.display=''; }
}
$('#reset').onclick = () => {
  if (!confirm('Delete all your work so far?')) return;
  localStorage.removeItem(KEY); S.fields={}; S.grades={}; S.notes=[]; S.ck={};
  $('#introMsg').textContent='Cleared.'; $('#reset').style.display='none';
};
$('#go').onclick = async () => {
  const who = $('#who').value.trim();
  if (!who) { $('#introMsg').textContent='Please enter your initials first.'; $('#who').focus(); return; }
  S.who = who;
  const sv = load();
  if (sv && sv.who === who) {
    S.fields=sv.fields||{}; S.grades=sv.grades||{}; S.notes=sv.notes||[]; S.ck=sv.ck||{};
  }

  // Reconnect to the file she chose last time if we still have the handle;
  // the browser drops the permission between visits, so it must be re-granted.
  // Both calls have to happen inside this click -- a permission prompt raised
  // later, outside a user gesture, is refused.
  if (canPickFile) {
    const prev = await loadHandle().catch(()=>null);
    if (prev && await ensurePermission(prev)) {
      fileHandle = prev;
    } else {
      $('#introMsg').textContent = 'Choose where to keep your results file…';
      await pickFile();
    }
    syncMsg(fileHandle ? `saving to ${fileHandle.name}`
                       : 'not saving to a file — use "Save a copy"', !fileHandle);
  } else {
    syncMsg('this browser cannot save to a file — use "Save a copy"', true);
  }

  resume();
  $('#intro').style.display='none'; $('#work').style.display='flex';
  $('#sTot').textContent = NSEC;
  show();
  writeFile();
};
function resume(){
  // First section with an ungraded or unfinished field, so a reopened session
  // lands where she stopped rather than at the beginning.
  for (let i=0;i<NSEC;i++){
    const sec = SECTIONS[i];
    const j = sec.fields.findIndex(f => !S.fields[f.id]);
    if (j >= 0) { S.si=i; S.fi=j; return; }
    if (!S.grades[sec.id]) { S.si=i; S.fi=sec.fields.length; return; }
  }
  S.si = NSEC; S.fi = 0;
}

// ---- circles: centre + diameter, both in tile fractions / microns ---------
let circles = [], sel = -1, z = 1, ox = 0, oy = 0;
let mode = null, startPt = null, moved = 0;
const fb = $('#fieldbox');

function umToPx(um){ return (um / TILE_UM) * fb.clientWidth * z; }
function toTile(cx, cy){
  const w = fb.clientWidth, h = fb.clientHeight;
  return { nx:(cx-ox)/(z*w), ny:(cy-oy)/(z*h) };
}
function drawCircles(){
  const w = fb.clientWidth, h = fb.clientHeight;
  $('#marks').innerHTML = circles.map((c,i) => {
    const d = umToPx(c.d), x = c.nx*w*z+ox, y = c.ny*h*z+oy;
    return `<div class="mk${i===sel?' sel':''}" data-i="${i}" style="left:${x-d/2}px;`
         + `top:${y-d/2}px;width:${d}px;height:${d}px">`
         + `<b>${i+1}</b><u data-x="${i}" title="Remove this circle">\u00d7</u>`
         + `<i data-h="${i}"></i></div>`;
  }).join('');
  const n = circles.length;
  $('#markn').style.display = n?'':'none'; $('#markn').textContent = n;
  $('#undo').style.display = n?'':'none';
  $('#vylabel').textContent = n ? (n===1?'Yes · 1 cell':`Yes · ${n} cells`) : 'Yes';
  if (n) armedOff();
  $('#markhint').innerHTML = n
    ? `<strong>${n} circled.</strong> Drag to move, drag the corner to resize, `
      + `<kbd>Delete</kbd> to remove. Press <kbd>Y</kbd> when done.`
    : 'Click each ballooned cell to circle it.';
}
function applyView(){
  $('#field').style.transform = `translate(${ox}px,${oy}px) scale(${z})`;
  $('#fscale').textContent = z>1.02 ? `253.6 µm across · ${z.toFixed(1)}×` : '253.6 µm across';
  drawCircles();
}
function resetView(){ z=1; ox=0; oy=0; applyView(); }
function clampPan(){
  const w=fb.clientWidth,h=fb.clientHeight;
  ox=Math.min(0,Math.max(w-w*z,ox)); oy=Math.min(0,Math.max(h-h*z,oy));
}
fb.addEventListener('wheel', e => {
  e.preventDefault();
  const r=fb.getBoundingClientRect(), mx=e.clientX-r.left, my=e.clientY-r.top;
  const nz=Math.min(6,Math.max(1,z*(e.deltaY<0?1.16:1/1.16)));
  ox=mx-(mx-ox)*(nz/z); oy=my-(my-oy)*(nz/z); z=nz; clampPan(); applyView();
}, {passive:false});

fb.addEventListener('pointerdown', e => {
  const r = fb.getBoundingClientRect();
  const px = e.clientX-r.left, py = e.clientY-r.top;
  startPt = {x:e.clientX, y:e.clientY, px, py}; moved = 0;
  fb.setPointerCapture(e.pointerId);
  const del = e.target.closest('u[data-x]');
  if (del) {
    // Handled on pointerdown and swallowed, so the X never also selects or
    // starts a drag on the ring beneath it.
    e.preventDefault(); e.stopPropagation();
    circles.splice(+del.dataset.x, 1); sel = -1; mode = null; startPt = null;
    drawCircles(); return;
  }
  const h = e.target.closest('i[data-h]');
  if (h) { sel = +h.dataset.h; mode = 'resize'; drawCircles(); return; }
  const m = e.target.closest('.mk');
  if (m) { sel = +m.dataset.i; mode = 'move'; drawCircles(); return; }
  // Empty space: pan when zoomed in, otherwise this will become a new circle
  // on pointerup if the pointer did not travel.
  mode = z > 1 ? 'pan' : 'maybe';
  sel = -1; drawCircles();
});
fb.addEventListener('pointermove', e => {
  if (!mode || !startPt) return;
  const dx = e.clientX-startPt.x, dy = e.clientY-startPt.y;
  moved += Math.abs(dx)+Math.abs(dy);
  if (mode==='pan'){ ox+=dx; oy+=dy; startPt.x=e.clientX; startPt.y=e.clientY; clampPan(); applyView(); }
  else if (mode==='move' && sel>=0){
    const w=fb.clientWidth,h=fb.clientHeight;
    circles[sel].nx += dx/(z*w); circles[sel].ny += dy/(z*h);
    circles[sel].nx=Math.max(0,Math.min(1,circles[sel].nx));
    circles[sel].ny=Math.max(0,Math.min(1,circles[sel].ny));
    startPt.x=e.clientX; startPt.y=e.clientY; drawCircles();
  } else if (mode==='resize' && sel>=0){
    const r=fb.getBoundingClientRect();
    const c=circles[sel], cx=c.nx*fb.clientWidth*z+ox, cy=c.ny*fb.clientHeight*z+oy;
    const dist=Math.hypot((e.clientX-r.left)-cx,(e.clientY-r.top)-cy);
    const um=(dist*2)/(fb.clientWidth*z)*TILE_UM;
    c.d=Math.max(MIN_UM,Math.min(MAX_UM,um)); drawCircles();
  }
});
['pointerup','pointercancel'].forEach(ev => fb.addEventListener(ev, e => {
  if (mode==='maybe' && moved<=6 && startPt){
    const m=toTile(startPt.px,startPt.py);
    if (m.nx>=0&&m.nx<=1&&m.ny>=0&&m.ny<=1){ circles.push({nx:m.nx,ny:m.ny,d:DEF_UM});
      sel=circles.length-1; drawCircles(); }
  }
  mode=null; startPt=null;
}));
$('#undo').onclick = () => { circles.pop(); sel=-1; drawCircles(); };

// ---- the field ------------------------------------------------------------
let armed = false;
function armedOff(){ armed=false; $('#vy').classList.remove('ask'); }
function cur(){ return SECTIONS[S.si]; }

function show(){
  if (S.si >= NSEC) return finish();
  const sec = cur();
  if (S.fi >= sec.fields.length) return openGrade();
  const f = sec.fields[S.fi];
  $('#field').src = f.i; $('#ctx').src = f.c;
  $('#sPos').textContent = S.si+1; $('#fPos').textContent = S.fi+1;
  $('#fTot').textContent = sec.fields.length;
  const doneN = Object.keys(S.fields).length;
  $('#prog').style.width = (doneN/NFIELD*100)+'%';
  const prev = S.fields[f.id];
  circles = prev && prev.c ? prev.c.map(a=>({nx:a[0],ny:a[1],d:a[2]})) : [];
  sel=-1; armedOff(); resetView(); tally(); S.t0=performance.now();
}
function tally(){
  const v=Object.values(S.fields);
  $('#tf').textContent=v.length;
  $('#tc').textContent=v.reduce((t,a)=>t+(a.c?a.c.length:0),0);
  const secs=v.map(a=>a.s).filter(x=>x>0).sort((a,b)=>a-b);
  $('#pace').textContent = secs.length>=5 ? `${secs[secs.length>>1].toFixed(0)}s/field` : '';
}
function judge(v){
  const f = cur().fields[S.fi];
  if (v==='y' && !circles.length && !armed){
    armed=true; $('#vy').classList.add('ask');
    $('#markhint').innerHTML='<strong>Circle each ballooned cell</strong>, then press '
      +'<kbd>Y</kbd> again — or press <kbd>Y</kbd> again now to record yes without circling.';
    return;
  }
  S.fields[f.id] = { v, s:+((performance.now()-S.t0)/1000).toFixed(1),
    c: v==='y' ? circles.map(c=>[+c.nx.toFixed(4),+c.ny.toFixed(4),Math.round(c.d)]) : [] };
  save(); flash({y:'var(--yes)',n:'var(--no)',u:'var(--unsure)'}[v]);
  S.fi++; show();
}
function flash(c){ const e=$('#flash'); e.style.background=c; e.classList.add('on');
  setTimeout(()=>e.classList.remove('on'),105); }
$('#vy').onclick=()=>judge('y'); $('#vn').onclick=()=>judge('n'); $('#vu').onclick=()=>judge('u');
$('#back').onclick=goBack;
function goBack(){
  if (S.fi>0){ S.fi--; show(); }
  else if (S.si>0){ S.si--; S.fi=cur().fields.length-1; show(); }
}

// ---- section grade --------------------------------------------------------
function uniqueFields(sec){
  const seen=new Set();
  return sec.fields.filter(f => !seen.has(f.i) && seen.add(f.i));
}
// Her circles, as percentages of the field, so one string of markup works at
// thumbnail size and at lightbox size without recomputing anything.
function markHTML(cells){
  return (cells||[]).map(c=>{
    const w=c[2]/TILE_UM*100;
    return `<i style="left:${(c[0]*100).toFixed(2)}%;top:${(c[1]*100).toFixed(2)}%;`
         + `width:${w.toFixed(2)}%;height:${w.toFixed(2)}%"></i>`;
  }).join('');
}
function openGrade(){
  const sec=cur(), shown=uniqueFields(sec);
  $('#gN').textContent=shown.length;
  $('#gg').innerHTML=shown.map((f,j)=>{
    const a=S.fields[f.id]||{}, n=(a.c||[]).length;
    // "unsure" is worth seeing at grading time: a section she could not read is
    // a different thing from a section with nothing in it.
    const tag = a.v==='u' ? '<u>unsure</u>' : '';
    return `<div class="gt" data-f="${f.id}" data-src="${f.i}">`
         + `<img src="${f.i}" alt="Field ${j+1}" loading="lazy">`
         + markHTML(a.c) + (n?`<b>${n}</b>`:'') + tag + `</div>`;
  }).join('');
  $('#gg').querySelectorAll('.gt').forEach(el=>{
    el.onclick=()=>{
      const a=S.fields[el.dataset.f]||{};
      $('#lbimg').src=el.dataset.src;
      $('#lbmarks').innerHTML=markHTML(a.c);
      $('#lightbox').style.display='grid';
    };
  });
  $('#noteBox').value=(S.grades[sec.id]&&S.grades[sec.id].note)||'';
  $('#grade').style.display='grid';
}
function setGrade(g){
  const sec=cur();
  S.grades[sec.id]={ g, note:$('#noteBox').value.trim() };
  save(); $('#grade').style.display='none';
  S.si++; S.fi=0;
  if (maybeCheckpoint()) return;
  show();
}
$('#g0').onclick=()=>setGrade('0'); $('#g1').onclick=()=>setGrade('1');
$('#g2').onclick=()=>setGrade('2'); $('#gu').onclick=()=>setGrade('unsure');
$('#lightbox').onclick=()=>{ $('#lightbox').style.display='none'; };

// ---- checkpoint -----------------------------------------------------------
const CHECKPOINTS=[2,10,20];
function maybeCheckpoint(){
  if (!CHECKPOINTS.includes(S.si) || S.ck[S.si]) return false;
  S.ck[S.si]=true; $('#ckN').textContent=S.si;
  $('#checkpoint').style.display='grid'; $('#ckText').focus(); return true;
}
$('#ckGo').onclick=()=>{
  const t=$('#ckText').value.trim();
  if (t) S.notes.push({at:S.si,text:t});
  $('#ckText').value=''; save(); $('#checkpoint').style.display='none'; show();
};

addEventListener('keydown', e=>{
  if ($('#work').style.display==='none') return;
  if (e.target.tagName==='INPUT'||e.target.tagName==='TEXTAREA') return;
  if ($('#lightbox').style.display==='grid'){ $('#lightbox').style.display='none'; return; }
  if ($('#checkpoint').style.display==='grid') return;
  const k=e.key.toLowerCase();
  if ($('#grade').style.display==='grid'){
    if (k==='0') setGrade('0'); else if (k==='1') setGrade('1');
    else if (k==='2') setGrade('2'); else if (k==='u') setGrade('unsure');
    return;
  }
  if (k==='y') judge('y');
  else if (k==='n') judge('n');
  else if (k==='u'||k==='?') judge('u');
  else if (e.key==='Delete'||e.key==='Backspace'){
    e.preventDefault();
    if (sel>=0){ circles.splice(sel,1); sel=-1; } else circles.pop();
    drawCircles();
  }
  else if (e.key==='Escape'){ sel=-1; drawCircles(); }
  else if (e.key==='ArrowLeft') goBack();
  else if (k==='0'||k==='r') resetView();
  else if (k==='='||k==='+'){ z=Math.min(6,z*1.2); clampPan(); applyView(); }
  else if (k==='-'){ z=Math.max(1,z/1.2); clampPan(); applyView(); }
});

// ---- export ---------------------------------------------------------------
function csv(){
  const rows=[['round','session','reviewer','record','section_id','field_id',
               'verdict','grade','n_cells','cells','seconds','note','utc']];
  const now=new Date().toISOString();
  const q=t=>'"'+String(t||'').replace(/"/g,'""').replace(/[\r\n]+/g,' ')+'"';
  SECTIONS.forEach(sec=>{
    sec.fields.forEach(f=>{
      const a=S.fields[f.id]; if(!a) return;
      // Cells as "x y d; x y d" -- centre in fractions of the field, diameter in um.
      const cells=(a.c||[]).map(c=>`${c[0]} ${c[1]} ${c[2]}`).join('; ');
      rows.push([ROUND,SESSION,S.who,'field',sec.id,f.id,
                 {y:'yes',n:'no',u:'unsure'}[a.v],'',(a.c||[]).length,q(cells),a.s,'',now]);
    });
    const g=S.grades[sec.id];
    if (g) rows.push([ROUND,SESSION,S.who,'grade',sec.id,'','',g.g,'','','',q(g.note),now]);
  });
  S.notes.forEach(n=>rows.push([ROUND,SESSION,S.who,'note','','','','','','','',q(n.text),now]));
  return rows.map(r=>r.join(',')).join('\n');
}
function fname(){
  const w=(S.who||'reviewer').replace(/[^A-Za-z0-9_-]/g,'');
  return `ballooning_${w}_${new Date().toISOString().slice(0,10)}_${SESSION}.csv`;
}
function showText(t,note){ $('#out').style.display='block'; $('#out').value=t;
  $('#doneMsg').textContent=note; $('#out').select(); }
async function offer(){
  const text=csv(), name=fname();
  if (!Object.keys(S.fields).length){ $('#doneMsg').textContent='Nothing recorded yet.'; return; }
  const host=window.claude&&window.claude.use;
  if (!host){
    try{
      const url=URL.createObjectURL(new Blob([text],{type:'text/csv'}));
      const a=document.createElement('a'); a.href=url; a.download=name;
      document.body.appendChild(a); a.click(); a.remove();
      setTimeout(()=>URL.revokeObjectURL(url),4000);
      $('#doneMsg').textContent=`Saved as ${name}. Please email that file back.`; return;
    }catch(e){}
  } else {
    let dl=null; try{ dl=await (window.claude.use('downloads')??null); }catch(e){}
    if (dl) for (const n of [name,name.replace(/\.csv$/,'.txt')]){
      try{ await dl.save({filename:n,data:text});
           $('#doneMsg').textContent=`Saved as ${n}. Please email that file back.`; return; }
      catch(err){ if(err&&err.code==='declined'){$('#doneMsg').textContent='Save cancelled.';return;}
                  if(err&&(err.code==='extension_not_enabled'||err.code==='rejected_extension'))continue;
                  break; } }
  }
  showText(text,'Could not save a file here — copy the text below and send it back.');
}
$('#dl').onclick=offer;
$('#copy').onclick=async()=>{ try{ await navigator.clipboard.writeText(csv());
  $('#doneMsg').textContent='Copied. Paste into an email and send it back.'; }
  catch(e){ showText(csv(),'Select the text below, copy it, and send it back.'); } };
$('#saveNow').onclick=()=>{ writeFile(); finish(true); };

$('#carryOn').onclick=()=>{
  $('#done').style.display='none'; $('#work').style.display='flex'; show();
};
function finish(partial){
  $('#work').style.display='none'; $('#done').style.display='grid';
  $('#carryOn').style.display = partial ? '' : 'none';
  const v=Object.values(S.fields);
  $('#dF').textContent=v.length;
  $('#dC').textContent=v.reduce((t,a)=>t+(a.c?a.c.length:0),0);
  $('#dG').textContent=Object.keys(S.grades).length;
  const secs=v.map(a=>a.s).filter(x=>x>0).sort((a,b)=>a-b);
  $('#dM').textContent=secs.length?secs[secs.length>>1].toFixed(0):'—';
  $('#doneLede').textContent = partial
    ? `${v.length} of ${NFIELD} fields and ${Object.keys(S.grades).length} of ${NSEC} sections so far.`
    : 'All sections complete.';
}
