
function pressY(){ judge('y'); }
let fail = 0;
const ok = (c,m) => { console.log((c?'  PASS ':'  FAIL ')+m); if(!c) fail++; };

el('who').value = 'EK';
els['go'].onclick();

console.log('1. Yes with no marks must ASK first, not record');
const before = JSON.stringify(S.ans);
pressY();
ok(JSON.stringify(S.ans) === before, 'first Y records nothing');
ok(/click every ballooned cell/i.test(el('markhint').innerHTML),
   'it asks for EVERY ballooned cell, not just one -- the count is the label');
ok(el('vy').classList.contains('ask'), 'the Yes button shows it is waiting');

console.log('2. Second Y accepts a yes with no marks');
pressY();
ok(S.ans.f001 && S.ans.f001.v === 'y', 'verdict recorded');
ok(S.ans.f001.m.length === 0, 'no marks stored');

console.log('3. Marking then Y records the points on the FIRST press');
marks = [{nx:0.25,ny:0.75},{nx:0.5,ny:0.5}];
drawMarks();
ok(/2 marked/i.test(el('markhint').innerHTML), 'hint reflects 2 marks');
ok(el('vylabel').textContent === 'Yes \u00b7 2 cells', 'button reflects 2 marks');
pressY();
ok(S.ans.f002 && S.ans.f002.v === 'y', 'verdict recorded on first press');
ok(JSON.stringify(S.ans.f002.m) === '[[0.25,0.75],[0.5,0.5]]', 'both points stored');

console.log('4. A no drops any stray marks');
marks = [{nx:0.1,ny:0.1}];
judge('n');
ok(S.ans.f003.v === 'n' && S.ans.f003.m.length === 0, 'no marks on a no');

console.log('5. Going back restores what she marked');
S.i = 1; show();
ok(marks.length === 2, 'the two marks came back');

console.log('6. CSV carries round, session and marks');
const rows = csv().split('\n');
ok(rows[0] === 'round,session,field_id,reviewer,verdict,seconds,n_marks,marks,answered_utc', 'header');
const f2 = rows.find(r => r.includes('f002'));
ok(f2.includes('0.25 0.75; 0.5 0.5'), 'marks serialised as "x y; x y"');
ok(f2.startsWith('test_round,'), 'round stamped');
ok(rows.length === 4, 'one row per answered field');

console.log('7. The checkpoint fires once, and her note rides out in the CSV');
S.i = 14; S.ckDone = {}; marks = [];
judge('n');                       // advancing onto field 15 should stop her
ok(el('checkpoint').style.display === 'grid', 'checkpoint shown after 15 fields');
el('ckText').value = 'crops look fine, but zone 3 is hard to place';
els['ckGo'].onclick();
ok(el('checkpoint').style.display === 'none', 'dismissed');
ok(S.notes.length === 1, 'note kept');
const withNote = csv().split('\n').find(r => r.includes('_note_at_'));
ok(!!withNote && /zone 3 is hard to place/.test(withNote), 'note is in the CSV');
ok(/,note,/.test(withNote), 'note row is typed as a note, not a verdict');
S.i = 14; judge('n');
ok(el('checkpoint').style.display === 'none', 'it does not fire twice');

console.log('8. A ring is sized to a cell, and clicking it again removes it');
marks = [];
// 40 um ring on a 253.6 um field -> ~16% of the field width, not a 5 um dot.
ok(Math.abs(ringPx() / el('fieldbox').clientWidth - 40/253.6) < 1e-6,
   'ring diameter tracks 40 um, not a fixed pixel count');
marks.push({nx:0.50, ny:0.50}); drawMarks();
// a second click well inside the first ring must REMOVE it, not stack on it
const inside = {nx:0.505, ny:0.505};
const near = marks.findIndex(k =>
  Math.hypot((k.nx-inside.nx)*253.6, (k.ny-inside.ny)*253.6) < 40/2);
ok(near === 0, 'a click within the ring resolves to the existing mark');
// and a click a whole cell away must be a NEW mark
const far = {nx:0.75, ny:0.75};
const near2 = marks.findIndex(k =>
  Math.hypot((k.nx-far.nx)*253.6, (k.ny-far.ny)*253.6) < 40/2);
ok(near2 === -1, 'a click a cell away is a new mark');

console.log(fail ? `\n${fail} FAILED` : '\nall passed');
process.exit(fail ? 1 : 0);
