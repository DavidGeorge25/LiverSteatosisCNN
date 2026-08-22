let fail=0; const ok=(c,m)=>{console.log((c?'  PASS ':'  FAIL ')+m); if(!c)fail++;};

el('who').value='EK'; els['go'].onclick();

console.log('1. Yes with no circles asks first');
const before=JSON.stringify(S.fields);
judge('y');
ok(JSON.stringify(S.fields)===before,'nothing recorded on first Y');
ok(/Circle each ballooned cell/.test(el('markhint').innerHTML),'it asks for circles');
judge('y');
ok(S.fields['s0_f0'] && S.fields['s0_f0'].v==='y','second Y records');

console.log('2. Circles carry centre AND diameter');
circles=[{nx:0.4,ny:0.6,d:52},{nx:0.7,ny:0.3,d:38}];
drawCircles();
ok(el('vylabel').textContent==='Yes · 2 cells','button shows the count');
judge('y');
const c=S.fields['s0_f1'].c;
ok(JSON.stringify(c)==='[[0.4,0.6,52],[0.7,0.3,38]]','centre + diameter stored');

console.log('3. A no drops circles');
circles=[{nx:0.1,ny:0.1,d:40}]; judge('n');
ok(S.fields['s0_f2'].c.length===0,'no circles on a no');

console.log('4. Finishing a section opens the grade screen, not the next field');
circles=[]; judge('n');
ok(el('grade').style.display==='grid','grade screen opened after last field');
ok(S.si===0,'still on the same section until graded');

console.log('5. Grading advances to the next section');
el('noteBox').value='mostly zone 3';
setGrade('2');
ok(S.grades['s0'].g==='2' && S.grades['s0'].note==='mostly zone 3','grade + note stored');
ok(S.si===1 && S.fi===0,'advanced to next section');

console.log('6. Checkpoint fires after section 2, once');
['n','n','n','n'].forEach(v=>{circles=[];judge(v);}); setGrade('1');
ok(el('checkpoint').style.display==='grid','checkpoint after 2 sections');
el('ckText').value='circling is fine'; els['ckGo'].onclick();
ok(S.notes.length===1,'note captured');
ok(el('checkpoint').style.display==='none','dismissed');

console.log('7. Back crosses the section boundary');
const si=S.si, fi=S.fi; goBack();
ok(S.si===si && S.fi===fi-1 || (fi===0 && S.si===si-1),'back went to the previous field');

console.log('8. Resume lands on the first unfinished field');
S.si=0; S.fi=0; resume();
ok(S.si===2,'resumed at the first section with work left');

console.log('9. The X badge deletes the circle it belongs to');
circles=[{nx:0.2,ny:0.2,d:40},{nx:0.5,ny:0.5,d:60},{nx:0.8,ny:0.8,d:30}];
drawCircles();
const html = el('marks').innerHTML;
ok((html.match(/data-x=/g)||[]).length===3,'every circle carries an X');
ok(/\u00d7/.test(html),'the X is a visible glyph, not a bare handler');
// delete the middle one the way the handler does
circles.splice(1,1); sel=-1; drawCircles();
ok(circles.length===2 && circles[1].d===30,'the right circle went');
ok(/>1</.test(el('marks').innerHTML) && /># /.test('# ')===false,'numbering redrawn');

console.log('10. With no File System Access API it degrades and says so');
ok(canPickFile===false,'harness has no picker, as Safari would not');
syncMsg('not saving to a file — use "Save a copy"', true);
ok(/Save a copy/.test(el('sync').textContent),'the fallback is stated, not silent');

console.log('11. CSV carries fields, grades and notes as distinct records');
const rows=csv().split('\n');
ok(rows[0].startsWith('round,session,reviewer,record,'),'header');
// Match the record COLUMN, not the substring: the header itself contains
// "verdict,grade,n_cells", so a naive includes(',grade,') counts it as a row.
const rec = r => r.split(',')[3];
const body = rows.slice(1);
ok(/0\.4 0\.6 52; 0\.7 0\.3 38/.test(rows.find(r=>r.includes('s0_f1'))),'cells serialised x y d');
ok(body.some(r=>rec(r)==='grade'),'grade rows present');
ok(body.some(r=>rec(r)==='note'),'note rows present');
ok(body.filter(r=>rec(r)==='grade').length===2,'one grade row per graded section');
ok(body.filter(r=>rec(r)==='field').length===Object.keys(S.fields).length,
   'one field row per answered field');

console.log('12. A repeated field is judged twice but shown once in the montage');
S.si=2; S.fi=0;
ok(cur().fields.length===5,'section 2 presents 5 fields including the repeat');
ok(uniqueFields(cur()).length===4,'the montage deduplicates by image');

// Work the section for real so the montage has something of hers to show.
circles=[{nx:0.3,ny:0.3,d:44},{nx:0.6,ny:0.7,d:38}]; judge('y');   // f0: 2 cells
circles=[]; judge('n');                                            // f1: none
circles=[{nx:0.5,ny:0.2,d:50}]; judge('y');                        // f2: 1 cell
circles=[]; judge('u');                                            // f3: unsure
circles=[]; judge('n');                                            // f4: the repeat

console.log('12b. The grade screen shows the cells she just circled');
ok(el('grade').style.display==='grid','grade opened after the last field');
ok((el('gg').innerHTML.match(/<img /g)||[]).length===4,'4 thumbnails, not 5');
ok(String(el('gN').textContent)==='4','the count she is told matches what she sees');
const marks=(el('gg').innerHTML.match(/<i style=/g)||[]).length;
ok(marks===3,`her 3 circles are drawn on the montage (got ${marks})`);
ok(/left:30\.00%/.test(el('gg').innerHTML),'a mark is placed by percentage, not pixels');
ok(/<b>2<\/b>/.test(el('gg').innerHTML) && /<b>1<\/b>/.test(el('gg').innerHTML),
   'each thumbnail carries its own cell count');
ok(/<u>unsure<\/u>/.test(el('gg').innerHTML),
   'an unsure field is flagged -- unreadable is not the same as empty');
el('grade').style.display='none';

console.log('13. Saving a copy mid-session leaves a way back');
finish(true);
ok(el('done').style.display==='grid','done screen shown');
ok(el('carryOn').style.display==='','carry-on offered on a partial save');
els['carryOn'].onclick();
ok(el('work').style.display==='flex','back at work');
finish(false);
ok(el('carryOn').style.display==='none','not offered when everything is complete');

console.log(fail?`\n${fail} FAILED`:'\nall passed');
process.exit(fail?1:0);
