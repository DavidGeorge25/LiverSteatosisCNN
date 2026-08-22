// Minimal DOM stub: enough to run the page's real app.js and drive the flow.
const els = {};
function el(id){
  if (!els[id]) els[id] = {
    id, textContent:'', innerHTML:'', value:'', style:{}, src:'',
    classList:{ c:new Set(), add(x){this.c.add(x)}, remove(x){this.c.delete(x)},
                contains(x){return this.c.has(x)} },
    clientWidth:600, clientHeight:600, naturalWidth:512, naturalHeight:512,
    addEventListener(){}, getBoundingClientRect(){return {left:0,top:0}},
    setPointerCapture(){}, select(){}, focus(){},
    querySelector(sel){ return el(id + sel); },
  };
  return els[id];
}
global.document = {
  querySelector: s => el(s.replace('#','')),
  getElementById: id => el(id),
  createElement: () => ({ style:{}, click(){}, remove(){}, setAttribute(){} }),
  body: { appendChild(){}, },
};
global.window = { addEventListener(){} };
global.addEventListener = () => {};
global.performance = { now: () => Date.now() };
global.localStorage = { store:{}, setItem(k,v){this.store[k]=v},
  getItem(k){return this.store[k]??null}, removeItem(k){delete this.store[k]} };
global.URL = { createObjectURL: () => 'blob:x', revokeObjectURL(){} };
global.Blob = class {};
global.navigator = { clipboard:{ writeText: async()=>{} } };
global.confirm = () => true;
global.setTimeout = (f)=>f;

// 20 fields, not 3: the checkpoint fires at 15, so a shorter fixture cannot
// reach it and the assertion would pass by never running.
const FIELDS = Array.from({length:20}, (_, i) =>
  ({ id:`f${String(i+1).padStart(3,'0')}`, img:`i${i}` }));
const IMGS = Object.fromEntries(FIELDS.map((f,i) => [f.img, {t:'AAA'+i, c:'BBB'+i}]));
el('data').textContent = JSON.stringify(FIELDS);
el('imgs').textContent = JSON.stringify(IMGS);
el('work').style.display = 'flex';

global.ROUND = 'test_round';
const fs = require('fs');
const src = fs.readFileSync(process.argv[2], 'utf8');
const drive = fs.readFileSync(__dirname + '/flow.js', 'utf8');
// One eval, not two: `let`/`const` declared inside an eval do not leak into the
// caller's scope, so the assertions have to share it with the code under test.
eval(src + '\n' + drive);

