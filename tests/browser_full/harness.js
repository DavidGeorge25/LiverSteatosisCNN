const els = {};
function el(id){
  if(!els[id]) els[id]={ id, textContent:'', innerHTML:'', value:'', style:{}, src:'',
    classList:{c:new Set(),add(x){this.c.add(x)},remove(x){this.c.delete(x)},contains(x){return this.c.has(x)}},
    clientWidth:600, clientHeight:600, naturalWidth:512, naturalHeight:512,
    addEventListener(){}, getBoundingClientRect(){return {left:0,top:0}},
    setPointerCapture(){}, select(){}, focus(){},
    querySelector(s){return el(id+s)}, querySelectorAll(){return []} };
  return els[id];
}
global.document={ querySelector:s=>el(s.replace('#','')), getElementById:id=>el(id),
  createElement:()=>({style:{},click(){},remove(){},setAttribute(){}}), body:{appendChild(){}} };
global.window={addEventListener(){}}; global.addEventListener=()=>{};
global.performance={now:()=>Date.now()};
global.localStorage={store:{},setItem(k,v){this.store[k]=v},getItem(k){return this.store[k]??null},
  removeItem(k){delete this.store[k]}};
global.URL={createObjectURL:()=>'blob:x',revokeObjectURL(){}};
global.Blob=class{}; global.navigator={clipboard:{writeText:async()=>{}}};
global.confirm=()=>true; global.setTimeout=f=>f();
// No File System Access API and no IndexedDB in the harness: the page must run
// exactly as it does in Safari, i.e. fall back rather than throw.
global.indexedDB={open(){const r={};setTimeout(()=>r.onerror&&r.onerror());return r;}};

// 3 sections x 4 fields: enough to exercise section advance, grading and a checkpoint.
// Section 2 carries a fifth field that REPEATS the image of its first -- that is
// what a real repeat looks like on the wire (own id, same image), and it is what
// the montage has to deduplicate.
const SECTIONS=[0,1,2].map(i=>({id:`s${i}`,
  fields:[0,1,2,3].map(j=>({id:`s${i}_f${j}`,i:`images/${i}_${j}.jpg`,c:`context/${i}_${j}.jpg`}))}));
SECTIONS[2].fields.push({id:'s2_f4',i:'images/2_0.jpg',c:'context/2_0.jpg'});
el('data').textContent=JSON.stringify(SECTIONS);
el('work').style.display='flex'; el('grade').style.display='none';
el('checkpoint').style.display='none'; el('lightbox').style.display='none';
global.ROUND='test';
const fs=require('fs');
eval(fs.readFileSync(process.argv[2],'utf8')+'\n'+fs.readFileSync(__dirname+'/flow.js','utf8'));
