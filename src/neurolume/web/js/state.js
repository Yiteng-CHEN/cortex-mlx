const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const MONO="12px 'IBM Plex Mono',ui-monospace,monospace";
let NL=40, G=12;
const KNOWN={ // 来自 neurolume models inspect 的实测结构；L=线性注意力层，A=全注意力层
  'devstral-small-2':{n:40,pat:'A'.repeat(40),label:'Devstral Small 2 · 40 层 · 全部为标准注意力'},
  'qwen3.6-27b':{n:64,pat:'LLLA'.repeat(16),label:'Qwen3.6-27B · 64 层 = 16 ×（3 线性注意力 + 1 全注意力）'}};
let PAT=KNOWN['devstral-small-2'].pat;
const MATS=[
  {name:'self_attn.q_proj',label:'注意力 Q',shape:[4096,5120]},
  {name:'self_attn.o_proj',label:'注意力 输出',shape:[5120,4096]},
  {name:'mlp.gate_proj',label:'MLP 门控',shape:[32768,5120]},
  {name:'mlp.down_proj',label:'MLP 下投影',shape:[5120,32768]}];
const pick=(n,k)=>Math.round(k*(n-1)/(G-1));
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x)), sstep=(a,b,x)=>{const t=clamp((x-a)/(b-a),0,1);return t*t*(3-2*t)};
function rng(seed){let s=seed>>>0;return()=>((s=Math.imul(s^s>>>15,1|s)+0x6D2B79F5>>>0),((s^s>>>7)>>>0)/4294967296)}
// 医学扫描常用的热度色标：蓝 → 青 → 绿 → 黄 → 红
const STOPS=[[0,[30,60,200]],[.25,[8,200,220]],[.5,[88,220,88]],[.75,[249,225,44]],[1,[255,58,44]]];
function heat(e){e=clamp(e,0,1);for(let i=1;i<STOPS.length;i++){if(e<=STOPS[i][0]){const[a,ca]=STOPS[i-1],[b,cb]=STOPS[i],t=(e-a)/(b-a);return[ca[0]+(cb[0]-ca[0])*t,ca[1]+(cb[1]-ca[1])*t,ca[2]+(cb[2]-ca[2])*t]}}return STOPS[4][1]}

// ======== 运行状态 ========
let LIVE=false, SIM=true, active=null, busy=false;
let act=new Float32Array(NL);          // 平滑后的显示强度 0..1
let lo=new Float32Array(NL).fill(1e9), hi=new Float32Array(NL).fill(-1e9);
// 显示方式：增强（默认，放大差别）/ 原始（直接按测量值的比例）。测量值的定义见 docs/SIGNALS.md
let SIGRAW=false;try{SIGRAW=localStorage.getItem('neurolume.sig')==='raw'}catch(_){}
let gmax=1e-6, lastRaw=null;
function pushEnergy(arr,simulated){    // 一个 token 的逐层活动 → 归一化到 0..1
  if(!arr||!arr.length)return;
  if(arr.length!==NL){ // 模型实际层数和界面当前层数不一致（例如切换模型后）：按真实层数重建
    resetLayers(arr.length);const k=active&&KNOWN[active];if(k&&k.n===arr.length){PAT=k.pat;buildHolo()}
    loadWeightsFor(active||'')}
  lastRaw=arr;
  let mn=1e9,mx=-1e9;for(let l=0;l<NL;l++){mn=Math.min(mn,arr[l]);mx=Math.max(mx,arr[l])}const span=mx-mn+1e-9;
  gmax=Math.max(gmax*.995,mx);
  for(let l=0;l<NL;l++){const v=arr[l];
    if(simulated){act[l]=Math.max(act[l],clamp(v,0,1));continue}
    if(SIGRAW){act[l]=Math.max(act[l],clamp(v/gmax,0,1));continue}           // 原始：与最近的最大值成正比
    lo[l]=Math.min(lo[l]*.999+v*.001,v);hi[l]=Math.max(hi[l]*.999+v*.001,v);   // 每层自己的运行范围
    const own=(v-lo[l])/(hi[l]-lo[l]+1e-6),rel=(v-mn)/span;                    // 相对自己 / 相对这一步里别的层
    const e=Math.pow(clamp(.45*own+.55*rel,0,1),2);                           // 平方：只让真正突出的层亮起来
    act[l]=Math.max(act[l],e)}
  kick=Math.min(1,kick+.35)   // 每个 token：脑电信号短暂变密
}
function resetLayers(n){NL=n;act=new Float32Array(NL);lo=new Float32Array(NL).fill(1e9);hi=new Float32Array(NL).fill(-1e9);
  dW=[...Array(NL)].map(()=>MATS.map(()=>new Float32Array(144)));dMax=1e-6;deltaReal=false;synth();$('nlayers').textContent=NL;buildNet();buildHolo()}
function setStructure(key,n){ // 切换模型时让立柱的层数与真实层数一致
  const K=KNOWN[key]||{n:40,pat:'A'.repeat(40),label:'演示后端 · 40 层'};const nn=n||K.n;
  PAT=nn===K.n?K.pat:'A'.repeat(nn);$('scanModel').textContent=nn===K.n?K.label:K.label.split(' · ')[0]+' · '+nn+' 层';
  resetLayers(nn);loadWeightsFor(key)}

// ======== 权重（真实采样优先） ========
let W=null,realWeights=false,dW=[],dMax=1e-6;
function synth(){const r=rng(7),gauss=()=>Math.sqrt(-2*Math.log(r()+1e-9))*Math.cos(6.2832*r());
  W=[];for(let l=0;l<NL;l++)W.push(MATS.map((m,k)=>{const sd=.012+.01*Math.sin(l/6+k)**2,v=new Float32Array(144);for(let i=0;i<144;i++)v[i]=gauss()*sd;return{v,std:sd,shape:m.shape}}));realWeights=false}
function loadWeights(j){W=j.layers.map(L=>L.m.map(m=>({v:Float32Array.from(m.v),std:m.std,shape:m.shape})));realWeights=true}
function noteData(){$('dataNote').innerHTML=realWeights?'权重值：<b>真实权重</b>':'权重值：<b style="color:var(--warn)">示例（运行 dump_weights.py 换成真实值）</b>'}
function trainStep(){ // 示例：等训练接入后换成适配器的真实 ΔW
  const r=Math.random;for(let l=0;l<NL;l++)for(let k=0;k<4;k++){const prof=(.3+.7*l/NL)*(k>=2?1:.6);
    for(let i=0;i<144;i++){dW[l][k][i]+=(r()-.45)*W[l][k].std*.08*prof*(r()<.35?1:0);dMax=Math.max(dMax,Math.abs(dW[l][k][i]))}}
  mode='train';setMode()
}

