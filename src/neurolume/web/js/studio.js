// ======== 工作室：回答时间轴与逐层预测（logit lens）、效果对比、导出、训练数据生成、训练结果回看、图解说明 ========
let CUR_ADAPTER=null;
const escH=t=>String(t==null?'':t).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const showTok=t=>{const s=String(t);if(s&&!s.trim()&&!/[\n\t]/.test(s))return '␠'.repeat(Math.min(3,s.length));return JSON.stringify(s).slice(1,-1).replace(/\\"/g,'"')||'∅'};   // 让空格、换行看得见

// ---- 信号显示方式

// ---- 回答时间轴：整段回答的逐字 × 逐层活动热力图，可以拖动回看任意一个字；逐层预测表跟着它走
// 数据来源：在对话里选中一段 AI 回答 → 只看选中的部分；没有选中 → 最近一次回答的全部内容
let LATEST=null,SRC=null,CUR=-1,PIN=null,tlTimer=null,tlRaf=0,selDriven=false;
let LENS={from:0,to:null,step:1};try{Object.assign(LENS,JSON.parse(localStorage.getItem('neurolume.lensRange')||'{}'))}catch(_){}
const saveLens=()=>{try{localStorage.setItem('neurolume.lensRange',JSON.stringify(LENS))}catch(_){}};
const ftok=f=>f.tok!=null?f.tok:f.t;                       // 模型真正输出的这个字（单个 token）
// 一个汉字常被拆成 2~3 个字节级 token，单独解码会是乱码“�”：显示为 ◇，并在提示里说明
const PART='\uFFFD',isPart=t=>String(t).indexOf(PART)>=0;
const dispTok=t=>isPart(t)?'◇':showTok(t);
const dispOut=f=>{const o=ftok(f);return isPart(o)?(f.t&&!isPart(f.t)?showTok(f.t):'◇'):showTok(o)};
function srcIdx(){if(!SRC)return[];const a=SRC.a||0,b=SRC.b==null||SRC.b<0?SRC.frames.length-1:SRC.b;const o=[];for(let i=a;i<=b;i++)o.push(i);return o}
// 某个字在第几层“定下来”：从这一层起，后面每一层的预测都等于最终输出
function lockLayer(f){if(!f.lens||!f.lens.length)return null;const rows=[...f.lens].sort((x,y)=>x[0]-y[0]),out=ftok(f);let L=null;
  for(let i=rows.length-1;i>=0;i--){if(rows[i][1]===out)L=rows[i][0];else break}return L}
function normMatrix(idx){ // 每个字的逐层活动 → 0..1（和层图的亮度处理一致）
  const F=SRC.frames,rows=idx.map(i=>F[i].e).filter(Boolean);if(!rows.length)return null;const n=rows[0].length;
  let gmx=1e-9;const mn=new Float32Array(n).fill(1e9),mx=new Float32Array(n).fill(-1e9);
  for(const r of rows)for(let l=0;l<n;l++){gmx=Math.max(gmx,r[l]);mn[l]=Math.min(mn[l],r[l]);mx[l]=Math.max(mx[l],r[l])}
  return idx.map(i=>{const r=F[i].e;if(!r)return null;
    let a=1e9,b=-1e9;for(const v of r){a=Math.min(a,v);b=Math.max(b,v)}
    return r.map((v,l)=>Math.pow(clamp(.45*(v-mn[l])/(mx[l]-mn[l]+1e-9)+.55*(v-a)/(b-a+1e-9),0,1),2))})}
function tlShow(){const has=SRC&&SRC.frames.length&&(mode==='infer'||mode==='lens');$('tl').hidden=!has;$('lensCtl').hidden=mode!=='lens';return has}
function lensLayerCount(){const f=SRC&&SRC.frames.find(x=>x.lens&&x.lens.length);return f?Math.max(...f.lens.map(r=>r[0]))+1:0}
function syncLensCtl(n){const fs=$('lensFrom'),ts=$('lensTo');if(fs.options.length!==n){const o=[...Array(n)].map((_,i)=>`<option value="${i}">L${i}</option>`).join('');fs.innerHTML=o;ts.innerHTML=o}
  const f=clamp(LENS.from||0,0,n-1),t=LENS.to==null?n-1:clamp(LENS.to,f,n-1);fs.value=f;ts.value=t;
  document.querySelectorAll('#lensPre button').forEach(b=>b.setAttribute('aria-pressed',String((b.dataset.p==='all'&&f===0&&t===n-1)||(b.dataset.p==='half'&&f===Math.floor(n/2)&&t===n-1)||(b.dataset.p==='last16'&&f===Math.max(0,n-16)&&t===n-1))));return[f,t]}
function tlRender(){cancelAnimationFrame(tlRaf);tlRaf=requestAnimationFrame(()=>{if(tlShow())drawTimeline();if(mode==='lens')drawLens()})}
// ---- 回答回放：每个字一根柱子 + 原文着色（替代原来 40 行的热力图，那张图字一多就只剩噪点）
// 推理视图：柱高 = 模型在这个字上有多犹豫（1 − 它给这个字的概率），颜色 = 把握；悬停能看到它当时还在考虑哪些字。
//           实测：逐层活动量在不同字之间只差 ±10%，画出来几乎一样高，所以不再用它画柱子（老版本服务没有概率时才退回用它）
// 逐层预测视图：柱高 = 第几层才定下来（越高 = 越晚才想好），颜色 = 最后一层的把握（青 = 很有把握，黄 = 一般，橙红 = 没把握）
const BANDC=[[76,195,217],[132,206,98],[242,152,58]],BANDN=new Proxy([],{get:(_,i)=>(LANG==='en'?['shallow layers','middle layers','deep layers']:['浅层','中层','深层'])[i]});
const pct=(a,q)=>{const s=a.filter(x=>x!=null&&isFinite(x)).sort((x,y)=>x-y);return s.length?s[Math.min(s.length-1,Math.max(0,Math.round(q*(s.length-1))))]:0};
const confC=p=>{const A=[232,96,70],B=[232,194,106],C=[76,195,217];return p<.55?mix3(A,B,clamp((p-.2)/.35,0,1)):mix3(B,C,clamp((p-.55)/.35,0,1))};
const mix3=(a,b,t)=>a.map((v,i)=>v+(b[i]-v)*t);
const rgb=(c,a)=>a==null?`rgb(${c[0]|0},${c[1]|0},${c[2]|0})`:`rgba(${c[0]|0},${c[1]|0},${c[2]|0},${a})`;
function lensView(idx){return mode==='lens'&&idx.some(i=>SRC.frames[i].lens)}
function tlMetrics(idx){
  const F=SRC.frames;
  if(lensView(idx)){const L=Math.max(2,lensLayerCount());
    // 大多数字都在后三分之一的层才定下来：纵轴只画实际出现的范围（左侧标出层号），差别才看得出来
    const lks=idx.map(i=>SRC.frames[i].lens?lockLayer(SRC.frames[i]):null),lo=Math.max(0,Math.min(L-4,Math.floor(pct(lks.map(x=>x==null?L-1:x),.03))-2));
    return {kind:'lens',L,lo,rows:idx.map((i,j)=>{const f=F[i];if(!f.lens||!f.lens.length)return null;const lk=lks[j];
      const top=f.lens.reduce((a,r)=>r[0]>a[0]?r:a),p=top[2],v=lk==null?1:Math.max(.04,(lk-lo)/(L-1-lo));
      return {v,c:confC(p),lk,p}})}}
  if(idx.some(i=>F[i].p!=null))return {kind:'conf',rows:idx.map(i=>{const p=F[i].p;return p==null?null:{v:Math.max(.03,Math.sqrt(1-p)),c:confC(p),p,alts:F[i].alts}})};
  const E=idx.map(i=>F[i].e);const f0=E.find(Boolean);if(!f0)return null;const n=f0.length;
  const b1=Math.round(n/3),b2=Math.round(2*n/3),band=l=>l<b1?0:l<b2?1:2;
  const med=[...Array(n)].map((_,l)=>Math.max(1e-9,pct(E.map(r=>r&&r[l]),.5)));
  const raw=E.map(r=>{if(!r)return null;const s=[0,0,0],k=[0,0,0];
    for(let l=0;l<n;l++){const b=band(l);s[b]+=r[l]/med[l];k[b]++}
    const bm=s.map((v,i)=>v/Math.max(1,k[i])),eff=(s[0]+s[1]+s[2])/n,dom=bm.indexOf(Math.max(...bm));return {eff,bm,dom}});
  const effs=raw.map(r=>r&&r.eff),lo=pct(effs,.05),hi=Math.max(lo+1e-9,pct(effs,.97));
  return {kind:'infer',base:Math.sqrt(clamp((1-lo)/(hi-lo),0,1)),bands:[[0,b1-1],[b1,b2-1],[b2,n-1]],
    rows:raw.map(r=>r&&{v:Math.max(.04,Math.sqrt(clamp((r.eff-lo)/(hi-lo),0,1))),c:BANDC[r.dom],dom:r.dom,eff:r.eff})}}
let TLM=null;
// 备选字：模型生成这个字时概率最高的几个候选（不含它最终选的那个）
const altTxt=(r,f)=>{const o=ftok(f),a=(r.alts||[]).filter(x=>x[0]!==o&&x[1]>=.01).slice(0,3),en=LANG==='en';
  return a.length?(en?' · also considered: ':' · 还考虑过：')+a.map(x=>(en?`“${dispTok(x[0])}” `:`「${dispTok(x[0])}」`)+Math.round(x[1]*100)+'%').join(en?', ':'、'):''};
const tipTxt=(r,f)=>(LANG==='en'?'confidence ':'把握 ')+Math.round(r.p*100)+'%'+altTxt(r,f);   // 原文区域不经过翻译器，直接按语言生成
function drawTimeline(){
  const c=$('tlc'),d=Math.min(devicePixelRatio||1,2),w=c.clientWidth,idx=srcIdx(),n=idx.length,h=84;
  c.width=w*d;c.height=h*d;c.style.height=h+'px';const g=c.getContext('2d');g.setTransform(d,0,0,d,0,0);g.clearRect(0,0,w,h);
  const M=n?tlMetrics(idx):null;TLM={M,idx};
  const top=6,bot=h-4,H=bot-top,cw=n?w/n:0,gap=cw>=5?1:0;
  g.strokeStyle=isDark?'rgba(255,255,255,.07)':'rgba(0,0,0,.07)';g.lineWidth=1;
  for(const q of [.25,.5,.75]){const y=Math.round(bot-H*q)+.5;g.beginPath();g.moveTo(0,y);g.lineTo(w,y);g.stroke()}
  if(M){const R=M.rows;
    for(let j=0;j<n;j++){const r=R[j];if(!r)continue;const bh=Math.max(2,r.v*H),x=j*cw;
      g.fillStyle=rgb(r.c,j===(CUR>=0?idx.indexOf(CUR):-2)?1:.82);g.fillRect(x,bot-bh,Math.max(1,cw-gap),bh)}
    // 字很多时加一条平滑趋势线，看清整体起伏
    if(n>40){const k=Math.max(2,Math.round(n/40)),pts=[];for(let j=0;j<n;j++){let s=0,m=0;for(let t=Math.max(0,j-k);t<=Math.min(n-1,j+k);t++)if(R[t]){s+=R[t].v;m++}pts.push(m?s/m:0)}
      g.strokeStyle=isDark?'rgba(255,255,255,.85)':'rgba(20,24,34,.8)';g.lineWidth=1.5;g.beginPath();pts.forEach((v,j)=>{const x=(j+.5)*cw,y=bot-v*H;j?g.lineTo(x,y):g.moveTo(x,y)});g.stroke()}
    if(M.kind==='lens'){g.font='10px '+getComputedStyle(document.body).getPropertyValue('--mono');g.fillStyle=isDark?'rgba(255,255,255,.75)':'rgba(0,0,0,.65)';g.textBaseline='top';g.fillText('L'+(M.L-1),3,1);g.textBaseline='bottom';g.fillText('L'+M.lo,3,h-1)}
    if(M.kind==='infer'&&M.base!=null){const y=Math.round(bot-M.base*H)+.5;g.setLineDash([4,4]);g.strokeStyle=isDark?'rgba(255,255,255,.35)':'rgba(0,0,0,.35)';g.beginPath();g.moveTo(0,y);g.lineTo(w,y);g.stroke();g.setLineDash([])}}
  const k=CUR>=0?idx.indexOf(CUR):-1;
  if(k>=0){g.strokeStyle=isDark?'#fff':'#000';g.lineWidth=1.5;const x=Math.round((k+.5)*cw)+.5;g.beginPath();g.moveTo(x,0);g.lineTo(x,h);g.stroke()}
  $('tlr').max=Math.max(0,n-1);$('tlr').value=k>=0?k:Math.max(0,n-1);
  const f=SRC.frames[k>=0?CUR:idx[n-1]],lk=f&&lockLayer(f),r=k>=0&&M?M.rows[k]:null;
  let info='';if(k>=0){info=` · 第 ${k+1} 个：「${dispOut(f)}」`;
    if(r&&M.kind==='lens')info+=lk!=null?` · 第 ${lk} 层定下来 · 把握 ${Math.round(r.p*100)}%`:` · 到最后一层都没定下来 · 把握 ${Math.round(r.p*100)}%`;
    else if(r&&M.kind==='conf')info+=` · 把握 ${Math.round(r.p*100)}%`+altTxt(r,f);
    else if(r)info+=` · 忙碌程度 ${r.eff.toFixed(2)} 倍 · 最忙：${BANDN[r.dom]}`}
  $('tlLab').textContent=(selDriven?'选中的内容':(busy?'正在生成':'最近一次回答'))+` · ${n} 个字`+(k>=0?info:' · 点柱子或下面的字回看');
  const sw=p=>`<i class="sw" style="background:${rgb(confC(p))}"></i>`;
  $('tlKey').innerHTML=!M?'':M.kind==='conf'
    ?`<span>柱子越高 = 模型在这个字上越犹豫（下面原文里颜色越深也一样）</span><span>${sw(.95)}很有把握</span><span>${sw(.6)}一般</span><span>${sw(.25)}没把握</span><span>悬停看它当时还考虑过哪些字</span>`
    :M.kind==='lens'
    ?`<span>柱子越高 = 越靠后的层才想好这个字（纵轴 L${M.lo}–L${M.L-1}）</span><span><i class="sw" style="background:${rgb(confC(.95))}"></i>很有把握</span><span><i class="sw" style="background:${rgb(confC(.6))}"></i>一般</span><span><i class="sw" style="background:${rgb(confC(.25))}"></i>没把握</span>`
    :`<span>柱子越高 = 这个字让模型越费劲${M.base!=null?'（虚线 = 这段回答的平常水平）':''}</span>${BANDC.map((col,i)=>`<span><i class="sw" style="background:${rgb(col)}"></i>${BANDN[i]} L${M.bands[i][0]}–${M.bands[i][1]}${['（字形、语法）','（意思、联想）','（决定下一个字）'][i]}</span>`).join('')}`;
  drawTlText(idx,M,k)}
// 原文着色：和柱子同一套颜色，深浅 = 柱高；点任意一个字回看
function drawTlText(idx,M,k){const box=$('tlt'),F=SRC.frames;if(!M){box.innerHTML='';return}
  let html='';for(let j=0;j<idx.length;j++){const f=F[idx[j]],t=f.t;if(!t)continue;const r=M.rows[j];
    const a=M.kind==='conf'?(r?Math.min(.75,r.v*1.1):0):(r?(isDark?.12:.1)+r.v*(isDark?.62:.5):0),bg=r&&a>.03?rgb(r.c,a):'transparent',low=M.kind!=='infer'&&r&&r.p<.5;
    html+=`<span data-k="${j}" class="${j===k?'on':''}${low?' low':''}" style="background:${bg}"${r&&r.p!=null?` title="${escH(tipTxt(r,f))}"`:''}>${escH(t)}</span>`}
  box.innerHTML=html;const on=box.querySelector('.on');
  if(on){const bt=on.offsetTop-box.offsetTop;if(bt<box.scrollTop||bt>box.scrollTop+box.clientHeight-20)box.scrollTop=bt-box.clientHeight/2}
  else if(busy&&!selDriven)box.scrollTop=box.scrollHeight}
function tlPick(i){if(!SRC||!SRC.frames[i])return;CUR=i;const f=SRC.frames[i];
  // 定格的层图：这个字在每一层的活动，既和“这一层在这段回答里的平常水平”比，也和“这个字在其他层”比（与 normMatrix 一致）
  if(f.e){act.fill(0);pushEnergy(f.e,false);const idx=srcIdx(),M=normMatrix(idx),k=idx.indexOf(i);PIN=M&&k>=0?M[k]:normMatrix([i])[0];if(PIN)for(let l=0;l<PIN.length&&l<act.length;l++)act[l]=PIN[l]}tlRender()}
function tlLive(frames){LATEST={frames};if(!selDriven){SRC={frames,a:0,b:-1};CUR=-1;PIN=null}tlRender()}
function tlFinish(frames,bubbles){
  LATEST={frames};for(const b of bubbles){const fr=frames.map((f,i)=>[f,i]).filter(([f])=>f.el===b);let pos=0;const text=b.textContent;
    for(const [f] of fr){const ix=f.t?text.indexOf(f.t,pos):-1;if(ix>=0){f.o0=ix;f.o1=ix+f.t.length;pos=f.o1}else{f.o0=f.o1=pos}}
    b._msg={frames,idx:fr.map(x=>x[1])}}
  if(!selDriven){SRC={frames,a:0,b:-1};CUR=-1}tlRender()}
// 在对话里选中一段 AI 回答 → 时间轴和逐层预测只看这一段
function selRange(){const s=getSelection();if(!s||s.isCollapsed||!s.rangeCount)return null;const r=s.getRangeAt(0);
  const n=r.commonAncestorContainer,b=(n.nodeType===1?n:n.parentElement).closest('.m.a');if(!b||!b._msg)return null;
  const pre=document.createRange();pre.selectNodeContents(b);pre.setEnd(r.startContainer,r.startOffset);const s0=pre.toString().length,s1=s0+r.toString().length;
  const hit=b._msg.idx.filter(i=>{const f=b._msg.frames[i];return f.o1>s0&&f.o0<s1});return hit.length?{frames:b._msg.frames,a:hit[0],b:hit[hit.length-1]}:null}
let _selT=0;document.addEventListener('selectionchange',()=>{clearTimeout(_selT);_selT=setTimeout(()=>{const r=selRange();
  if(r){selDriven=true;SRC=r;CUR=-1;PIN=null;tlRender()}else if(selDriven){selDriven=false;if(LATEST)SRC={frames:LATEST.frames,a:0,b:-1};CUR=-1;PIN=null;tlRender()}},150)});
// 交互：点击/拖动热力图、滑块、回放
(()=>{const c=$('tlc');let down=false;const at=e=>{const idx=srcIdx();if(!idx.length)return;const r=c.getBoundingClientRect();tlPick(idx[clamp(Math.floor((e.clientX-r.left)/r.width*idx.length),0,idx.length-1)])};
  c.addEventListener('pointerdown',e=>{down=true;c.setPointerCapture(e.pointerId);at(e)});c.addEventListener('pointermove',e=>{if(down)at(e)});c.addEventListener('pointerup',()=>down=false);
  $('tlt').addEventListener('click',e=>{const sp=e.target.closest('span[data-k]');if(!sp)return;clearInterval(tlTimer);tlTimer=null;$('tlPlay').textContent='▶ 回放';tlPick(srcIdx()[+sp.dataset.k])});
  const tip=$('tip');c.addEventListener('pointerleave',()=>tip.hidden=true);
  c.addEventListener('mousemove',e=>{if(!TLM||!TLM.M||!TLM.idx.length){tip.hidden=true;return}const r=c.getBoundingClientRect(),j=clamp(Math.floor((e.clientX-r.left)/r.width*TLM.idx.length),0,TLM.idx.length-1),f=SRC.frames[TLM.idx[j]],m=TLM.M.rows[j];if(!f||!m){tip.hidden=true;return}
    tip.innerHTML=`<b>第 ${j+1} 个字：「${escH(dispOut(f))}」</b><br>`+(TLM.M.kind==='conf'?escH(tipTxt(m,f)).replace(' · ','<br>'):TLM.M.kind==='lens'?(m.lk!=null?`第 ${m.lk} 层定下来`:'到最后一层都没定下来')+` · 把握 ${Math.round(m.p*100)}%`:`忙碌程度 ${m.eff.toFixed(2)} 倍 · 最忙：${BANDN[m.dom]}`);
    tip.hidden=false;tip.style.left=Math.min(innerWidth-320,e.clientX+14)+'px';tip.style.top=(e.clientY+16)+'px'});
  $('tlr').oninput=()=>{clearInterval(tlTimer);tlTimer=null;$('tlPlay').textContent='▶ 回放';const idx=srcIdx();tlPick(idx[+$('tlr').value])};
  $('tlPlay').onclick=()=>{if(tlTimer){clearInterval(tlTimer);tlTimer=null;$('tlPlay').textContent='▶ 回放';return}const idx=srcIdx();if(!idx.length)return;
    let k=CUR>=0?idx.indexOf(CUR):idx.length-1;if(k>=idx.length-1)k=-1;$('tlPlay').textContent='❚❚ 暂停';
    tlTimer=setInterval(()=>{k++;if(k>=idx.length){clearInterval(tlTimer);tlTimer=null;$('tlPlay').textContent='▶ 回放';return}tlPick(idx[k])},110)};
  $('tlLatest').onclick=()=>{getSelection().removeAllRanges();selDriven=false;if(LATEST)SRC={frames:LATEST.frames,a:0,b:-1};CUR=-1;PIN=null;tlRender()};
  const setL=(f,t)=>{LENS.from=f;LENS.to=t;saveLens();tlRender()};
  $('lensPre').onclick=e=>{const b=e.target.closest('button');if(!b)return;const n=lensLayerCount();if(!n)return;
    if(b.dataset.p==='all')setL(0,n-1);else if(b.dataset.p==='half')setL(Math.floor(n/2),n-1);else setL(Math.max(0,n-16),n-1)};
  $('lensFrom').onchange=()=>{LENS.from=+$('lensFrom').value;if(LENS.to!=null&&LENS.to<LENS.from)LENS.to=LENS.from;saveLens();tlRender()};
  $('lensTo').onchange=()=>{LENS.to=+$('lensTo').value;if(LENS.to<LENS.from)LENS.from=LENS.to;saveLens();tlRender()};
  $('lensStep').value=String(LENS.step||1);$('lensStep').onchange=()=>{LENS.step=+$('lensStep').value;saveLens();tlRender()};
  addEventListener('resize',tlRender)})();

const BF={};   // gid → {busy} | {done} | {err}
async function backfillLens(gid,frames){BF[gid]={busy:true};
  try{const r=await api('/api/lens/backfill',{gid});const fr=frames.filter(f=>f.gid===gid),T=r.tokens||[],Lz=r.lens||[];
    let j=0;for(const f of fr){const k=T.indexOf(ftok(f),j);if(k<0)continue;f.lens=Lz[k];j=k+1}   // 按 token 顺序对齐
    BF[gid]={done:true}}
  catch(e){BF[gid]={err:'无法补算逐层预测：'+(e.message||e)}}
  tlRender()}
// ---- 逐层预测表：选中的那个字前后若干列；每一行是一层（全部层或按密度抽取），最后一行是模型真正的输出
function drawLens(){
  const v=$('lensv'),idx=SRC?srcIdx().filter(i=>SRC.frames[i].lens):[];
  if(!idx.length){
    // 生成时没开这个视图：请服务器对这段回答补算一次（不拖慢生成；再跑一遍前向，读出每一层的预测）
    const gids=SRC?[...new Set(srcIdx().map(i=>SRC.frames[i].gid).filter(g=>g!=null))]:[];
    if(LIVE&&!SIM&&gids.length){const st=gids.map(g=>BF[g]),err=st.find(x=>x&&x.err);
      if(err){v.innerHTML=`<div class="empty">${escH(err.err)}</div>`;return}
      if(!busy)gids.forEach(g=>{if(!BF[g])backfillLens(g,SRC.frames)});
      const n=SRC.frames.filter(f=>gids.includes(f.gid)).length;
      v.innerHTML=`<div class="empty">${busy?'回答生成完以后，会在这里补算逐层预测。':`正在补算这段回答的逐层预测（${n} 个字）：把回答再过一遍模型，读出每一层的猜测，不影响生成速度…`+(n>250?'回答越长越久，1000 字大约要 15 秒。':'')}</div>`;return}
    v.innerHTML=`<div class="empty">${!LIVE||SIM?'逐层预测需要连接真实模型。':'打开这个视图后再提问，就能看到每一层在“猜”下一个字。'}</div>`;return}
  const W=Math.max(4,Math.floor((v.clientWidth-56)/64));let k=CUR>=0?idx.indexOf(CUR):idx.length-1;if(k<0)k=idx.length-1;
  const st=clamp(k-Math.floor(W*.6),0,Math.max(0,idx.length-W)),cols=idx.slice(st,st+W),F=SRC.frames;
  const all=[...new Set(cols.flatMap(i=>F[i].lens.map(r=>r[0])))].sort((a,b)=>a-b),last=all[all.length-1];
  const [lf,lt]=syncLensCtl(last+1),stp=LENS.step||1;
  const layers=all.filter(l=>l>=lf&&l<=lt&&((l-lf)%stp===0||l===lt));
  const locks=cols.map(i=>lockLayer(F[i]));
  const cell=(i,l,c)=>{const r=F[i].lens.find(x=>x[0]===l);if(!r)return'<td></td>';const p=r[2],a=Math.round(12+70*p),lk=locks[c]!=null&&l>=locks[c];
    return `<td class="${lk?'lock':''}${l===locks[c]?' lock1':''}${i===CUR?' cur':''}" data-i="${i}" title="L${l} · p=${p}${lk?' · 已等于最终输出':''}${isPart(r[1])?' · ◇ = 一个汉字或符号的一部分（多字节编码的片段，单独无法显示）':''}" style="background:color-mix(in srgb, var(--accent) ${a}%, transparent)">${escH(dispTok(r[1]))}</td>`};
  v.innerHTML=`<table><tbody>${layers.map(l=>`<tr><th>L${l}</th>${cols.map((i,c)=>cell(i,l,c)).join('')}</tr>`).join('')}
    <tr class="out"><th>输出</th>${cols.map(i=>`<td data-i="${i}" class="${i===CUR?'cur':''}"${isPart(ftok(F[i]))?' title="◇ = 一个汉字或符号的一部分（多字节编码的片段），合起来才是完整的字"':''}>${escH(dispOut(F[i]))}</td>`).join('')}</tr>
    <tr class="lk"><th>定下来</th>${locks.map((L,c)=>`<td data-i="${cols[c]}">${L==null?'—':'L'+L}</td>`).join('')}</tr></tbody></table>`;
  const lkh=(v.querySelector('tr.lk')||{}).offsetHeight||0;v.querySelectorAll('tr.out td,tr.out th').forEach(e=>e.style.bottom=lkh+'px');   // “输出”行紧贴在“定下来”行上方
  v.querySelectorAll('td[data-i]').forEach(td=>td.onclick=()=>tlPick(+td.dataset.i))}

// ---- 训练结果回看：“训练”视图里随时查看任意一个适配器真实改动了哪些层
let DELTA_ID=null;
async function reviewDelta(force){
  if(!LIVE||trainRunning)return;let list=[];try{list=(await api('/api/train/adapters')).adapters.filter(a=>!active||a.model===active)}catch(_){}
  const bar=$('trainbars');bar.hidden=mode!=='train';
  if(!list.length){$('trainbarsLab').textContent='还没有训练过的适配器。点“去训练…”用你的数据训练一次，完成后这里显示真实的各层改变幅度，并且可以随时回来查看。';$('trainbarsBars').innerHTML='';return}
  let id=force||(CUR_ADAPTER&&list.some(a=>a.id===CUR_ADAPTER)?CUR_ADAPTER:null)||DELTA_ID||list[0].id;if(!list.some(a=>a.id===id))id=list[0].id;
  if(id!==DELTA_ID){await pullDelta(id);DELTA_ID=id}
  $('trainbarsLab').innerHTML=`查看适配器 <select id="tbSel" aria-label="选择要查看的适配器">${list.map(a=>`<option value="${escH(a.id)}" ${a.id===id?'selected':''}>${escH(a.created)}${a.dataset?' · '+escH(a.dataset):''}${a.val_first!=null&&a.val_loss!=null?' · 验证损失 '+a.val_first.toFixed(2)+' → '+a.val_loss.toFixed(2):''}</option>`).join('')}</select> · 每根柱子是一层被训练改变的总幅度（ΔW 大小），悬停层图看具体位置和数值`;
  $('tbSel').onchange=()=>reviewDelta($('tbSel').value);drawLayerBars($('trainbarsBars'))}

// ---- 各层改变幅度（训练完成面板）
function drawLayerBars(el){if(!el)return;const g=glow&&glow.length?glow:new Float32Array(NL);
  el.innerHTML=[...g].map((v,i)=>`<i style="height:${Math.max(1,Math.round(v*100))}%" title="L${i}: ${v.toFixed(2)}"></i>`).join('')}

// ---- 效果对比：同一个问题，原模型 vs 训练后
async function openCompare(){
  openOv('效果对比');if(!LIVE){$('mbody').innerHTML=needLive();return}
  if(!CUR_ADAPTER){$('mbody').innerHTML='<p class="hint">先在上方的适配器下拉框里选一个训练结果，再来对比。</p>';return}
  let samples=[];try{samples=(await api('/api/compare/samples?adapter='+encodeURIComponent(CUR_ADAPTER))).samples}catch(_){}
  $('mbody').innerHTML=`<p class="hint">同一个问题分别让原模型和训练后的模型回答（不重新加载模型，适配器临时关掉再打开）。用没参与训练的验证题最能看出真实效果。</p>
   ${samples.length?`<div class="lab hint">验证集里的问题（点一下填入）</div><div style="display:flex;flex-direction:column;gap:6px">${samples.map((x,i)=>`<button type="button" data-s="${i}" style="text-align:left">${escH(x.prompt.slice(0,120))}</button>`).join('')}</div>`:''}
   <textarea id="cmpQ" rows="3" placeholder="输入一个问题" aria-label="对比用的问题" style="height:auto"></textarea>
   <div class="row"><button class="btn1" id="cmpGo">对比</button><button id="cmpExp">导出这个模型…</button></div><div class="err" id="cmpErr"></div><div class="cmp" id="cmpOut"></div>`;
  let exp='';
  document.querySelectorAll('[data-s]').forEach(b=>b.onclick=()=>{const x=samples[+b.dataset.s];$('cmpQ').value=x.prompt;exp=x.expected});
  $('cmpQ').oninput=()=>{exp=''};
  $('cmpExp').onclick=()=>openExport(CUR_ADAPTER);
  $('cmpGo').onclick=async()=>{const q=$('cmpQ').value.trim();if(!q)return;$('cmpErr').textContent='';$('cmpGo').disabled=true;
    $('cmpOut').innerHTML='<div class="hint">正在生成两次回答…</div>';busy=true;
    try{const r=await api('/api/compare',{prompt:q});
      $('cmpOut').innerHTML=(exp?`<div class="card"><b>数据里的参考答案</b>${escH(exp)}</div>`:'')+`<div class="card"><b>原模型</b>${escH(r.base)}</div><div class="card"><b>训练后</b>${escH(r.tuned)}</div>`}
    catch(e){$('cmpOut').innerHTML='';$('cmpErr').textContent=e.message}
    busy=false;$('cmpGo').disabled=false}}
$('btnCompare').onclick=openCompare;

// ---- 导出
let expTimer=null;
async function openExport(adapter){
  openOv('导出模型');if(!LIVE){$('mbody').innerHTML=needLive();return}
  const draw=async()=>{let d={job:{state:'idle'},exports:[]};try{d=await api('/api/export')}catch(_){}
    const j=d.job,run=j.state==='running';
    $('mbody').innerHTML=`<p class="hint">把训练好的适配器合并进模型，得到一个可以单独使用的新模型。导出时会先卸载当前模型，需要几分钟和较多磁盘空间。</p>
     ${adapter?`<div class="card"><div class="nm">适配器：${escH(adapter)}</div>
      <label class="hint" for="expName">名字（可选）</label><input type="text" id="expName" placeholder="例如 my-coder" aria-label="导出名字">
      <div class="row"><button class="btn1" data-k="mlx" ${run?'disabled':''}>导出为 MLX 模型（LM Studio 可直接用）</button><button data-k="ollama" ${run?'disabled':''}>导出到 Ollama</button></div>
      <div class="rp">导出到 Ollama 会先把权重还原成未量化的格式（体积约为原来的 4 倍），再由 Ollama 导入并压缩。新结构的模型 Ollama 不一定支持，失败时导出的文件夹会保留。</div></div>`:''}
     ${j.state!=='idle'?`<div class="card"><div class="nm">${run?'正在导出…':j.state==='done'?'导出完成':'导出失败'}</div><div class="rp">${escH(j.path||'')}</div>
       ${j.lmstudio?`<div class="rp">已放进 LM Studio 的模型目录：${escH(j.lmstudio)}</div>`:''}${j.ollama?`<div class="rp">Ollama 里可以直接运行：ollama run ${escH(j.ollama)}</div>`:''}
       ${j.state==='done'&&j.kind==='ollama'&&!j.ollama&&j.ollama_cmd?`<div class="rp">没有找到 Ollama。安装后在终端运行：${escH(j.ollama_cmd)}</div>`:''}${j.error?`<div class="err">${escH(j.error)}</div>`:''}</div>`:''}
     <div class="err" id="expErr"></div>
     ${d.exports.length?`<div class="lab hint">已导出的模型</div>`+d.exports.map(x=>`<div class="rp">${escH(x.name)} · ${escH(x.kind)} · ${escH(x.created)} · ${escH(x.path)}</div>`).join(''):''}`;
    document.querySelectorAll('[data-k]').forEach(b=>b.onclick=async()=>{$('expErr').textContent='';
      try{await api('/api/export',{kind:b.dataset.k,adapter,name:$('expName').value.trim()});applyState(await api('/api/state'));draw()}catch(e){$('expErr').textContent=e.message}});
    clearInterval(expTimer);expTimer=run?setInterval(()=>{if($('ov').hidden){clearInterval(expTimer);return}draw()},3000):null};
  draw()}

// ---- 用自己的资料生成训练数据
let genTimer=null;
async function renderGen(){
  const body=$('mbody');
  body.innerHTML=`<p class="hint">选几份资料，当前加载的模型会帮你生成训练数据。生成的每一条都要你检查：不对的取消勾选或直接改，然后保存成数据集。</p>
   <label class="card"><span class="row"><input type="radio" name="gm" value="qa" checked> <b>学会这些知识</b><span class="rp">从文档里生成“问题 + 答案”</span></span></label>
   <label class="card"><span class="row"><input type="radio" name="gm" value="style"> <b>学会我的风格</b><span class="rp">从你写的代码或文章里，反推“需求”，答案就是你自己写的原文</span></span></label>
   <input type="file" id="gfiles" multiple aria-label="选择资料">
   <div class="row"><button class="btn1" id="ggo">开始生成</button><button id="gback">返回</button></div><div class="err" id="gerr"></div><div id="gout"></div>`;
  $('gback').onclick=()=>{clearInterval(genTimer);renderWiz()};
  $('ggo').onclick=async()=>{const fs=[...$('gfiles').files];$('gerr').textContent='';if(!fs.length){$('gerr').textContent='请先选择文件';return}
    try{const files=await Promise.all(fs.map(async f=>({name:f.name,text:await f.text()})));
      await api('/api/datagen/start',{files,mode:document.querySelector('input[name=gm]:checked').value,lang:LANG});pollGen()}catch(e){$('gerr').textContent=e.message}};
  const st=await api('/api/datagen').catch(()=>({}));if(st.state==='running'||st.state==='done')pollGen()}
async function pollGen(){
  clearInterval(genTimer);
  const tick=async()=>{let st;try{st=await api('/api/datagen')}catch(_){return}
    if(!$('gout'))return clearInterval(genTimer);
    if(st.state==='running'){$('gout').innerHTML=`<div class="prog"><i style="width:${Math.round(st.done/st.total*100)||3}%"></i></div><div class="row"><span class="hint">已处理 ${st.done}/${st.total} 段，得到 ${st.items.length} 条</span><button id="gcancel">停止</button></div>`;
      $('gcancel').onclick=()=>api('/api/datagen/cancel',{});return}
    clearInterval(genTimer);showGen(st)};
  await tick();genTimer=setInterval(tick,2000)}
function showGen(st){
  const items=st.items||[];
  $('gout').innerHTML=`${st.error?`<div class="err">${escH(st.error)}</div>`:''}${st.truncated?'<div class="hint">资料较多，这次只处理了前 40 段。</div>':''}
   <div class="hint">共 ${items.length} 条。取消勾选不要的，可以直接修改文字。</div>
   <div style="display:flex;flex-direction:column;gap:8px;max-height:46vh;overflow:auto">${items.map((it,i)=>`<div class="card gen-item"><input type="checkbox" data-gi="${i}" checked aria-label="保留第 ${i+1} 条"><div><textarea data-gq="${i}" aria-label="问题">${escH(it.q)}</textarea><textarea data-ga="${i}" aria-label="答案">${escH(it.a)}</textarea></div></div>`).join('')}</div>
   <div class="row"><input type="text" id="gname" placeholder="数据集名字" aria-label="数据集名字" value="${escH((items[0]&&items[0].src||'generated').replace(/\.[^.]+$/,''))}"><button class="btn1" id="gsave" ${items.length?'':'disabled'}>保存为数据集并继续</button></div>`;
  $('gsave').onclick=async()=>{const keep=items.map((it,i)=>({q:document.querySelector(`[data-gq="${i}"]`).value.trim(),a:document.querySelector(`[data-ga="${i}"]`).value.trim(),on:document.querySelector(`[data-gi="${i}"]`).checked})).filter(x=>x.on&&x.q&&x.a);
    try{const meta=await api('/api/datagen/save',{name:$('gname').value.trim(),items:keep});wiz.meta=meta;wiz.step=2;renderWiz()}catch(e){$('gerr').textContent=e.message}}}

// ---- 这张图怎么看
const LEARN=[
 ['你在看什么','模型由很多“层”叠起来，每生成一个字，信息都会从第一层流到最后一层。层图里每一列是一层，每一行是层里的一个矩阵；脑图里每一圈（或星云的每一段高度）也对应一层。'],
 ['亮度测的是什么','每生成一个字，测量每一层对这个字的内部表示改了多少：改动的平均大小除以输入的平均大小。这是真实测量值，不代表“重要性”。为了看得出变化，亮度把层与层之间的差别放大了；鼠标停在层图上可以看到这一层的原始测量值。'],
 ['连线和格子为什么有亮有暗','在回答回放里点一个字，层图会定格在生成这个字的那一刻。每一层的亮度 = 这一层在这个字上把内部表示改了多少，同时和两样东西比较：这一层在这段回答里的平常水平，以及这个字在其他层的改动。相邻两层之间那一束线的亮度 = 两层亮度的几何平均，所以只有两层都很活跃时线才会粗、亮；只要有一层安静，这束线就是淡的。格子的颜色 = 这一层的亮度 × 这个位置权重的大小，所以同一层里图案是固定的，只是整体变亮或变暗。注意：一束线里 16 条细线具体连在哪两个矩阵之间、亮点怎么跑，是为了好看画的，不是测量值；真实测量只到“每一层”这一级。'],
 ['注意力（Q、输出）','让当前的字去“看”前面的字：Q 决定要找什么信息，输出矩阵把找到的信息写回来。全注意力层每一步都能看到前面所有字。'],
 ['线性注意力层','Qwen3.6 里四层中有三层是线性注意力（DeltaNet）：不再逐个回看前面的字，而是维护一份不断更新的“记忆”，长文本时更快。层号旁边的 ◆ 表示全注意力层。'],
 ['MLP（门控、下投影）','对每个字单独做的计算，很多“知识”存放在这里：门控决定打开哪些通道，下投影把结果写回去。'],
 ['训练时改了什么（LoRA）','微调不会改动原模型文件，而是给部分矩阵加上一个很小的修正 ΔW = B×A。“训练”视图显示 ΔW 在哪里、是变大还是变小；训练完成面板里的柱子是每层改动的总量。'],
 ['逐层预测（logit lens）','把某一层的中间结果直接送到模型的输出端，看看“如果在这一层停下”会输出什么字。前面的层还没有形成能直接“读出来”的表示，常出现乱码、标点或别的语言，这是正常现象；越往后越接近最终答案。表格最下面的“定下来”一行，是从第几层起每一层都猜中了最终输出：数字越小，这个字越早被决定。下方的时间轴可以拖动回看整段回答里任意一个字。'],
];
$('btnLearn').onclick=()=>{openOv('这张图怎么看？');$('mbody').innerHTML=`<div class="learn">${LEARN.map(([h,p])=>`<h4>${h}</h4><p>${p}</p>`).join('')}<p style="margin-top:12px">更详细的说明见项目里的 docs/SIGNALS.md。</p></div>`};
