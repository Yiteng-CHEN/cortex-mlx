// ======== 工作室：回答时间轴与逐层预测（logit lens）、效果对比、导出、训练数据生成、训练结果回看、图解说明 ========
let CUR_ADAPTER=null;
const escH=t=>String(t==null?'':t).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const showTok=t=>JSON.stringify(String(t)).slice(1,-1).replace(/\\"/g,'"')||'∅';   // 让空格、换行看得见

// ---- 信号显示方式
function setSig(raw){SIGRAW=raw;try{localStorage.setItem('neurolume.sig',raw?'raw':'enh')}catch(_){}$('btnSig').textContent=raw?'显示：原始':'显示：增强'}
setSig(SIGRAW);$('btnSig').onclick=()=>setSig(!SIGRAW);

// ---- 回答时间轴：整段回答的逐字 × 逐层活动热力图，可以拖动回看任意一个字；逐层预测表跟着它走
// 数据来源：在对话里选中一段 AI 回答 → 只看选中的部分；没有选中 → 最近一次回答的全部内容
let LATEST=null,SRC=null,CUR=-1,PIN=null,tlTimer=null,tlRaf=0,selDriven=false;
let LENS_DEN=1;try{LENS_DEN=+(localStorage.getItem('neurolume.lensDen')||1)||1}catch(_){}
const ftok=f=>f.tok!=null?f.tok:f.t;                       // 模型真正输出的这个字（单个 token）
function srcIdx(){if(!SRC)return[];const a=SRC.a||0,b=SRC.b==null||SRC.b<0?SRC.frames.length-1:SRC.b;const o=[];for(let i=a;i<=b;i++)o.push(i);return o}
// 某个字在第几层“定下来”：从这一层起，后面每一层的预测都等于最终输出
function lockLayer(f){if(!f.lens||!f.lens.length)return null;const rows=[...f.lens].sort((x,y)=>x[0]-y[0]),out=ftok(f);let L=null;
  for(let i=rows.length-1;i>=0;i--){if(rows[i][1]===out)L=rows[i][0];else break}return L}
function normMatrix(idx){ // 每个字的逐层活动 → 0..1（和上面的“增强 / 原始”显示一致）
  const F=SRC.frames,rows=idx.map(i=>F[i].e).filter(Boolean);if(!rows.length)return null;const n=rows[0].length;
  let gmx=1e-9;const mn=new Float32Array(n).fill(1e9),mx=new Float32Array(n).fill(-1e9);
  for(const r of rows)for(let l=0;l<n;l++){gmx=Math.max(gmx,r[l]);mn[l]=Math.min(mn[l],r[l]);mx[l]=Math.max(mx[l],r[l])}
  return idx.map(i=>{const r=F[i].e;if(!r)return null;if(SIGRAW)return r.map(v=>v/gmx);
    let a=1e9,b=-1e9;for(const v of r){a=Math.min(a,v);b=Math.max(b,v)}
    return r.map((v,l)=>Math.pow(clamp(.45*(v-mn[l])/(mx[l]-mn[l]+1e-9)+.55*(v-a)/(b-a+1e-9),0,1),2))})}
function tlShow(){const has=SRC&&SRC.frames.length&&(mode==='infer'||mode==='lens');$('tl').hidden=!has;return has}
function tlRender(){cancelAnimationFrame(tlRaf);tlRaf=requestAnimationFrame(()=>{if(tlShow())drawTimeline();if(mode==='lens')drawLens()})}
function drawTimeline(){
  const c=$('tlc'),d=Math.min(devicePixelRatio||1,2),w=c.clientWidth,idx=srcIdx(),n=idx.length,M=normMatrix(idx);
  const L=(SRC.frames.find(f=>f.e)||{e:[]}).e.length||NL,h=Math.max(70,Math.min(150,L*2.2));
  c.width=w*d;c.height=h*d;c.style.height=h+'px';const g=c.getContext('2d');g.setTransform(d,0,0,d,0,0);g.clearRect(0,0,w,h);
  if(!n)return;const cw=w/n,rh=h/L;
  if(M)for(let j=0;j<n;j++){const r=M[j];if(!r)continue;for(let l=0;l<L;l++){const col=heat(r[l]);g.fillStyle=`rgb(${col[0]|0},${col[1]|0},${col[2]|0})`;g.fillRect(j*cw,l*rh,Math.ceil(cw),Math.ceil(rh))}}
  const k=CUR>=0?idx.indexOf(CUR):-1;
  if(k>=0){g.strokeStyle=isDark?'#fff':'#000';g.lineWidth=2;g.strokeRect(k*cw+1,1,Math.max(2,cw-2),h-2)}
  $('tlr').max=n-1;$('tlr').value=k>=0?k:n-1;
  const f=SRC.frames[k>=0?CUR:idx[n-1]],lk=f&&lockLayer(f);
  $('tlLab').textContent=(selDriven?'选中的内容':(busy?'正在生成':'最近一次回答'))+` · ${n} 个字`+(k>=0?` · 第 ${k+1} 个：「${showTok(ftok(f))}」`+(lk!=null?` · 从第 ${lk} 层起就定下来了`:''):' · 拖动或点击热力图回看任意一个字');
}
function tlPick(i){if(!SRC||!SRC.frames[i])return;CUR=i;const f=SRC.frames[i];
  if(f.e){act.fill(0);pushEnergy(f.e,false);PIN=normMatrix([i])[0]}tlRender()}
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
  $('tlr').oninput=()=>{clearInterval(tlTimer);tlTimer=null;$('tlPlay').textContent='▶ 回放';const idx=srcIdx();tlPick(idx[+$('tlr').value])};
  $('tlPlay').onclick=()=>{if(tlTimer){clearInterval(tlTimer);tlTimer=null;$('tlPlay').textContent='▶ 回放';return}const idx=srcIdx();if(!idx.length)return;
    let k=CUR>=0?idx.indexOf(CUR):idx.length-1;if(k>=idx.length-1)k=-1;$('tlPlay').textContent='❚❚ 暂停';
    tlTimer=setInterval(()=>{k++;if(k>=idx.length){clearInterval(tlTimer);tlTimer=null;$('tlPlay').textContent='▶ 回放';return}tlPick(idx[k])},110)};
  $('tlLatest').onclick=()=>{getSelection().removeAllRanges();selDriven=false;if(LATEST)SRC={frames:LATEST.frames,a:0,b:-1};CUR=-1;PIN=null;tlRender()};
  $('lensDen').value=String(LENS_DEN);$('lensDen').onchange=()=>{LENS_DEN=+$('lensDen').value;try{localStorage.setItem('neurolume.lensDen',LENS_DEN)}catch(_){}tlRender()};
  addEventListener('resize',tlRender)})();

// ---- 逐层预测表：选中的那个字前后若干列；每一行是一层（全部层或按密度抽取），最后一行是模型真正的输出
function drawLens(){
  const v=$('lensv'),idx=SRC?srcIdx().filter(i=>SRC.frames[i].lens):[];
  if(!idx.length){v.innerHTML=`<div class="empty">${!LIVE||SIM?'逐层预测需要连接真实模型。':'打开这个视图后再提问，就能看到每一层在“猜”下一个字。'}</div>`;return}
  const W=Math.max(4,Math.floor((v.clientWidth-60)/86));let k=CUR>=0?idx.indexOf(CUR):idx.length-1;if(k<0)k=idx.length-1;
  const st=clamp(k-Math.floor(W*.6),0,Math.max(0,idx.length-W)),cols=idx.slice(st,st+W),F=SRC.frames;
  const all=[...new Set(cols.flatMap(i=>F[i].lens.map(r=>r[0])))].sort((a,b)=>a-b),last=all[all.length-1];
  const layers=all.filter((l,j)=>j%LENS_DEN===0||l===last);
  const locks=cols.map(i=>lockLayer(F[i]));
  const cell=(i,l,c)=>{const r=F[i].lens.find(x=>x[0]===l);if(!r)return'<td></td>';const p=r[2],a=Math.round(12+70*p),lk=locks[c]!=null&&l>=locks[c];
    return `<td class="${lk?'lock':''}${l===locks[c]?' lock1':''}${i===CUR?' cur':''}" data-i="${i}" title="L${l} · p=${p}${lk?' · 已等于最终输出':''}" style="background:color-mix(in srgb, var(--accent) ${a}%, transparent)">${escH(showTok(r[1]))}</td>`};
  v.innerHTML=`<table><tbody>${layers.map(l=>`<tr><th>L${l}</th>${cols.map((i,c)=>cell(i,l,c)).join('')}</tr>`).join('')}
    <tr class="out"><th>输出</th>${cols.map(i=>`<td data-i="${i}" class="${i===CUR?'cur':''}">${escH(showTok(ftok(F[i])))}</td>`).join('')}</tr>
    <tr class="lk"><th>定下来</th>${locks.map((L,c)=>`<td data-i="${cols[c]}">${L==null?'—':'L'+L}</td>`).join('')}</tr></tbody></table>`;
  v.querySelectorAll('td[data-i]').forEach(td=>td.onclick=()=>tlPick(+td.dataset.i))}

// ---- 训练结果回看：“训练”视图里随时查看任意一个适配器真实改动了哪些层
let DELTA_ID=null;
async function reviewDelta(force){
  if(!LIVE||trainRunning)return;let list=[];try{list=(await api('/api/train/adapters')).adapters.filter(a=>!active||a.model===active)}catch(_){}
  const bar=$('trainbars');bar.hidden=mode!=='train';
  if(!list.length){$('trainbarsLab').textContent='还没有训练过的适配器：层图里现在是示例 ΔW。训练完成后，这里会显示真实的各层改变幅度，并且可以随时回来查看。';$('trainbarsBars').innerHTML='';return}
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
 ['亮度测的是什么','每生成一个字，测量每一层对这个字的内部表示改了多少：改动的平均大小除以输入的平均大小。这是真实测量值，不代表“重要性”。“增强”显示会放大层与层之间的差别，“原始”显示直接按测量值的比例。'],
 ['注意力（Q、输出）','让当前的字去“看”前面的字：Q 决定要找什么信息，输出矩阵把找到的信息写回来。全注意力层每一步都能看到前面所有字。'],
 ['线性注意力层','Qwen3.6 里四层中有三层是线性注意力（DeltaNet）：不再逐个回看前面的字，而是维护一份不断更新的“记忆”，长文本时更快。层号旁边的 ◆ 表示全注意力层。'],
 ['MLP（门控、下投影）','对每个字单独做的计算，很多“知识”存放在这里：门控决定打开哪些通道，下投影把结果写回去。'],
 ['训练时改了什么（LoRA）','微调不会改动原模型文件，而是给部分矩阵加上一个很小的修正 ΔW = B×A。“训练”视图显示 ΔW 在哪里、是变大还是变小；训练完成面板里的柱子是每层改动的总量。'],
 ['逐层预测（logit lens）','把某一层的中间结果直接送到模型的输出端，看看“如果在这一层停下”会输出什么字。前面的层还没有形成能直接“读出来”的表示，常出现乱码、标点或别的语言，这是正常现象；越往后越接近最终答案。表格最下面的“定下来”一行，是从第几层起每一层都猜中了最终输出：数字越小，这个字越早被决定。下方的时间轴可以拖动回看整段回答里任意一个字。'],
];
$('btnLearn').onclick=()=>{openOv('这张图怎么看？');$('mbody').innerHTML=`<div class="learn">${LEARN.map(([h,p])=>`<h4>${h}</h4><p>${p}</p>`).join('')}<p style="margin-top:12px">更详细的说明见项目里的 docs/SIGNALS.md。</p></div>`};
