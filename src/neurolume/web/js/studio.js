// ======== 工作室：逐层预测（logit lens）、生成回放、效果对比、导出、训练数据生成、图解说明 ========
let CUR_ADAPTER=null;
const escH=t=>String(t==null?'':t).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const showTok=t=>JSON.stringify(String(t)).slice(1,-1).replace(/\\"/g,'"')||'∅';   // 让空格、换行看得见

// ---- 信号显示方式
function setSig(raw){SIGRAW=raw;try{localStorage.setItem('neurolume.sig',raw?'raw':'enh')}catch(_){}$('btnSig').textContent=raw?'显示：原始':'显示：增强'}
setSig(SIGRAW);$('btnSig').onclick=()=>setSig(!SIGRAW);

// ---- 逐层预测：最近 16 个字，每个字一列
let LENSF=[],lensSel=-1;
function pushLens(tok,rows){LENSF.push({tok,rows});if(LENSF.length>16)LENSF.shift();lensSel=-1;if(mode==='lens')drawLens()}
function drawLens(frames,sel){
  const F=frames||LENSF,s=sel==null?lensSel:sel,v=$('lensv');
  if(!F.length){v.innerHTML=`<div class="empty">${!LIVE||SIM?'逐层预测需要连接真实模型。':'打开这个视图后再提问，就能看到每一层在“猜”下一个字。（会让生成稍慢一些）'}</div>`;return}
  const layers=[...new Set(F.flatMap(f=>f.rows.map(r=>r[0])))].sort((a,b)=>a-b);
  const cell=(f,l)=>{const r=f.rows.find(x=>x[0]===l);if(!r)return'<td></td>';const p=r[2],a=Math.round(15+70*p);
    return `<td title="p=${p}" style="background:color-mix(in srgb, var(--accent) ${a}%, transparent)">${escH(showTok(r[1]))}</td>`};
  v.innerHTML=`<table><tbody>${layers.slice(0,-1).map(l=>`<tr><th>L${l}</th>${F.map((f,i)=>cell(f,l).replace('<td',i===s?'<td class="cur"':'<td')).join('')}</tr>`).join('')}
    <tr class="out"><th>输出</th>${F.map((f,i)=>`<td${i===s?' class="cur"':''}>${escH(showTok(f.tok))}</td>`).join('')}</tr></tbody></table>`}

// ---- 生成回放：拖动滑块，脑图和层图回到生成某个字的那一刻
function addReplay(anchor,frames){
  const bar=document.createElement('div');bar.className='rp-bar';
  bar.innerHTML=`<button type="button">▶ 回放</button><input type="range" min="0" max="${frames.length-1}" value="${frames.length-1}" aria-label="回放位置"><span class="lab"></span><span class="tok"></span>`;
  anchor.after(bar);const [btn,rng,lab,tok]=[bar.querySelector('button'),bar.querySelector('input'),bar.querySelector('.lab'),bar.querySelector('.tok')];
  let timer=null;
  const show=k=>{const f=frames[k];lab.textContent=`第 ${k+1}/${frames.length} 段`;tok.textContent=showTok(f.t);
    if(f.e){act.fill(0);pushEnergy(f.e,false)}
    const lf=frames.slice(Math.max(0,k-15),k+1).filter(x=>x.lens).map(x=>({tok:x.t,rows:x.lens}));
    if(lf.length&&mode==='lens')drawLens(lf,lf.length-1)};
  rng.oninput=()=>{clearInterval(timer);timer=null;btn.textContent='▶ 回放';show(+rng.value)};
  btn.onclick=()=>{if(timer){clearInterval(timer);timer=null;btn.textContent='▶ 回放';return}
    if(+rng.value>=frames.length-1)rng.value=0;btn.textContent='❚❚ 暂停';
    timer=setInterval(()=>{const k=+rng.value;show(k);if(k>=frames.length-1){clearInterval(timer);timer=null;btn.textContent='▶ 回放';return}rng.value=k+1},120)};
  show(frames.length-1)}

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
 ['逐层预测（logit lens）','把某一层的中间结果直接送到模型的输出端，看看“如果在这一层停下”会输出什么字。通常前面的层在乱猜，越往后越接近最终答案——能看到答案是在第几层“想出来”的。'],
];
$('btnLearn').onclick=()=>{openOv('这张图怎么看？');$('mbody').innerHTML=`<div class="learn">${LEARN.map(([h,p])=>`<h4>${h}</h4><p>${p}</p>`).join('')}<p style="margin-top:12px">更详细的说明见项目里的 docs/SIGNALS.md。</p></div>`};
