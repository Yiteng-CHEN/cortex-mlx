// ======== 模型库 / 训练向导（需要本地服务） ========
async function api(path,body){
  const r=await fetch(path,{method:body===undefined?'GET':'POST',headers:{'Content-Type':'application/json'},body:body===undefined?undefined:JSON.stringify(body)});
  const j=await r.json().catch(()=>({}));if(!r.ok){const e=new Error(j.error||('HTTP '+r.status));e.code=j.code;throw e}return j}
const fmtB=b=>b>=1e9?(b/1e9).toFixed(1)+' GB':(b/1e6).toFixed(0)+' MB';
let libTimer=null,trTimer=null,libData=null,libMsg='',delArm=null,trainRunning=false,deltaReal=false,glow=new Float32Array(0),wiz={step:1,dataset:null,files:[],meta:null};
function openOv(t){$('mtitle').textContent=t;$('ov').hidden=false;document.body.style.overflow='hidden';$('mclose').focus()}
function closeOv(){$('ov').hidden=true;clearInterval(libTimer);clearInterval(trTimer);libTimer=trTimer=null;document.body.style.overflow=''}
$('mclose').onclick=closeOv;$('ov').addEventListener('mousedown',e=>{if(e.target===$('ov'))closeOv()});
addEventListener('keydown',e=>{if(e.key==='Escape'&&!$('ov').hidden)closeOv()});
const needLive=()=>`<div class="card"><b>这个功能需要本地服务</b><p class="hint">现在看到的是演示页面，没有连接到你电脑上的 Neurolume。在项目文件夹里双击 <b>Neurolume.command</b>，浏览器会自动打开完整版本；之后下载模型、加载、训练都能在界面上点按钮完成。</p></div>`;
$('btnLib').onclick=openLibrary;$('btnTrain').onclick=openTrain;

// ---- 模型库
async function openLibrary(){
  openOv('模型库');if(!LIVE){$('mbody').innerHTML=needLive();return}
  $('mbody').innerHTML=`<p class="hint" id="libnote"></p><div id="liblist" style="display:flex;flex-direction:column;gap:10px"></div>
   <form class="row" id="libadd"><input type="text" id="librepo" placeholder="添加其他模型：粘贴 Hugging Face 上的名字，如 mlx-community/…" aria-label="模型名"><button class="btn1" type="submit">检查并添加</button></form><div class="err" id="liberr"></div>
   <p class="hint">想换模型：先“下载”，下载完点“加载”即可。已下载的模型保存在项目的 models 文件夹里，不用的可以删除来腾出空间。</p>`;
  $('libadd').addEventListener('submit',async e=>{e.preventDefault();$('liberr').textContent='';const v=$('librepo').value.trim();if(!v)return;
    try{$('liberr').textContent='正在检查…';await api('/api/library/add',{repo:v});$('librepo').value='';$('liberr').textContent='';await refreshLib();await refreshModels()}catch(er){$('liberr').textContent=er.message}});
  $('liblist').addEventListener('click',libClick);
  await refreshLib();libTimer=setInterval(refreshLib,1500)}
async function refreshLib(){try{libData=await api('/api/library')}catch(e){return}renderLib()}
function renderLib(){
  if(!libData||!$('liblist'))return;const d=libData;
  $('libnote').textContent=`磁盘可用 ${fmtB(d.free_bytes)} · 模型保存在 ${d.hub}`;
  $('liblist').innerHTML=d.items.map(it=>{
    const j=it.job,running=j&&j.state==='running',pct=running&&j.total?Math.min(99,Math.round(j.bytes/j.total*100)):null,dump=(d.dumps||{})[it.key];
    const size=it.downloaded?fmtB(it.disk_bytes):(j&&j.total?'约 '+fmtB(j.total):(it.size?'约 '+fmtB(it.size):''));
    const pills=[it.loaded?'<span class="pill act">已加载</span>':'',it.downloaded?'<span class="pill ok">已下载</span>':(running?'<span class="pill act">下载中'+(pct!=null?' '+pct+'%':'')+'</span>':'<span class="pill">未下载</span>'),size?`<span class="pill">${size}</span>`:''].join(' ');
    let btns='';
    if(running)btns=`<button data-a="cancel" data-k="${it.key}">取消下载</button>`;
    else if(it.downloaded){
      btns=(it.loaded?`<button data-a="unload" data-k="${it.key}">卸载</button>`:`<button class="btn1" data-a="load" data-k="${it.key}">加载</button>`)
        +(it.weights?'':`<button data-a="weights" data-k="${it.key}" ${dump&&dump.state==='running'?'disabled':''}>${dump&&dump.state==='running'?'生成中…':'生成权重图'}</button>`)
        +`<button data-a="delete" data-k="${it.key}">${delArm===it.key?'再点一次确认删除':'删除'}</button>`}
    else btns=`<button class="btn1" data-a="download" data-k="${it.key}">下载</button>`;
    if(it.source==='custom'&&!running)btns+=`<button data-a="remove" data-k="${it.key}">从列表移除</button>`;
    return `<div class="card"><div class="row"><span class="nm">${it.name}</span>${pills}</div><div class="rp">${it.repo}${it.weights?' · 权重图已生成':''}</div>
      ${running?`<div class="prog"><i style="width:${pct||3}%"></i></div>`:''}${j&&j.state==='error'?`<div class="err">下载失败：${j.error}</div>`:''}${dump&&dump.state==='error'?`<div class="err">生成权重图失败：${dump.error}</div>`:''}
      <div class="row">${btns}</div></div>`}).join('')}
async function libClick(e){
  const b=e.target.closest('button');if(!b)return;const a=b.dataset.a,k=b.dataset.k;$('liberr').textContent='';
  try{
    if(a==='download')await api('/api/library/download',{key:k});
    else if(a==='cancel')await api('/api/library/cancel',{key:k});
    else if(a==='load'){closeOv();selectModel(k);return}
    else if(a==='unload'){applyState(await api('/api/unload',{}))}
    else if(a==='weights')await api('/api/library/weights',{key:k});
    else if(a==='remove'){await api('/api/library/remove',{key:k});await refreshModels()}
    else if(a==='delete'){if(delArm!==k){delArm=k;renderLib();setTimeout(()=>{if(delArm===k){delArm=null;renderLib()}},4000);return}
      delArm=null;const r=await api('/api/library/delete',{key:k});$('liberr').textContent='已删除，释放 '+fmtB(r.freed);await refreshModels()}
    await refreshLib()}catch(er){$('liberr').textContent=er.message}}
async function refreshModels(){try{fillOptions(await api('/api/state'))}catch(e){}}

// ---- 训练向导
const STEPS=['选数据','选模型和强度','训练','完成'];
async function openTrain(){
  openOv('训练');if(!LIVE){$('mbody').innerHTML=needLive();return}
  let st={state:'idle'};try{st=await api('/api/train/status')}catch(e){}
  wiz.step=st.state==='running'?3:(st.state==='done'?4:1);
  if(st.state==='failed'||st.state==='cancelled')wiz.step=3;
  renderWiz(st);trTimer=setInterval(pollTrain,1500)}
function stepsBar(){return `<div class="steps">${STEPS.map((s,i)=>`<span class="${wiz.step===i+1?'on':''}">${i+1} ${s}</span>`).join('')}</div>`}
async function renderWiz(st){
  const body=$('mbody');if(!body)return;
  if(wiz.step===1){
    let ds=[];try{ds=(await api('/api/train/datasets')).datasets}catch(e){}
    body.innerHTML=stepsBar()+`<p class="hint">训练让 AI 学会你的写法和做事套路（例如你的代码风格、常用做法）。想让它记住书本里的知识，放进知识库更合适——这个功能稍后加入。</p>
     <div class="drop" id="drop" tabindex="0" role="button">把文件拖到这里，或点击选择<br><small>支持 .jsonl（对话/问答）、.md、.txt 和代码文件；含密钥的样本会被自动剔除</small></div>
     <input type="file" id="fin" multiple hidden>
     <div id="flist" class="hint"></div><div class="row"><button class="btn1" id="fgo" disabled>导入并检查</button></div><div class="err" id="ferr"></div><div id="fres"></div>
     <div class="card"><div class="nm">没有现成的训练数据？</div><div class="rp">选几份你自己的资料（文档、笔记、代码），让当前加载的模型帮你生成问答对，你检查修改后再拿去训练。</div><div class="row"><button id="genOpen">用我的资料生成训练数据…</button></div></div>
     ${ds.length?`<div class="lab hint">或者使用已导入的数据</div>`+ds.map(d=>`<div class="card"><div class="row"><span class="nm">${d.name}</span><span class="pill">${d.n_train} 条训练 / ${d.n_valid} 条验证</span><span class="sp"></span><button data-use="${d.id}">用这份</button></div></div>`).join(''):''}`;
    const drop=$('drop'),fin=$('fin');
    const setFiles=fs=>{wiz.files=[...fs];$('flist').textContent=wiz.files.length?wiz.files.map(f=>f.name).join('、'):'';$('fgo').disabled=!wiz.files.length};
    drop.onclick=()=>fin.click();drop.onkeydown=e=>{if(e.key==='Enter'||e.key===' ')fin.click()};fin.onchange=()=>setFiles(fin.files);
    drop.ondragover=e=>{e.preventDefault();drop.classList.add('over')};drop.ondragleave=()=>drop.classList.remove('over');
    drop.ondrop=e=>{e.preventDefault();drop.classList.remove('over');setFiles(e.dataTransfer.files)};
    $('fgo').onclick=async()=>{$('ferr').textContent='';$('fgo').disabled=true;
      try{const files=await Promise.all(wiz.files.map(async f=>({name:f.name,text:await f.text()})));
        const meta=await api('/api/train/dataset',{name:wiz.files[0].name.replace(/\.[^.]+$/,''),files});showMeta(meta)}catch(er){$('ferr').textContent=er.message;$('fgo').disabled=false}};
    body.querySelectorAll('[data-use]').forEach(b=>b.onclick=()=>{const m=ds.find(d=>d.id===b.dataset.use);showMeta(m)});
    $('genOpen').onclick=()=>renderGen();
  }else if(wiz.step===2){
    const info=await api('/api/state'),real=info.models.filter(m=>!m.simulated&&m.downloaded),it=n=>plan(wiz.meta.n_train,n);
    body.innerHTML=stepsBar()+`<div class="card"><div class="nm">数据：${wiz.meta.name}</div><div class="rp">${wiz.meta.n_train} 条训练样本，${wiz.meta.n_valid} 条用来检验</div></div>
     ${real.length?'':'<div class="err">还没有已下载的模型。请先到“模型库”下载一个。</div>'}
     <label class="hint" for="tmodel">训练哪个模型</label><select id="tmodel">${real.map(m=>`<option value="${m.name}" ${m.name===info.active?'selected':''}>${m.name}</option>`).join('')}</select>
     <div class="hint">训练强度</div>
     <label class="card"><span class="row"><input type="radio" name="pre" value="quick" checked> <b>快速试训</b><span class="rp">约 ${it('quick')} 步 · 先确认流程能跑通、看看效果</span></span></label>
     <label class="card"><span class="row"><input type="radio" name="pre" value="standard"> <b>标准</b><span class="rp">约 ${it('standard')} 步 · 数据多时用</span></span></label>
     <p class="hint">开始后会先卸载当前模型来腾出内存，训练期间聊天暂停。原模型文件不会被改动：训练结果是一个单独的“适配器”，不满意可以随时不用。</p>
     <div class="row"><button id="tback">上一步</button><button class="btn1" id="tgo" ${real.length?'':'disabled'}>开始训练</button></div><div class="err" id="terr"></div>`;
    $('tback').onclick=()=>{wiz.step=1;renderWiz()};
    $('tgo').onclick=async()=>{$('tgo').disabled=true;try{await api('/api/train/start',{model:$('tmodel').value,dataset:wiz.meta.id,preset:body.querySelector('input[name=pre]:checked').value});
      wiz.step=3;glow=new Float32Array(NL);deltaReal=false;renderWiz(await api('/api/train/status'));applyState(await api('/api/state'))}catch(er){$('terr').textContent=er.message;$('tgo').disabled=false}};
  }else if(wiz.step===3){drawRun(st||await api('/api/train/status'))}
  else if(wiz.step===4){drawDone(st||await api('/api/train/status'))}}
const plan=(n,p)=>Math.round(Math.max(20,Math.min(p==='quick'?100:600,n*(p==='quick'?1:3))));
function showMeta(m){wiz.meta=m;$('fres').innerHTML=`<div class="card"><div class="nm">已准备好：${m.n_train} 条训练 + ${m.n_valid} 条验证</div>${(m.warnings||[]).map(w=>`<div class="err" style="color:var(--warn)">• ${w}</div>`).join('')}<pre class="pv">${(m.preview||[]).join('\n')}</pre><div class="row"><button class="btn1" id="fnext">下一步</button></div></div>`;
  $('fnext').onclick=()=>{wiz.step=2;renderWiz()}}
function drawRun(st){
  const pct=st.iters?Math.round(st.iter/st.iters*100):0,run=st.state==='running';
  const mm=s=>s==null?'—':(s>=3600?Math.floor(s/3600)+' 时 ':'')+Math.floor(s%3600/60)+' 分';
  $('mbody').innerHTML=stepsBar()+`<div class="row"><span class="nm">${st.state==='running'?'正在训练':st.state==='failed'?'训练失败':st.state==='cancelled'?'已取消':'训练'}</span><span class="rp">${st.model||''} · ${st.dataset_name||''}</span></div>
   <div class="prog"><i style="width:${pct}%"></i></div>
   <div class="kv"><div><small>进度</small><b>${st.iter||0} / ${st.iters||'—'}</b></div><div><small>训练损失</small><b>${st.loss==null?'—':st.loss.toFixed(3)}</b></div><div><small>验证损失</small><b>${st.val_loss==null?'—':st.val_loss.toFixed(3)}</b></div><div><small>速度</small><b>${st.tok_s?Math.round(st.tok_s)+' tok/s':'—'}</b></div><div><small>峰值内存</small><b>${st.peak_gb?st.peak_gb+' GB':'—'}</b></div><div><small>剩余时间</small><b>${run?mm(st.eta):'—'}</b></div></div>
   <canvas id="lossc" height="90" style="width:100%;border:1px solid var(--line);border-radius:8px"></canvas>
   <p class="hint">损失（loss）越低，说明它越能复现你的数据。你可以关闭这个窗口，训练会在后台继续；主页的“训练”视图会显示正在被修改的层和权重。</p>
   ${st.error?`<div class="err">${st.error}</div>`:''}
   <div class="row">${run?'<button id="tcancel">取消训练</button>':'<button class="btn1" id="tagain">重新开始</button>'}</div>`;
  if($('tcancel'))$('tcancel').onclick=async()=>{renderWiz(await api('/api/train/cancel',{}))};
  if($('tagain'))$('tagain').onclick=()=>{wiz.step=1;renderWiz()};
  const c=$('lossc'),x=c.getContext('2d'),w=c.width=c.clientWidth*2,h=c.height=180,hs=st.history||[];x.clearRect(0,0,w,h);
  if(hs.length>1){const ys=hs.map(a=>a[1]),mn=Math.min(...ys),mxv=Math.max(...ys),X=i=>10+(w-20)*i/(hs.length-1),Y=v=>h-14-(h-28)*(v-mn)/(mxv-mn+1e-9);
    x.strokeStyle='#4cc3d9';x.lineWidth=2.5;x.beginPath();hs.forEach((a,i)=>i?x.lineTo(X(i),Y(a[1])):x.moveTo(X(i),Y(a[1])));x.stroke()}}
function drawDone(st){
  const first=(st.history||[])[0],mk=st.model,vf=st.val_first,vl=st.val_loss;
  const pct=vf&&vl!=null?Math.round((vf-vl)/vf*100):null;
  const verdict=pct==null?'':pct>=5?'在没参与训练的验证数据上明显变好了。':pct>0?'在验证数据上略有改善，可以再多准备些数据或用“标准”强度。':'在验证数据上没有变好，可能是数据太少或和验证数据差别大，建议先对比看看回答。';
  $('mbody').innerHTML=stepsBar()+`<div class="card"><div class="nm">训练完成</div><div class="rp">训练损失 ${first?first[1].toFixed(3)+' → ':''}${st.loss==null?'—':st.loss.toFixed(3)} · 用时 ${Math.round(st.elapsed/60)} 分钟</div></div>
   ${vf!=null&&vl!=null?`<div class="card"><div class="nm">验证集损失：训练前 ${vf.toFixed(3)} → 训练后 ${vl.toFixed(3)}${pct!=null?`（${pct>=0?'降低':'升高'} ${Math.abs(pct)}%）`:''}</div><div class="rp">验证集是没参与训练的那部分数据，损失越低说明模型越会“举一反三”。${verdict}</div></div>`:''}
   <div class="card"><div class="nm">各层改变了多少</div><div class="lbars" id="lbars"></div><div class="rp">每根柱子是一层，高度是这一层被训练改变的幅度（适配器 ΔW 的大小）。主页的“训练”视图能看到具体位置。</div></div>
   <p class="hint">建议先用“试用并对比”看看同一个问题训练前后的回答，再决定是否继续使用。效果不好就点“恢复原模型”，不会有任何损失。</p>
   <div class="row"><button class="btn1" id="tuse">试用并对比</button><button id="tplain">恢复原模型（不用适配器）</button><button id="texp">导出…</button><button id="tagain">再训练一次</button></div><div class="err" id="terr"></div>`;
  drawLayerBars($('lbars'));
  $('tuse').onclick=async()=>{closeOv();await selectModel(mk,st.adapter);if(CUR_ADAPTER)openCompare()};
  $('texp').onclick=()=>openExport(st.adapter);
  $('tplain').onclick=async()=>{closeOv();selectModel(mk,null)};
  $('tagain').onclick=()=>{wiz.step=1;renderWiz()}}
async function pollTrain(){
  let st;try{st=await api('/api/train/status')}catch(e){return}
  trainRunning=st.state==='running';
  if(wiz.step===3&&st.state!=='running'&&st.state!=='idle'){if(st.state==='done'){wiz.step=4;await pullDelta(st.id);renderWiz(st)}else drawRun(st);return}
  if(wiz.step===3)drawRun(st)}
// 训练过程中，页面主体也同步：定时取适配器真实的权重变化
setInterval(async()=>{if(!LIVE)return;let st;try{st=await api('/api/train/status')}catch(e){return}
  const was=trainRunning;trainRunning=st.state==='running';
  if(trainRunning&&st.iter>0)await pullDelta(st.id);
  if(was&&!trainRunning&&st.state==='done')await pullDelta(st.id)},5000);
async function pullDelta(id){try{applyDelta(await api('/api/train/delta?adapter='+encodeURIComponent(id)))}catch(e){}}
function applyDelta(d){
  if(!d.layers||!d.layers.length)return;const idx={};MATS.forEach((m,k)=>idx[m.name]=k);
  glow=new Float32Array(NL);let mx=1e-12,gm=1e-12;
  for(const L of d.layers){if(L.i>=NL)continue;let tot=0;for(const m of L.m){const k=idx[m.name];if(k===undefined)continue;dW[L.i][k].set(m.v);for(const v of m.v)mx=Math.max(mx,Math.abs(v));tot+=m.norm}glow[L.i]=tot;gm=Math.max(gm,tot)}
  for(let l=0;l<NL;l++)glow[l]/=gm;dMax=mx;deltaReal=true;if(mode!=='train'){mode='train';setMode()}keyText()}

