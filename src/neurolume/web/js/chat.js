// ======== 聊天 ========
const history=[];
let AGENT=false;try{AGENT=localStorage.getItem('neurolume.agent')==='1'}catch(_){}
let BOOKS=false;try{BOOKS=localStorage.getItem('neurolume.books')==='1'}catch(_){}
let WEB=false;try{WEB=localStorage.getItem('neurolume.web')==='1'}catch(_){}
function esc(s){return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
function render(el,text){ // 极简：``` 代码块
  const parts=text.split('```');el.innerHTML=parts.map((p,i)=>i%2?`<pre>${esc(p.replace(/^\w*[ \t]*\n/,''))}</pre>`:esc(p)).join('')}
function addMsg(role,text){const d=document.createElement('div');d.className='m '+role;render(d,text);$('msgs').appendChild(d);$('msgs').scrollTop=1e9;return d}
function sysMsg(t){const d=addMsg('sys',t);return d}
function setBadge(){
  const b=$('mode');
  if(LIVE&&active&&!SIM){b.className='badge live';b.textContent='实时 · 真实模型';$('scanSig').innerHTML='信号：<b>来自模型的实时逐层活动</b>'}
  else if(LIVE&&active){b.className='badge demo';b.textContent='演示后端 · 模拟';$('scanSig').innerHTML='信号：<b>模拟</b>'}
  else if(LIVE){b.className='badge demo';b.textContent='未加载模型';$('scanSig').innerHTML='信号：<b>等待模型</b>'}
  else{b.className='badge demo';b.textContent='演示模式 · 未连接本地服务';$('scanSig').innerHTML='信号：<b>模拟</b>'}
  SIM=!(LIVE&&active&&!SIM_BACKEND);keyText()
}
let SIM_BACKEND=true;
async function send(text){
  if(busy)return;
  if(LIVE&&AGENT){let ai={};try{ai=await api('/api/agent')}catch(_){}if(!ai.workspace){sysMsg('已开启 Agent，但还没有选择项目文件夹：请先点“Agent”按钮选择。');return}}
  busy=true;$('send').disabled=true;
  addMsg('u',text);history.push({role:'user',content:text});
  const el=addMsg('a','');let out='',gotEnergy=false,st=null,srcs=null,cur=el,seg='',last=null,simResp=false;const cards={},frames=[];
  const place=n=>{(last||el).after(n);last=n};
  const toAgent=()=>{if(cur===el&&!seg){el.hidden=true}cur=null};
  try{
    if(LIVE&&active){
      const r=await fetch('/v1/chat/completions',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:active,stream:true,web:WEB,books:BOOKS,agent:AGENT,lens:mode==='lens',lang:LANG,messages:history,max_tokens:AGENT?3072:1024})});
      if(!r.ok){const j=await r.json().catch(()=>({}));throw new Error(j.error||r.status)}
      const rd=r.body.getReader(),dec=new TextDecoder();let buf='';
      for(;;){const{done,value}=await rd.read();if(done)break;buf+=dec.decode(value,{stream:true});
        let i;while((i=buf.indexOf('\n\n'))>=0){const line=buf.slice(0,i).trim();buf=buf.slice(i+2);if(!line.startsWith('data:'))continue;const p=line.slice(5).trim();if(p==='[DONE]')continue;
          const o=JSON.parse(p);if(o.error)throw new Error(o.error);const d=o.choices&&o.choices[0].delta;
          if(d&&d.content){if(!cur){cur=document.createElement('div');cur.className='m a';place(cur);seg='';if(out)out+='\n\n'}out+=d.content;seg+=d.content;render(cur,seg);$('msgs').scrollTop=1e9}
          const nx=o.neurolume;if(nx&&nx.simulated)simResp=true;if(nx&&(nx.energy||nx.lens||nx.conf!=null)&&!simResp){frames.push({t:(d&&d.content)||'',tok:nx.token,e:nx.energy||null,lens:nx.lens||null,p:nx.conf,alts:nx.alts||null,gid:nx.gid,el:cur||el});tlLive(frames)}
          if(o.neurolume&&o.neurolume.agent){toAgent();onAgent(o.neurolume.agent,place,cards);$('msgs').scrollTop=1e9}
          if(o.neurolume&&o.neurolume.status){if(!st){st=document.createElement('div');st.className='m sys';el.after(st)}st.textContent='🌐 '+o.neurolume.status}
          if(o.neurolume&&o.neurolume.sources)srcs=(srcs||[]).concat(o.neurolume.sources);
          if(o.neurolume){if(o.neurolume.energy){gotEnergy=true;pushEnergy(o.neurolume.energy,o.neurolume.simulated)}if(o.neurolume.tps)$('scanTps').textContent=o.neurolume.tps+' tok/s'}}}
    }else{ // 演示：不连接任何模型
      const demo="这是演示回复：页面没有连接到本地服务，所以回答和各层的活动都是模拟的。\n\n在 Mac 上运行 `uv run --extra mlx neurolume serve --model devstral-small-2` 后打开 http://127.0.0.1:8765，这里就会显示 Devstral 真实生成的文字，大脑图和网络层会用它真实的逐层活动着色。";
      const words=demo.split(/(?<=\s)/);let k=0;
      for(const w of words){out+=w;render(el,out);$('msgs').scrollTop=1e9;
        const c=(k*2.7)%(NL+8)-4;pushEnergy(Array.from({length:NL},(_,i)=>Math.exp(-(((i-c)/3)**2))*(.65+.35*((i*7+k)%5)/4)),true);k++;await new Promise(r=>setTimeout(r,45))}
    }
    if(srcs&&srcs.length){const d=document.createElement('div');d.className='m sys';d.style.textAlign='left';
      d.append('来源：');srcs.forEach((x,i)=>{const a=document.createElement(x.url?'a':'span');if(x.url)a.href=x.url;a.target='_blank';a.rel='noopener noreferrer';a.textContent=`[${i+1}] ${x.title||x.url}`;a.style.marginRight='10px';d.appendChild(a)});
      (st||last||el).after(d)}
    if(st)st.textContent='🌐 已联网搜索';
    history.push({role:'assistant',content:out});
    if(LIVE&&!simResp&&frames.length)tlFinish(frames,[...new Set(frames.map(f=>f.el))]);
    if(LIVE&&active&&!AGENT&&!simResp&&out.trim())sampleBar(last||el,text,out);
    if(LIVE&&active&&!SIM_BACKEND&&!gotEnergy)sysMsg('提示：这次回复没有收到逐层活动信号，所以立柱和层图不会亮。请把终端窗口里最后 20 行发给开发者。');
  }catch(e){el.hidden=false;el.className='m sys';el.textContent='出错：'+e.message;history.pop()}
  busy=false;$('send').disabled=false;$('inp').focus();
}
// ---- 存为训练样本：满意的回答直接存；不满意就先改成你想要的样子再存（这正是训练要学的）
function sampleBar(after,q,a){const d=document.createElement('div');d.className='m-act';
  d.innerHTML=`<button type="button" data-k="save" title="把这一问一答存进训练样本，攒够了在“训练”里一键变成数据集">存为训练样本</button><button type="button" data-k="edit" title="先把问题或回答改成你真正想要的样子，再存">改一改再存…</button><span class="m-note"></span>`;
  after.after(d);const note=d.querySelector('.m-note');
  const done=r=>{d.querySelectorAll('button').forEach(b=>b.disabled=true);note.textContent=r.duplicate?`这条已经存过了（共 ${r.n} 条）`:`✓ 已存为训练样本（共 ${r.n} 条）`};
  d.querySelector('[data-k=save]').onclick=async()=>{try{done(await api('/api/samples/add',{messages:[{role:'user',content:q},{role:'assistant',content:a}]}))}catch(e){note.textContent=e.message}};
  d.querySelector('[data-k=edit]').onclick=()=>editSample(q,a,done)}
function editSample(q,a,done){openOv('改一改再存为训练样本');
  $('mbody').innerHTML=`<p class="hint">训练就是让模型学会“看到这样的问题，就这样回答”。把回答改成你真正想要的样子（内容、语气、格式都可以改），模型学的就是改好的版本。</p>
   <label class="hint" for="sq">问题</label><textarea id="sq" class="smp" rows="3"></textarea>
   <label class="hint" for="sa">回答</label><textarea id="sa" class="smp" rows="12"></textarea>
   <div class="row"><button class="btn1" id="ssave">存为训练样本</button><button id="scancel">取消</button></div><div class="err" id="serr"></div>`;
  $('sq').value=q;$('sa').value=a;$('sa').focus();$('scancel').onclick=closeOv;
  $('ssave').onclick=async()=>{$('serr').textContent='';try{const r=await api('/api/samples/add',{messages:[{role:'user',content:$('sq').value},{role:'assistant',content:$('sa').value}],edited:$('sq').value!==q||$('sa').value!==a});closeOv();done(r)}catch(e){$('serr').textContent=e.message}}}
$('form').addEventListener('submit',e=>{e.preventDefault();const t=$('inp').value.trim();if(!t)return;$('inp').value='';send(t)});
$('inp').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey&&!e.isComposing){e.preventDefault();$('form').requestSubmit()}});
function setWeb(v){WEB=v;try{localStorage.setItem('neurolume.web',v?'1':'0')}catch(_){}$('btnWeb').setAttribute('aria-pressed',String(v));$('btnWeb').textContent=v?'联网搜索：开':'联网搜索'}
$('btnWeb').onclick=()=>{if(!LIVE){sysMsg('联网搜索需要先启动本地服务。');return}setWeb(!WEB);sysMsg(WEB?'已开启联网搜索：模型需要时会自己搜索，你的问题会发给搜索服务。':'已关闭联网搜索：Neurolume 不会向外发送任何内容。')};
$('btnWebSet').onclick=async()=>{
  openOv('搜索设置');if(!LIVE){$('mbody').innerHTML=needLive();return}
  let c={provider:'duckduckgo',searxng_url:''};try{c=await api('/api/search')}catch(_){}
  $('mbody').innerHTML=`<p class="hint">联网搜索打开后，模型需要最新信息时会自己搜索并阅读网页，回答里会标出来源。默认关闭；搜索时你的问题会发给所选搜索服务。网页内容只当作资料，不会当作指令执行；不会读取本机或内网地址。</p>
   <form id="wform" style="display:flex;flex-direction:column;gap:10px">
    <label><input type="radio" name="wp" value="duckduckgo" ${c.provider==='duckduckgo'?'checked':''}> DuckDuckGo（免费，不需要账号）</label>
    <label><input type="radio" name="wp" value="searxng" ${c.provider==='searxng'?'checked':''}> 我自己的 SearXNG（更稳定、更隐私，需要自己部署）</label>
    <input type="text" id="wurl" placeholder="SearXNG 地址，如 http://localhost:8080" value="${c.searxng_url.replace(/"/g,'&quot;')}" aria-label="SearXNG 地址">
    <div class="row"><button class="btn1" type="submit">保存</button><button type="button" id="wtest">测试搜索</button></div><div class="err" id="werr"></div><div class="hint" id="wres"></div></form>`;
  const cfg=()=>({provider:document.querySelector('input[name=wp]:checked').value,searxng_url:$('wurl').value.trim()});
  $('wform').addEventListener('submit',async e=>{e.preventDefault();$('werr').textContent='';try{await api('/api/search',cfg());$('wres').textContent='已保存。'}catch(er){$('werr').textContent=er.message}});
  $('wtest').onclick=async()=>{$('werr').textContent='';$('wres').textContent='正在测试…';try{await api('/api/search',cfg());const r=await api('/api/search/test',{q:'Apple MLX'});
    if(r.ok){$('wres').textContent='成功，第一条结果：'+(r.results[0]?r.results[0].title:'(空)')}else{$('wres').textContent='';$('werr').textContent=r.error}}catch(er){$('wres').textContent='';$('werr').textContent=er.message}}
};
function onAgent(a,place,cards){
  if(a.kind==='tool'||(a.kind==='result'&&!a.proposal)){const d=document.createElement('div');d.className='m sys agl';d.textContent=(a.ok?'🔧 ':'⚠️ ')+a.name+' '+a.args+(a.ok?'':'  → '+a.summary);place(d);return}
  if(a.kind==='proposal')return place(buildCard(a.proposal,cards));
  if(a.kind==='result'&&a.proposal&&cards[a.proposal.id])setCard(cards[a.proposal.id],a.proposal)}
const PST={pending:'等待你的批准',applied:'✓ 已应用',rejected:'已拒绝',failed:'写入失败',cancelled:'已取消',undone:'已撤销'};
function setCard(c,p){c.st.textContent=PST[p.status]||p.status;const done=p.status!=='pending';c.ok.disabled=done;c.no.disabled=done;c.ok.hidden=c.no.hidden=done}
function buildCard(p,cards){
  const d=document.createElement('div');d.className='m sys pcard';
  const h=document.createElement('div');h.className='ph';h.textContent='📝 '+p.path+(p.is_new?'  （新文件）':'');d.appendChild(h);
  const pre=document.createElement('pre');pre.className='diff';
  p.diff.split('\n').slice(0,400).forEach(l=>{const sp=document.createElement('span');sp.className=l.startsWith('+')&&!l.startsWith('+++')?'da':l.startsWith('-')&&!l.startsWith('---')?'dd':l.startsWith('@@')||l.startsWith('---')||l.startsWith('+++')?'dh':'';sp.textContent=l+'\n';pre.appendChild(sp)});d.appendChild(pre);
  const row=document.createElement('div');row.className='pb';
  const ok=document.createElement('button');ok.className='btn1';ok.textContent='应用修改';const no=document.createElement('button');no.textContent='拒绝';
  const st=document.createElement('span');st.className='hint';row.append(ok,no,st);d.appendChild(row);
  const c={ok,no,st};cards[p.id]=c;setCard(c,p);
  const decide=async yes=>{ok.disabled=no.disabled=true;try{setCard(c,await api('/api/agent/decide',{id:p.id,approve:yes}))}catch(e){st.textContent=e.message}};
  ok.onclick=()=>decide(true);no.onclick=()=>decide(false);return d}
function setAgent(v){AGENT=v;try{localStorage.setItem('neurolume.agent',v?'1':'0')}catch(_){}$('btnAgent').textContent=v?'Agent：开':'Agent'}
setAgent(AGENT);
async function renderAgent(msg){
  let ai={workspace:null,changes:[]};try{ai=await api('/api/agent')}catch(e){msg=e.message}
  $('mbody').innerHTML=`<p class="hint">选一个项目文件夹后，模型可以查看里面的文件、搜索代码，并提出修改。每一处修改都会先显示 diff，等你点“应用修改”才会真正写入；写入后可以在下面撤销。密钥、.env 和 .git 里的文件不会被读取或修改。</p>
   <label style="display:flex;gap:8px;align-items:center"><input type="checkbox" id="agOn" ${AGENT?'checked':''}> 聊天时启用 Agent（读取项目并提出修改）</label>
   <div class="card"><div class="nm">项目文件夹</div><div class="rp" id="agWs">${ai.workspace?escA(ai.workspace):'还没有选择'}</div>
    <div class="row"><button type="button" id="agPick" class="btn1">选择文件夹…</button><button type="button" id="agClear" ${ai.workspace?'':'disabled'}>取消选择</button></div>
    <div class="row"><input type="text" id="agPath" placeholder="或者粘贴文件夹路径，如 ~/Documents/myproject" aria-label="文件夹路径"><button type="button" id="agSet">使用这个路径</button></div></div>
   <div class="err" id="agErr"></div><div class="hint" id="agRes">${escA(msg||'')}</div>
   <div class="lab hint">最近的修改</div>
   <div style="display:flex;flex-direction:column;gap:6px">${ai.changes.length?ai.changes.map(c=>`<div class="row" style="justify-content:space-between"><span class="rp">${escA(c.path)}${c.is_new?'（新文件）':''} · ${escA(c.created)}</span>${c.undone?'<span class="hint">已撤销</span>':`<button type="button" data-undo="${c.id}">撤销</button>`}</div>`).join(''):'<div class="hint">还没有修改。</div>'}</div>`;
  $('agOn').onchange=e=>{setAgent(e.target.checked);if(AGENT&&!ai.workspace)$('agRes').textContent='请再选择一个项目文件夹。'};
  const setWs=async p=>{$('agErr').textContent='';try{await api('/api/agent/workspace',{path:p});setAgent(true);await renderAgent('已设置项目文件夹，并开启了 Agent。')}catch(e){$('agErr').textContent=e.message}};
  $('agPick').onclick=async()=>{$('agErr').textContent='';$('agRes').textContent='请在弹出的窗口里选择文件夹…';try{const r=await api('/api/agent/pick',{});if(r.path)await setWs(r.path);else $('agRes').textContent=''}catch(e){$('agRes').textContent='';$('agErr').textContent=e.message}};
  $('agSet').onclick=()=>setWs($('agPath').value);
  $('agClear').onclick=async()=>{try{await api('/api/agent/workspace',{path:''});await renderAgent('已取消选择。')}catch(e){$('agErr').textContent=e.message}};
  document.querySelectorAll('[data-undo]').forEach(b=>b.onclick=async()=>{try{await api('/api/agent/undo',{id:b.dataset.undo});await renderAgent('已撤销。')}catch(e){$('agErr').textContent=e.message}});
}
$('btnAgent').onclick=()=>{openOv('Agent');if(!LIVE){$('mbody').innerHTML=needLive();return}renderAgent()};
function setBooks(v){BOOKS=v;try{localStorage.setItem('neurolume.books',v?'1':'0')}catch(_){}$('btnBooks').textContent=v?'知识库：开':'知识库'}
setBooks(BOOKS);
const escA=t=>String(t).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const b64of=f=>new Promise((ok,no)=>{const r=new FileReader();r.onload=()=>ok(String(r.result).split(',')[1]||'');r.onerror=()=>no(new Error('读取文件失败'));r.readAsDataURL(f)});
let bkArm=null;
async function renderBooks(msg){
  let list=[];try{list=(await api('/api/books')).books}catch(e){msg=e.message}
  const tot=list.filter(b=>b.enabled).reduce((a,b)=>a+b.chunks,0);
  $('mbody').innerHTML=`<p class="hint">把书和文档导入到这里，对话时模型会先从书里找出相关段落再回答，并标出引用的是哪本书的第几段。全部在你自己的电脑上处理，不会上传。支持 txt / md / html / epub / docx / 代码文件；PDF 需要电脑里装有 pypdf 或 pdftotext，否则请先导出为 txt 或 epub。</p>
   <label style="display:flex;gap:8px;align-items:center;margin:8px 0"><input type="checkbox" id="bkOn" ${BOOKS?'checked':''}> 聊天时使用知识库${tot?`（当前启用 ${tot} 段）`:''}</label>
   <div class="row"><input type="file" id="bkFile" multiple accept=".txt,.md,.markdown,.rst,.html,.htm,.epub,.docx,.pdf,.py,.js,.ts,.c,.cpp,.h,.java,.go,.rs,.swift,.json,.csv,.tex" aria-label="选择要导入的书"></div>
   <div class="err" id="bkErr"></div><div class="hint" id="bkRes">${escA(msg||'')}</div>
   <div id="bkList" style="margin-top:10px;display:flex;flex-direction:column;gap:6px">${list.length?list.map(b=>`<div class="row" style="justify-content:space-between;gap:8px"><label style="display:flex;gap:8px;align-items:center;min-width:0"><input type="checkbox" data-en="${b.id}" ${b.enabled?'checked':''}><span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${escA(b.title)}</span></label><span class="hint">${b.chunks} 段 · ${(b.chars/1e4).toFixed(1)} 万字</span><button type="button" data-rm="${b.id}">${bkArm===b.id?'再点一次确认':'删除'}</button></div>`).join(''):'<div class="hint">还没有导入任何书。</div>'}</div>`;
  $('bkOn').onchange=e=>{setBooks(e.target.checked);sysMsg(BOOKS?'已开启知识库：回答前会先从你导入的书里查资料。':'已关闭知识库。')};
  document.querySelectorAll('[data-en]').forEach(c=>c.onchange=async()=>{try{await api('/api/books/enable',{id:c.dataset.en,on:c.checked});renderBooks()}catch(e){$('bkErr').textContent=e.message}});
  document.querySelectorAll('[data-rm]').forEach(b=>b.onclick=async()=>{if(bkArm!==b.dataset.rm){bkArm=b.dataset.rm;renderBooks();return}bkArm=null;try{await api('/api/books/remove',{id:b.dataset.rm})}catch(e){}renderBooks('已删除。')});
  $('bkFile').onchange=async e=>{const fs=[...e.target.files];if(!fs.length)return;let ok=0,errs=[];
    for(const f of fs){$('bkRes').textContent=`正在导入 ${f.name}（${ok+errs.length+1}/${fs.length}），大书需要一会儿…`;
      try{if(f.size>55e6)throw new Error('文件超过 55MB');const m=await api('/api/books/add',{name:f.name.replace(/\.[^.]+$/,''),filename:f.name,data:await b64of(f)});ok++;if(!BOOKS)setBooks(true)}catch(er){errs.push(f.name+'：'+er.message)}}
    await renderBooks(`导入完成：成功 ${ok} 本${errs.length?'，失败 '+errs.length+' 本':''}。`);if(errs.length)$('bkErr').textContent=errs.join('；')}
}
$('btnBooks').onclick=()=>{if(!LIVE){openOv('知识库');$('mbody').innerHTML=needLive();return}openOv('知识库');renderBooks()};
$('clear').onclick=()=>{history.length=0;$('msgs').innerHTML='';sysMsg(LIVE?'已清空。':'已清空（演示模式）。')};
$('adapter').addEventListener('change',()=>{if(LIVE&&active)selectModel(active,$('adapter').value||null)});
$('model').addEventListener('change',()=>{const k=$('model').value;if(!k)return;
  const o=$('model').selectedOptions[0];if(LIVE&&o&&o.dataset.dl==='0'){openLibrary();refreshModels();return}setStructure(k);if(LIVE)selectModel(k);else sysMsg('演示模式：只切换了结构预览，回复仍是模拟的。')});

async function selectModel(name,adapter){
  $('model').disabled=true;$('send').disabled=true;const m=sysMsg(`正在加载 ${name}…（首次约一分钟，会先卸载上一个模型）`);
  try{
    const r=await fetch('/api/select',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name,adapter:adapter||null})}),j=await r.json();
    if(!r.ok)throw new Error(j.error);applyState(j);history.length=0;m.textContent=`${name} 已加载${adapter?'（带适配器）':''}。`;
  }catch(e){m.textContent='加载失败：'+e.message;try{applyState(await (await fetch('/api/state')).json())}catch(_){}}
  $('model').disabled=false;$('send').disabled=busy;
}
let _wTok=0;
async function loadWeightsFor(name){
  const tok=++_wTok;
  try{const url=LIVE?`/api/weights?model=${encodeURIComponent(name)}`:(name==='devstral-small-2'?'weights-devstral-small-2.json':null);
    if(!url)throw 0;const r=await fetch(url);if(!r.ok)throw 0;const j=await r.json();if(tok!==_wTok||j.layers.length!==NL)throw 0;loadWeights(j)}
  catch(e){if(!W||W.length!==NL||realWeights)synth();if(LIVE&&name&&name===active&&!SIM_BACKEND&&WSTATE!=='loading')setTimeout(()=>genWeights(name),300)}
  noteData();
}
let adaptersAll=[];
function fillOptions(s){
  const sel=$('model');
  sel.innerHTML=s.models.map(m=>`<option value="${m.name}" data-dl="${m.downloaded?1:0}" ${m.name===s.active?'selected':''}>${m.name}${m.simulated?'（演示）':''}${m.loaded?' ✓':''}${m.downloaded?'':' · 未下载'}</option>`).join('');
  if(!s.active)sel.insertAdjacentHTML('afterbegin','<option value="" selected disabled>选择一个模型…</option>');
  fillAdapters(s)}
async function fillAdapters(s){
  const ad=$('adapter');try{adaptersAll=(await api('/api/train/adapters')).adapters}catch(e){adaptersAll=[]}
  const mine=adaptersAll.filter(a=>a.model===s.active);ad.hidden=!s.active||!mine.length;
  ad.innerHTML='<option value="">不使用适配器</option>'+mine.map(a=>`<option value="${a.id}" ${a.id===s.adapter?'selected':''}>适配器 ${a.created}${a.loss!=null?' · loss '+a.loss.toFixed(2):''}</option>`).join('')}
function applyState(s){
  fillOptions(s);
  active=s.active;SIM_BACKEND=s.simulated!==false;trainRunning=!!s.training;
  if(active)setStructure(active,s.layers);
  SIM=!(LIVE&&active&&!SIM_BACKEND);setBadge();
  CUR_ADAPTER=s.adapter||null;$('btnCompare').hidden=!(LIVE&&s.adapter)
}
async function boot(){
  dW=[...Array(NL)].map(()=>MATS.map(()=>new Float32Array(144)));synth();buildNet();setMode();
  try{const r=await fetch('/api/state');if(!r.ok)throw 0;const s=await r.json();if(!s.models)throw 0;LIVE=true;applyState(s);
    sysMsg(s.active?`已连接。当前模型：${s.active}`:'已连接本地服务。请在上方选择一个模型开始。')}
  catch(e){LIVE=false;$('model').innerHTML='<option value="devstral-small-2">Devstral Small 2（演示）</option><option value="qwen3.6-27b">Qwen3.6-27B（演示）</option>';$('scanModel').textContent=KNOWN['devstral-small-2'].label;setBadge();
    sysMsg('当前是演示模式：回复和活动均为模拟。启动本地服务后会自动连接真实模型。');loadWeightsFor('devstral-small-2')}
  noteData();
}


