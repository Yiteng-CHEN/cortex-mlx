// ======== 神经网络层图：矩阵点阵 + 层间连线 + 脉冲 ========
const CELL=5,TILE=CELL*G,COLW=TILE+44,TOP=26,ROWH=TILE+14;
const net=$('net'),nx=net.getContext('2d'),dpr=Math.min(devicePixelRatio||1,2);
let NETH=TOP+4*ROWH, mode='infer';
function buildNet(){
  net.width=NL*COLW*dpr;net.height=NETH*dpr;net.style.width=NL*COLW+'px';net.style.height=NETH+'px';
  $('gut').style.height=NETH+'px';$('gut').innerHTML='';
  MATS.forEach((m,k)=>{const s=document.createElement('span');s.textContent=m.label;s.style.top=(TOP+k*ROWH+TILE/2-7)+'px';$('gut').appendChild(s)});
}
// 每条连线（层 l 的矩阵 a → 层 l+1 的矩阵 b）固定一个随机相位和强度系数
const er=rng(5);const EDGE=[...Array(64*16)].map(()=>({ph:er(),k:.45+.55*er()}));
const tileX=l=>l*COLW+6, tileY=k=>TOP+k*ROWH;
function drawNet(t,isDark){
  nx.setTransform(dpr,0,0,dpr,0,0);nx.clearRect(0,0,NL*COLW,NETH);nx.font=MONO;
  const dim=isDark?'rgba(120,140,170,':'rgba(70,90,120,';
  // 1) 连线：未激活时淡淡的灰线，激活时按热度着色并有脉冲点沿线跑
  for(let l=0;l<NL-1;l++){
    const e=Math.sqrt(act[l]*act[l+1]);
    for(let a=0;a<4;a++)for(let b=0;b<4;b++){
      const ed=EDGE[(l%64)*16+a*4+b],x0=tileX(l)+TILE,y0=tileY(a)+TILE/2,x1=tileX(l+1),y1=tileY(b)+TILE/2,mx=(x0+x1)/2;
      const s=clamp(e*ed.k*1.15,0,1);
      if(s>.04){const h=heat(.25+.75*s);nx.strokeStyle=`rgba(${h[0]|0},${h[1]|0},${h[2]|0},${.25+.7*s})`;nx.lineWidth=.8+1.6*s}
      else{nx.strokeStyle=dim+'.16)';nx.lineWidth=.6}
      nx.beginPath();nx.moveTo(x0,y0);nx.bezierCurveTo(mx,y0,mx,y1,x1,y1);nx.stroke();
      if(s>.12){const u=(t*.0005*(.7+ed.k)+ed.ph)%1,it=1-u,bx=it*it*it*x0+3*it*it*u*mx+3*it*u*u*mx+u*u*u*x1,by=it*it*it*y0+3*it*it*u*y0+3*it*u*u*y1+u*u*u*y1,h=heat(.5+.5*s);
        nx.fillStyle=`rgba(${h[0]|0},${h[1]|0},${h[2]|0},${Math.min(1,s*1.3)})`;nx.beginPath();nx.arc(bx,by,1.5+1.8*s,0,6.283);nx.fill()}
    }
  }
  // 2) 残差流：每层顶部一段，按该层活动发亮
  for(let l=0;l<NL;l++){const e=act[l],h=heat(.15+.85*e);
    nx.fillStyle=e>.05?`rgba(${h[0]|0},${h[1]|0},${h[2]|0},${.35+.65*e})`:dim+'.25)';nx.fillRect(tileX(l)-6,16,COLW-8,3);
    nx.fillStyle=isDark?'#8a90a0':'#5d6578';nx.fillText('L'+l,tileX(l),11)}
  // 3) 矩阵点阵
  for(let l=0;l<NL;l++){
    const x0=tileX(l),e=act[l];
    for(let k=0;k<4;k++){
      const y0=tileY(k),w=W[l][k],d=dW[l][k];
      for(let r=0;r<G;r++)for(let c=0;c<G;c++){
        const i=r*G+c,v=w.v[i],mag=Math.min(1,Math.abs(v)/(3*w.std));let rgb,a;
        if(mode==='infer'){const lit=clamp(e*(.35+.9*mag),0,1);rgb=lit>.03?heat(.15+.85*lit):[110,130,165];a=lit>.03?.55+.45*lit:.1+.28*mag}
        else if(mode==='weights'){rgb=v>=0?[255,154,82]:[76,195,217];a=.1+.9*mag}
        else{const q=Math.min(1,Math.abs(d[i])/dMax*1.4);rgb=d[i]>=0?[184,227,106]:[255,120,80];a=.06+.94*q}
        nx.fillStyle=`rgba(${rgb[0]|0},${rgb[1]|0},${rgb[2]|0},${Math.min(1,a)})`;nx.fillRect(x0+c*CELL,y0+r*CELL,CELL-1,CELL-1);
      }
      if(hov&&hov.l===l&&hov.k===k){nx.strokeStyle=isDark?'#fff':'#000';nx.lineWidth=1.5;nx.strokeRect(x0+hov.c*CELL-.5,y0+hov.r*CELL-.5,CELL,CELL)}
    }
  }
}
let hov=null;const tip=$('tip');
net.addEventListener('pointermove',ev=>{
  const b=net.getBoundingClientRect(),x=ev.clientX-b.left,y=ev.clientY-b.top,l=Math.floor(x/COLW),px=x-tileX(l),k=Math.floor((y-TOP)/ROWH),py=y-TOP-k*ROWH;
  const c=Math.floor(px/CELL),r=Math.floor(py/CELL);
  if(l<0||l>=NL||k<0||k>3||px<0||py<0||c<0||c>=G||r<0||r>=G){hov=null;tip.hidden=true;return}
  hov={l,k,r,c};const w=W[l][k],i=r*G+c,sh=w.shape;
  let t=`<b>第 ${l} 层 · ${MATS[k].name}</b><br>位置：行 ${pick(sh[0],r)} / ${sh[0]}，列 ${pick(sh[1],c)} / ${sh[1]}<br>权重：${w.v[i].toFixed(5)}`+(realWeights?'':'（示例）')+`<br>该矩阵标准差：${w.std.toFixed(4)}`;
  if(mode==='infer')t+=`<br>该层当前活动：${act[l].toFixed(2)}${SIM&&!DEMO?'（模拟）':''}`+(lastRaw&&(!SIM||DEMO)&&lastRaw[l]!=null?`<br>测量值：${lastRaw[l].toFixed(4)}`:'');
  t+=`<br><span style="color:var(--mute)">${MAT_EXPL[k]}</span>`;
  if(mode==='train')t+=deltaReal?`<br>ΔW：${dW[l][k][i]>=0?'+':''}${dW[l][k][i].toExponential(2)}（真实，来自适配器）`:'<br>ΔW：还没有训练结果';
  tip.innerHTML=t;tip.hidden=false;
  const tw=tip.offsetWidth,th=tip.offsetHeight;tip.style.left=Math.min(innerWidth-tw-8,ev.clientX+14)+'px';tip.style.top=Math.max(8,Math.min(innerHeight-th-8,ev.clientY+14))+'px';
});
net.addEventListener('pointerleave',()=>{hov=null;tip.hidden=true});
function setMode(){document.querySelectorAll('#modes button').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.m===mode)));
  const lens=mode==='lens';$('nwrap').hidden=lens;$('lensv').hidden=!lens;$('step').hidden=lens;$('trainbars').hidden=mode!=='train';
  if(typeof tlRender==='function')tlRender();if(mode==='train'&&typeof reviewDelta==='function')reviewDelta();keyText()}
const MAT_EXPL=['Q（查询）：这一层的每个字“想找什么信息”','输出：把注意力找到的信息写回这个字的表示里','门控：决定 MLP 里哪些“知识通道”打开','下投影：把 MLP 算出的结果压回原来的维度，写回去'];
function keyText(){
  $('key').innerHTML={
   infer:`<span>低<i class="grad"></i>高：该层这一步对当前 token 的改动幅度${SIM&&!DEMO?'（模拟）':''}</span><span>连线发光 = 相邻两层同时活跃，亮点沿线移动</span>`,
   train:`<span><i class="sw" style="background:#b8e36a"></i>权重增大</span><span><i class="sw" style="background:#ff7850"></i>权重减小</span><span>${deltaReal?'ΔW 来自适配器的真实值（与权重图同一位置）':'还没有训练结果：点“去训练…”完成一次训练后，这里显示适配器真实改动的位置和大小'}</span>`,
   lens:`<span>每一列是一个生成的字，每一行是一层：格子里是这一层“如果在这里停下”最可能输出的字，颜色越深越有把握。最下面一行是模型真正输出的字。</span>`,
   weights:`<span><i class="sw" style="background:#ff9a52"></i>正权重</span><span><i class="sw" style="background:#4cc3d9"></i>负权重</span><span>颜色深浅 = |权重| ÷ 该矩阵标准差（3σ 封顶）</span>`}[mode]}
$('modes').addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;mode=b.dataset.m;setMode()});
$('step').onclick=()=>openTrain();

