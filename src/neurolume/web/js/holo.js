// ======== 3D 全息神经网络（原创：每层一圈节点；相邻层反向旋转；一圈 = 一层，和模型真实层数一致） ========
const cv=$('gl'),renderer=new THREE.WebGLRenderer({canvas:cv,antialias:true,alpha:true});
renderer.setPixelRatio(Math.min(devicePixelRatio||1,2));
const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(38,1,.1,50);camera.position.set(0,.15,6.3);
const rig=new THREE.Group();scene.add(rig);
let isDark=true;
const ADD=THREE.AdditiveBlending,hud=$('hud'),hx=hud.getContext('2d');
let holo=null,H={},PER=36;
const Rs=(l,n)=>{const y=Yl(l,n)/1.14;return .1+.95*Math.sqrt(Math.max(0,1-y*y))}, Yl=(l,n)=>-1.05+2.1*l/Math.max(1,n-1);
const GOLD=[[.05,.03,.01],[.55,.28,.04],[1,.55,.1],[1,.82,.3],[1,.97,.82]];
function gold(e,o,k){e=clamp(e,0,1)*4;const i=Math.min(3,e|0),f=e-i,a=GOLD[i],b=GOLD[i+1];o[0]=(a[0]+(b[0]-a[0])*f)*k;o[1]=(a[1]+(b[1]-a[1])*f)*k;o[2]=(a[2]+(b[2]-a[2])*f)*k}
function mkLines(segs,color,op){const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.BufferAttribute(new Float32Array(segs),3));
  return new THREE.LineSegments(g,new THREE.LineBasicMaterial({color,transparent:true,opacity:op,blending:ADD,depthWrite:false}))}
function arcLines(rad,start,span,n,color,op){const a=[];for(let i=0;i<n;i++){const t0=start+span*i/n,t1=start+span*(i+.6)/n;a.push(Math.cos(t0)*rad,0,Math.sin(t0)*rad,Math.cos(t1)*rad,0,Math.sin(t1)*rad)}return mkLines(a,color,op)}
function dashedRing(rad,dashes,color,op){return arcLines(rad,0,6.2832,dashes,color,op)}
let _hdark=null;   // 白色主题下 ADD 叠加混合会让球体消失，所以换成普通混合
function applyHoloTheme(){if(!holo||_hdark===isDark)return;_hdark=isDark;const bm=isDark?ADD:THREE.NormalBlending;
  (holo.children||[]).forEach(o=>{if(o.material){o.material.blending=bm;o.material.needsUpdate=true}})}
let SHAPE='nebula';try{SHAPE=localStorage.getItem('neurolume.shape')==='sphere'?'sphere':'nebula'}catch(_){}
function hazeTex(){try{if(typeof document==='undefined'||!THREE.CanvasTexture)return null;const c=document.createElement('canvas');c.width=c.height=64;const g=c.getContext('2d'),gr=g.createRadialGradient(32,32,0,32,32,32);
  gr.addColorStop(0,'rgba(255,255,255,1)');gr.addColorStop(.3,'rgba(255,255,255,.35)');gr.addColorStop(1,'rgba(255,255,255,0)');g.fillStyle=gr;g.fillRect(0,0,64,64);return new THREE.CanvasTexture(c)}catch(_){return null}}
function gauss(r){return (r()+r()+r()+r()-2)*.866}   // 近似正态
const nebPr=u=>.45+.55*Math.pow(Math.sin(Math.PI*clamp(u*.92+.04,0,1)),.6);
const nebCx=u=>.34*Math.sin(u*5.1+.7)+.14*Math.sin(u*11+2),nebCz=u=>.34*Math.cos(u*4.3+2.1)+.14*Math.cos(u*9.7);
function nebPt(r,u,l,sp){   // 旋臂：角度随距离卷曲，形成漩涡状的气体丝带
  const lobe=Math.floor(r()*3),d=Math.pow(r(),.8)*nebPr(u)*1.15*(sp||1),a=lobe*2.0944+l*.07+.9*Math.sin(u*3.2+lobe*1.7)+2.4*d+gauss(r)*(.25+.5*d);
  return [a,d,lobe]}
function buildHolo(){ // 原创：由发光碎片组成的金色球体全息。每层 = 球面上的一圈碎片（层数与模型真实层数一致）
  if(holo)rig.remove(holo);
  holo=new THREE.Group();rig.add(holo);PER=NL>48?44:60;
  const r=rng(21),N=NL*PER,hybrid=PAT.indexOf('L')>=0,band=2.1/Math.max(1,NL-1);
  const P=new Float32Array(N*3),C=new Float32Array(N*3),FP=new Float32Array(N*6),FC=new Float32Array(N*6);
  const ox=new Float32Array(N),oz=new Float32Array(N),lay=new Uint8Array(N),typ=new Uint8Array(N),jit=new Float32Array(N),a0=new Float32Array(N),rad=new Float32Array(N),yo=new Float32Array(N),len=new Float32Array(N),rho=new Float32Array(N);
  for(let l=0;l<NL;l++){const rb=Rs(l,NL)*(1+.05*Math.sin(l*1.7))*((hybrid&&PAT[l]==='L')?.94:1);
    const u=l/Math.max(1,NL-1),neb=SHAPE==='nebula';
    for(let j=0;j<PER;j++){const i=l*PER+j,q=r();lay[i]=l;jit[i]=r();a0[i]=6.2832*(j+r()*.9)/PER+l*.11;
      if(neb){ox[i]=nebCx(u);oz[i]=nebCz(u);
        if(r()<.15){a0[i]=6.2832*r();rad[i]=Math.abs(gauss(r))*.6*nebPr(u)+.04}   // 核心附近的亮星团
        else{const [a,d]=nebPt(r,u,l);a0[i]=a;rad[i]=d+.03}
        rho[i]=Math.min(1.05,rad[i]/(1.1*nebPr(u)));yo[i]=gauss(r)*band*2.6*(.5+rho[i])}
      else{yo[i]=(r()-.5)*band*1.7;const y3=Yl(l,NL)+yo[i],Hm=Math.max(.02,rb);let h=Hm;
        if(r()<.08)h=Hm*(.96+.06*r());                            // 极少量点勾出球面轮廓
        else for(let t=0;t<14;t++){h=Hm*Math.sqrt(r());const r3=Math.sqrt(h*h+y3*y3);if(r()<Math.exp(-Math.pow(r3/.42,1.6))+.03)break}   // 按水平面积均匀取点，再按三维离球心的距离接受：越靠近球心越密，不会沿纵轴聚集
        rad[i]=h;rho[i]=Math.min(1.05,Math.sqrt(h*h+y3*y3)/1.0)}
      typ[i]=q<.66?0:q<.88?1:2;len[i]=.02+.11*r()*r()*(.45+rho[i])}}
  const ch=new Uint8Array(N);for(let i=0;i<N;i++)ch[i]=(Math.floor(((a0[i]%TAU)+TAU)%TAU/TAU*8)+8*Math.min(5,Math.floor(lay[i]*6/NL)))%48;
  const pg=new THREE.BufferGeometry();pg.setAttribute('position',new THREE.BufferAttribute(P,3));pg.setAttribute('color',new THREE.BufferAttribute(C,3));
  holo.add(new THREE.Points(pg,new THREE.PointsMaterial({size:.045,vertexColors:true,transparent:true,blending:ADD,depthWrite:false,sizeAttenuation:true})));
  const fg=new THREE.BufferGeometry();fg.setAttribute('position',new THREE.BufferAttribute(FP,3));fg.setAttribute('color',new THREE.BufferAttribute(FC,3));
  // （点上的小线段已去掉，只保留点；fg 仍保留以兼容更新逻辑，但不加入场景）
  // 层间连线：稀疏的斜线，相邻两层同时活跃时发亮
  const A=[],B=[];
  const inner=l=>{const a=[];for(let j=0;j<PER;j++){const i=l*PER+j;if(rho[i]<.62)a.push(i)}return a};
  for(let l=0;l<NL-1;l++){const u=inner(l),v=inner(l+1);if(!u.length||!v.length)continue;
    for(let k=0;k<u.length;k+=2){const i=u[k];let bi=v[0],bd=1e9;for(const w of v){let d=Math.abs(a0[w]-a0[i]);d=Math.min(d,6.2832-d);d+=Math.abs(rho[w]-rho[i])*2;if(d<bd){bd=d;bi=w}}A.push(i);B.push(bi)}}
  const S=A.length,LP=new Float32Array(S*6),LC=new Float32Array(S*6);
  const lg=new THREE.BufferGeometry();lg.setAttribute('position',new THREE.BufferAttribute(LP,3));lg.setAttribute('color',new THREE.BufferAttribute(LC,3));
  // （层间连线已去掉：lg 仍保留以兼容更新逻辑，但不加入场景）
  // 核心：几圈相互倾斜的金环 + 亮点
  const cores=[];[[.2,.4,0],[.3,1.2,.6],[.4,-.7,1.1]].forEach(([rr,rx,rz],k)=>{const m=new THREE.Mesh(new THREE.TorusGeometry(rr,.005,6,90),new THREE.MeshBasicMaterial({color:0xffd070,transparent:true,opacity:.7,blending:ADD,depthWrite:false}));
    m.rotation.set(rx,0,rz);cores.push(m)});
  const cdg=new THREE.BufferGeometry();cdg.setAttribute('position',new THREE.BufferAttribute(new Float32Array(3),3));
  const coreDot=new THREE.Points(cdg,new THREE.PointsMaterial({size:.22,color:0xffc860,transparent:true,opacity:.5,blending:ADD,depthWrite:false}));
  // 悬浮微粒
  const D=380,DP=new Float32Array(D*3),DC=new Float32Array(D*3);
  for(let i=0;i<D;i++){const a=r()*6.2832,b=Math.acos(2*r()-1),q=.7+.9*r();DP[i*3]=Math.sin(b)*Math.cos(a)*q;DP[i*3+1]=Math.cos(b)*q;DP[i*3+2]=Math.sin(b)*Math.sin(a)*q;const k=.15+.35*r();DC[i*3]=k*.2;DC[i*3+1]=k*.8;DC[i*3+2]=k*.95}
  const dg=new THREE.BufferGeometry();dg.setAttribute('position',new THREE.BufferAttribute(DP,3));dg.setAttribute('color',new THREE.BufferAttribute(DC,3));
  holo.add(new THREE.Points(dg,new THREE.PointsMaterial({size:.018,vertexColors:true,transparent:true,blending:ADD,depthWrite:false})));
  // 星云雾气：柔光精灵
  let hz=null;
  {const HN=1500,HP=new Float32Array(HN*3),HC=new Float32Array(HN*3),HB=new Float32Array(HN*3),hk=new Float32Array(HN),hw=new Float32Array(HN),hm2=new Float32Array(HN);
    // 雾气跟着点走：先统计点的密度，雾气只出现在点密集的地方（球体还要越靠近球面越淡，球面附近没有雾气）
    const fx=new Float32Array(N),fy=new Float32Array(N),fz=new Float32Array(N),CS=.28,cnt=new Map(),key=(x,y,z)=>Math.floor(x/CS)+','+Math.floor(y/CS)+','+Math.floor(z/CS);
    for(let i=0;i<N;i++){fx[i]=ox[i]+Math.cos(a0[i])*rad[i];fz[i]=oz[i]+Math.sin(a0[i])*rad[i];fy[i]=Yl(lay[i],NL)+yo[i];const k=key(fx[i],fy[i],fz[i]);cnt.set(k,(cnt.get(k)||0)+1)}
    const dens=(x,y,z)=>{let c=0;for(let a=-1;a<=1;a++)for(let b=-1;b<=1;b++)for(let d=-1;d<=1;d++)c+=cnt.get(key(x+a*CS,y+b*CS,z+d*CS))||0;return c};
    const ds=[];for(let i=0;i<N;i+=3)ds.push(dens(fx[i],fy[i],fz[i]));ds.sort((p,q)=>p-q);const ref=Math.max(4,ds[Math.floor(ds.length*.85)]||4);
    const PAL=[[.95,.25,.55],[.2,.75,.95],[.5,.3,.95],[.15,.9,.7]],neb=SHAPE==='nebula';
    for(let i=0;i<HN;i++){const j=Math.floor(r()*N),x=fx[j]+gauss(r)*.09,y=fy[j]+gauss(r)*.09,z=fz[j]+gauss(r)*.09;
      HP[i*3]=x;HP[i*3+1]=y;HP[i*3+2]=z;hk[i]=r();
      let w=clamp(dens(x,y,z)/ref,0,1);w=w*w*(3-2*w);if(!neb)w=Math.min(1,w*1.5)*Math.pow(Math.max(0,1-rho[j]/.95),.9);hw[i]=w;
      let c;if(neb){const lobe=Math.floor(r()*3);c=PAL[(lobe+(r()<.25?1:0))%4];hm2[i]=w*w}
      else c=[.15+.1*hk[i],.6+.2*hk[i],.9];
      HB[i*3]=c[0];HB[i*3+1]=c[1];HB[i*3+2]=c[2]}
    const hg=new THREE.BufferGeometry();hg.setAttribute('position',new THREE.BufferAttribute(HP,3));hg.setAttribute('color',new THREE.BufferAttribute(HC,3));
    const tex=hazeTex(),hm=new THREE.PointsMaterial({size:.62,vertexColors:true,transparent:true,blending:ADD,depthWrite:false,sizeAttenuation:true,opacity:.3});if(tex)hm.map=tex;
    holo.add(new THREE.Points(hg,hm));hz={HN,HP,HC,HB,hk,hw,hm2,hg,hm}}
  H={hz,ox,oz,P,C,FP,FC,LP,LC,A,B,lay,typ,jit,a0,rad,yo,len,rho,ch,S,N,pg,fg,lg,cores,coreDot,as:new Float32Array(NL)};_hdark=null;applyHoloTheme();
}
let rotY=.5,rotX=.16,vy=.002,drag=false,px0=0,py0=0;
// ---- 球体 = 电脑硬件占用 + 大脑皮层：占用越高，"思考的电信号"越频繁。信号像闪电一样沿球面掠过各个区域，带分叉和渐隐的尾迹
let kick=0;                                  // 每个 token 会让信号短暂变密
let HW={gpu:null,mem:null,gb:null,tot:null,model:null},uS=0;
async function pollHw(){if(!LIVE)return;try{const j=await api('/api/hw');HW={gpu:j.gpu,mem:j.mem_pct,gb:j.mem_used_gb,tot:j.mem_total_gb,model:j.model_gb}}catch(_){}}
setInterval(pollHw,1000);
function hwLevel(mean){
  if(LIVE&&(HW.gpu!=null||HW.mem!=null)){const g=HW.gpu!=null?HW.gpu:(busy?.7:0),m=HW.mem!=null?HW.mem*.8:0;return clamp(Math.max(g,m)+.15*mean,0,1)}
  return clamp(.1+1.2*mean,0,1)   // 演示模式或读不到硬件：用模型自己的活动代替
}
const CY=[.31,.85,.96],GD=[1,.68,.22],lerp3=(a,b,k)=>[a[0]+(b[0]-a[0])*k,a[1]+(b[1]-a[1])*k,a[2]+(b[2]-a[2])*k];
const LGOLD=[[.4,.22,.02],[.7,.4,.02],[.9,.5,0],[1,.62,.04]];   // 白色主题用的深琥珀色
function lgold(e,o){e=clamp(e,0,1)*3;const i=Math.min(2,e|0),f=e-i,a=LGOLD[i],b=LGOLD[i+1];o[0]=a[0]+(b[0]-a[0])*f;o[1]=a[1]+(b[1]-a[1])*f;o[2]=a[2]+(b[2]-a[2])*f}
const DGOLD=[[.3,.16,.02],[.75,.45,.06],[1,.68,.14],[1,.88,.5]];   // 暗金色，不会变成白色
function dgold(e,o,k){e=clamp(e,0,1)*3;const i=Math.min(2,e|0),f=e-i,a=DGOLD[i],b=DGOLD[i+1];o[0]=(a[0]+(b[0]-a[0])*f)*k;o[1]=(a[1]+(b[1]-a[1])*f)*k;o[2]=(a[2]+(b[2]-a[2])*f)*k}
const TAU=6.2832;
// —— 闪电式信号：球面上一条带抖动的路径，亮头沿路径前进，身后留下渐隐的尾迹，中途可能分叉
const BOLTS=[],BK=28;
const v3={cross:(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]],norm:a=>{const l=Math.hypot(a[0],a[1],a[2])||1;return[a[0]/l,a[1]/l,a[2]/l]},
  randUnit:()=>{const z=Math.random()*2-1,a=Math.random()*TAU,r=Math.sqrt(1-z*z);return[r*Math.cos(a),z,r*Math.sin(a)]}};
function randTangent(a){let t=v3.randUnit();const d=t[0]*a[0]+t[1]*a[1]+t[2]*a[2];return v3.norm([t[0]-a[0]*d,t[1]-a[1]*d,t[2]-a[2]*d])}
function makeBolt(t0,a,len,dur,tail,amp){
  const t1=randTangent(a),c=v3.cross(a,t1),pts=new Float32Array(BK*3),ph=[Math.random()*TAU,Math.random()*TAU],jw=.12+.2*Math.random();let walk=0;
  for(let k=0;k<BK;k++){const u=k/(BK-1),th=len*u;walk+=(Math.random()-.5)*.09;const J=jw*(.6*Math.sin(6*u+ph[0])+.4*Math.sin(15*u+ph[1]))+walk;
    const p=v3.norm([a[0]*Math.cos(th)+t1[0]*Math.sin(th)+c[0]*J,a[1]*Math.cos(th)+t1[1]*Math.sin(th)+c[1]*J,a[2]*Math.cos(th)+t1[2]*Math.sin(th)+c[2]*J]);
    pts[k*3]=p[0];pts[k*3+1]=p[1];pts[k*3+2]=p[2]}
  const b={t0,dur,tail,amp,pts};BOLTS.push(b);return b}
function spawnEvent(t,as,tot){
  let a;if(tot>.15&&Math.random()<.7){let r=Math.random()*tot,l=0;for(;l<NL-1;l++){r-=as[l];if(r<=0)break}   // 起点常落在当前活跃的那一层的高度，但沿球面朝随机方向掠过
    const lat=Math.asin(clamp(Yl(l,NL)/1.14,-.97,.97)),lon=Math.random()*TAU;a=[Math.cos(lat)*Math.cos(lon),Math.sin(lat),Math.cos(lat)*Math.sin(lon)]}
  else a=v3.randUnit();
  const dur=1500+1500*Math.random(),b=makeBolt(t,a,.9+1.4*Math.random(),dur,1000+600*Math.random(),.75+.25*Math.random());
  for(let n=Math.random()<.65?(Math.random()<.4?2:1):0;n>0;n--){const k0=7+Math.floor(Math.random()*14),p=[b.pts[k0*3],b.pts[k0*3+1],b.pts[k0*3+2]];
    makeBolt(t+dur*k0/(BK-1),p,.4+.7*Math.random(),dur*.55,b.tail*.8,b.amp*.7)}
  while(BOLTS.length>14)BOLTS.shift()}
function updateHolo(t){
  const {P,C,FP,FC,LP,LC,A,B,lay,typ,jit,a0,rad,yo,len,rho,S,N,ox,oz}=H,neb=SHAPE==="nebula",fl=.92+.08*Math.sin(t*.012)*Math.sin(t*.0047),sec=t*.0005,col=[0,0,0],dt=Math.min(100,Math.max(0,t-(H.t0||t)));H.t0=t;
  const cs=new Float32Array(NL),sn=new Float32Array(NL),mL=new Float32Array(NL);let mean=0,tot=0;
  for(let l=0;l<NL;l++){const th=neb?sec*(.7+.45*Math.sin(l*.3)):(l%2?1:-1)*sec*(.1+.09*((l*7)%5)/4);cs[l]=Math.cos(th);sn[l]=Math.sin(th);mean+=act[l]}mean/=NL;
  uS+=(hwLevel(mean)-uS)*(1-Math.pow(.5,dt/700));            // 平滑：占用变化时球体缓缓过渡
  H.run=(H.run||0)+(((busy||trainRunning)?1:0)-(H.run||0))*(1-Math.pow(.5,dt/500));const run=H.run,uE=Math.max(uS,.4)*run;   // 只有模型真正在运行（回复或训练）时才有放电/变色，静止时完全不出现
  kick*=Math.pow(.5,dt/500);
  const as=H.as;for(let l=0;l<NL;l++){as[l]+=(act[l]-as[l])*(1-Math.pow(.5,dt/(act[l]>as[l]?180:650)));tot+=as[l]}   // 层活动取平滑值
  if(run>.05&&Math.random()<dt/1000*(.15+1.7*uE+1.2*kick))spawnEvent(t,as,tot);                                              // 占用越高，信号越频繁
  // 收集所有"亮着"的路径点（位置 + 亮度）
  const ax=[],ay=[],az=[],aw=[];
  for(let q=BOLTS.length-1;q>=0;q--){const b=BOLTS[q],age=t-b.t0;if(age>b.dur+b.tail){BOLTS.splice(q,1);continue}if(age<0)continue;
    for(let k=0;k<BK;k++){const tp=age-b.dur*k/(BK-1);if(tp<0||tp>b.tail)continue;
      const f=1-tp/b.tail,w=b.amp*f*f*(tp<160?1.4-tp/400:1);if(w>.03){ax.push(b.pts[k*3]);ay.push(b.pts[k*3+1]);az.push(b.pts[k*3+2]);aw.push(w)}}}
  const na=aw.length,gate=sstep(0,.45,uE),sig2=neb?.05:.075;
  const bp=Math.sin(t*.0016*(1+.7*uE)),br=1+(.1+.3*uE)*bp;                       // 核心亮度起伏（不改变球体大小）
  for(let i=0;i<N;i++){const l=lay[i],ca=Math.cos(a0[i]),sa=Math.sin(a0[i]),ux=ca*cs[l]-sa*sn[l],uz=sa*cs[l]+ca*sn[l],R=rad[i],y=Yl(l,NL)+yo[i];
    const x=ox[i]+ux*R,z=oz[i]+uz*R;P[i*3]=x;P[i*3+1]=y;P[i*3+2]=z;
    const il=1/(Math.sqrt(x*x+y*y+z*z)+1e-6),dx0=x*il,dy0=y*il,dz0=z*il;let bolt=0;
    for(let k=0;k<na;k++){const ex=dx0-ax[k],ey=dy0-ay[k],ez=dz0-az[k],d2=ex*ex+ey*ey+ez*ez;if(d2<sig2){const q=1-d2/sig2;bolt+=aw[k]*q*q}}
    bolt=Math.min(1,bolt)*(neb?.45+.55*Math.min(1,rho[i]/.9):.8+.2*Math.min(1,rho[i]/.9));                       // 越靠外层（皮层）越亮，球心的碎片只微微泛光
    const amb=.18*clamp(Math.sin(x*3.1+y*1.7+t*.00031+1.3)*.5+Math.sin(y*2.9-z*2.3+t*.00023+4.1)*.5+.15,0,1);   // 极淡的、缓慢漂移的背景微光
    const m=clamp(gate*(Math.pow(bolt,1.15)*(neb?1.5:2.4)+amb*.7)+as[l]*(.1+.2*uE)*.5,0,1);mL[l]+=m/PER;
    const L=len[i]*(1+1.2*m)*(neb?.7:1),ty=typ[i],dx=ty===0?-uz:ty===2?ux:0,dy=ty===1?1:0,dz=ty===0?ux:ty===2?uz:0;
    FP[i*6]=x;FP[i*6+1]=y;FP[i*6+2]=z;FP[i*6+3]=x+dx*L;FP[i*6+4]=y+dy*L;FP[i*6+5]=z+dz*L;
    const bb=(.2+.5*jit[i])*(1-.5*gate*(1-sstep(0,.5,m))),mm=sstep(.1,.45,m),sh=Math.min(1,bolt*1.1)*(1-gate);let r,gg,bl;   // sh：休眠时偶尔掠过的青蓝色微光
    if(isDark){dgold(m,col,fl*(.45+.75*m));const cb=bb*fl*(1+1.6*sh);r=(.1*cb)*(1-mm)+col[0]*mm;gg=(.75*cb)*(1-mm)+col[1]*mm;bl=(.95*cb)*(1-mm)+col[2]*mm}
    else{lgold(m,col);const nb=.55+.9*bb,wv=clamp(sh*1.4,0,1);                    // 白色主题：休眠深蓝，活跃深琥珀，微光偏亮青蓝
      r=((.02*nb)*(1-mm)+col[0]*mm)*(1-wv)+.05*wv;gg=((.14*nb)*(1-mm)+col[1]*mm)*(1-wv)+.62*wv;bl=((.22*nb)*(1-mm)+col[2]*mm)*(1-wv)+.85*wv}
    C[i*3]=r;C[i*3+1]=gg;C[i*3+2]=bl;FC[i*6]=r;FC[i*6+1]=gg;FC[i*6+2]=bl;FC[i*6+3]=r*.35;FC[i*6+4]=gg*.35;FC[i*6+5]=bl*.35}
  for(let k=0;k<S;k++){const a=A[k],b=B[k],s=Math.sqrt(mL[lay[a]]*mL[lay[b]]);
    LP[k*6]=P[a*3];LP[k*6+1]=P[a*3+1];LP[k*6+2]=P[a*3+2];LP[k*6+3]=P[b*3];LP[k*6+4]=P[b*3+1];LP[k*6+5]=P[b*3+2];
    const mm=sstep(.05,.4,s),base=.16;let r,g,bl;
    if(isDark){dgold(.3+.7*s,col,fl*(.3+.7*s));r=(.05*base*(1-mm)+col[0]*mm)*fl;g=(.5*base*(1-mm)+col[1]*mm)*fl;bl=(.65*base*(1-mm)+col[2]*mm)*fl}
    else{lgold(.3+.7*s,col);r=.02*(1-mm)+col[0]*mm;g=.2*(1-mm)+col[1]*mm;bl=.3*(1-mm)+col[2]*mm}
    LC[k*6]=r;LC[k*6+1]=g;LC[k*6+2]=bl;LC[k*6+3]=r;LC[k*6+4]=g;LC[k*6+5]=bl}
  for(const g of [H.pg,H.fg,H.lg]){g.attributes.position.needsUpdate=true;g.attributes.color.needsUpdate=true}
  if(H.hz){const {HN,HP,HC,HB,hk,hw,hm2,hg}=H.hz;
    for(let i=0;i<HN;i++){const x=HP[i*3],y=HP[i*3+1],z=HP[i*3+2],il=1/(Math.sqrt(x*x+y*y+z*z)+1e-6),dx0=x*il,dy0=y*il,dz0=z*il,w=hw[i];let b=0;
      for(let k=0;k<na;k++){const ex=dx0-ax[k],ey=dy0-ay[k],ez=dz0-az[k],d2=ex*ex+ey*ey+ez*ez;if(d2<sig2*2){const q=1-d2/(sig2*2);b+=aw[k]*q*q}}
      b=Math.min(1,b)*gate;const br0=(.16+.1*Math.sin(t*.0004+hk[i]*40)+.05*uE)*w*(neb?1.6:3.4),gl=(neb?.45:.75)*b*w;
      const wm=hm2[i]*gate;   // 核心偏暖金只在模型工作（占用高）时出现，休眠时保持冷色
      for(let c=0;c<3;c++){const base=HB[i*3+c]*(1-wm)+[1,.72,.35][c]*wm;
        if(isDark)HC[i*3+c]=base*br0*(1-b)+[1,.62,.2][c]*gl;
        else HC[i*3+c]=(1-(1-base)*.7*w*(1-b*.6))*(1-b*w)+[.7,.4,.05][c]*b*w}}
    hg.attributes.color.needsUpdate=true;H.hz.hm.opacity=neb?(isDark?.34:.15):(isDark?.5:.24)}
  // 核心：休眠时青蓝，占用高时偏暗金
  const kk=sstep(.08,.75,uE),tc=isDark?lerp3(CY,GD,kk):lerp3([.02,.16,.25],[.8,.48,.04],kk);
  H.cores.forEach((m,i)=>{m.rotation.y=sec*(.8+.4*i)*(i%2?-1:1);m.material.color.setRGB(tc[0],tc[1],tc[2]);m.material.opacity=.3+.55*Math.min(1,uE*1.4)});
  H.coreDot.material.color.setRGB(tc[0],tc[1],tc[2]);H.coreDot.material.opacity=.25+.55*Math.min(1,uE*1.5)*br;H.coreDot.material.size=.16+.24*Math.min(1,uE*1.5);
}
// 刻度尺与层号（2D 叠加层）：随模型真实层数变化，每 5 层一个标签
const _v=THREE.Vector3?new THREE.Vector3():null;
function drawHud(){
  const w=hud.clientWidth,h=hud.clientHeight,d=Math.min(devicePixelRatio||1,2);
  if(hud.width!==w*d||hud.height!==h*d){hud.width=w*d;hud.height=h*d}
  hx.setTransform(d,0,0,d,0,0);hx.clearRect(0,0,w,h);if(!_v||!_v.project)return;
  rig.updateMatrixWorld(true);hx.font='10px '+"'IBM Plex Mono',ui-monospace,monospace";hx.textBaseline='middle';
  const step=NL>48?8:5;let xr=0;
  for(let l=0;l<NL;l++){_v.set(0,Yl(l,NL),0).applyMatrix4(rig.matrixWorld).project(camera);const y=(1-_v.y)/2*h;
    const major=l%step===0||l===NL-1,hot=act[l]>.35;xr=w*.17;
    hx.strokeStyle=hot?(isDark?'#ffd36a':'#7a4a00'):(isDark?'rgba(111,200,226,.55)':'rgba(6,34,46,.6)');hx.lineWidth=1;hx.beginPath();hx.moveTo(xr-(major?12:6),y);hx.lineTo(xr,y);hx.stroke();
    if(major){hx.fillStyle=hot?(isDark?'#ffd36a':'#7a4a00'):(isDark?'#8fd6ea':'#06222e');hx.textAlign='right';hx.fillText('L'+l+(PAT[l]==='A'&&PAT.indexOf('L')>=0?' ◆':''),xr-16,y)}}
  const hwt=(LIVE&&(HW.gpu!=null||HW.mem!=null))?`显卡 ${HW.gpu!=null?Math.round(HW.gpu*100)+'%':'—'} · 内存 ${HW.gb!=null?HW.gb+'/'+HW.tot+' GB':'—'}${HW.model?' · 模型 '+HW.model+' GB':''}`:(LIVE?'硬件占用：读取中…':'硬件占用：模拟');
  hx.textAlign='left';hx.fillStyle=isDark?'#8fd6ea':'#06222e';hx.fillText(tr('球体 = '+hwt),14,h-36);
}
function resizeGL(){const s=cv.parentElement,w=s.clientWidth,h=s.clientHeight;if(!w||!h)return;renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix()}
new ResizeObserver(resizeGL).observe(cv.parentElement);
cv.addEventListener('pointerdown',e=>{drag=true;px0=e.clientX;py0=e.clientY;cv.setPointerCapture(e.pointerId);cv.style.cursor='grabbing'});
cv.addEventListener('pointermove',e=>{if(!drag)return;rotY+=(e.clientX-px0)*.008;rotX=clamp(rotX+(e.clientY-py0)*.005,-.9,.9);px0=e.clientX;py0=e.clientY;vy=0});
cv.addEventListener('pointerup',()=>{drag=false;cv.style.cursor='grab';vy=.002});
buildHolo();
function setShape(v){SHAPE=v;try{localStorage.setItem('neurolume.shape',v)}catch(_){}const b=$('btnShape');if(b)b.textContent=v==='nebula'?'形态：星云':'形态：球体';buildHolo()}
{const b=$('btnShape');if(b){b.textContent=SHAPE==='nebula'?'形态：星云':'形态：球体';b.addEventListener('click',()=>setShape(SHAPE==='nebula'?'sphere':'nebula'))}}

