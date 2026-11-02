// ======== 主循环 ========
const reduce=matchMedia('(prefers-reduced-motion: reduce)').matches;
let lastCheck=0;
function frame(t){
  if(t-lastCheck>800){lastCheck=t;isDark=(getComputedStyle(document.body).backgroundColor.match(/\d+/g)||[0]).map(Number).reduce((a,b)=>a+b,0)<384}
  applyHoloTheme();
  const dt=Math.min(100,Math.max(0,t-(frame.t0||t)));frame.t0=t;const dk=Math.pow(.5,dt/380);  // 半衰期约 0.4 秒，和帧率无关
  for(let l=0;l<NL;l++)act[l]*=dk;
  if(PIN&&!busy&&!trainRunning&&PIN.length===NL)for(let l=0;l<NL;l++)act[l]=Math.max(act[l],PIN[l]);   // 回看某个字时，层图保持在那一刻
  if(trainRunning&&glow.length===NL)for(let l=0;l<NL;l++)act[l]=Math.max(act[l],glow[l]*(.5+.35*Math.sin(t*.002+l*.6)));
  if(!drag&&!reduce)rotY+=vy;rig.rotation.y=rotY;rig.rotation.x=rotX;
  for(const step of [()=>updateHolo(t),()=>renderer.render(scene,camera),drawHud,()=>drawNet(t,isDark)]){
    try{step()}catch(e){reportDrawError(e)}}
  requestAnimationFrame(frame);
}
let _lastDrawErr='',_drawErrs=0;
function reportDrawError(e){ // 某一帧画错了不要让整个动画停掉；结构/权重长度不一致时自动重建
  const m=String(e&&e.message||e);if(m!==_lastDrawErr){_lastDrawErr=m;console.error('draw error:',e)}
  if(++_drawErrs%30===1){try{if(!W||W.length!==NL)synth();if(!H||!H.N||H.N!==NL*PER)buildHolo();buildNet()}catch(_){}}}
setWeb(WEB);boot();resizeGL();requestAnimationFrame(frame);initLang();
