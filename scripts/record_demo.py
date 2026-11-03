# Records docs/media/demo.gif from the online demo (a real recorded session) with headless Chromium.
# Usage (needs three.min.js r128 at rec/three.min.js and the web UI copied to site/neurolume): python scripts/record_demo.py en
import sys, threading, http.server, functools, time, os, shutil, json
lang=sys.argv[1] if len(sys.argv)>1 else 'en'
THREE=open('rec/three.min.js','rb').read()
class Q(http.server.SimpleHTTPRequestHandler):
    def log_message(self,*a): pass
srv=http.server.ThreadingHTTPServer(("127.0.0.1",0),functools.partial(Q, directory="site")); threading.Thread(target=srv.serve_forever,daemon=True).start()
base=f"http://127.0.0.1:{srv.server_address[1]}/neurolume/"
from playwright.sync_api import sync_playwright
OUT=f'rec/frames_{lang}'; shutil.rmtree(OUT,ignore_errors=True); os.makedirs(OUT)
W,H=1440,900; frames=[]; errs=[]
with sync_playwright() as p:
    b=p.chromium.launch()
    ctx=b.new_context(viewport={"width":W,"height":H},color_scheme="dark"); pg=ctx.new_page(); pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.add_init_script(f"try{{localStorage.setItem('neurolume.lang','{lang}');localStorage.setItem('neurolume.shape','nebula')}}catch(e){{}}")
    pg.route("**/three.min.js", lambda r: r.fulfill(body=THREE, content_type="application/javascript"))
    pg.route("**/fonts.googleapis.com/**", lambda r: r.abort())
    def shot(dur, label=''):
        f=f"{OUT}/{len(frames):04d}.jpg"; pg.screenshot(path=f, type='jpeg', quality=88); frames.append((f,dur,label))
    pg.goto(base); pg.wait_for_function("typeof DEMO!=='undefined'&&DEMO&&document.getElementById('msgs').innerText.length>50",timeout=120000); pg.wait_for_timeout(1500)
    for i in range(10): shot(0.12,'idle'); pg.wait_for_timeout(120)
    # generation: slow the replay so every token can be captured, play back at the recorded speed later
    pg.evaluate("DEMO.__tps=DEMO.tps;DEMO.tps=0.5")
    pg.click("#send"); pg.wait_for_function("busy===true",timeout=120000); pg.evaluate("document.getElementById('scanTps').textContent=DEMO.__tps+' tok/s'")
    n_prev=-1
    while pg.evaluate("busy"):
        n=pg.evaluate("SRC?SRC.frames.length:0")
        if n!=n_prev: shot(1/21.6*1.0,'gen'); n_prev=n
        pg.wait_for_timeout(150)
    pg.evaluate("DEMO.tps=DEMO.__tps"); pg.wait_for_timeout(600); shot(1.6,'done')
    # scroll to the answer replay
    top=pg.evaluate("document.querySelector('#tl').offsetTop-330")
    for k in range(1,9): pg.evaluate(f"document.querySelector('.mon').scrollTop={top}*{k}/8"); pg.wait_for_timeout(80); shot(0.07,'scroll')
    pg.wait_for_timeout(500); shot(1.3,'replay')
    # hover the most hesitant token
    k=pg.evaluate("(()=>{const R=TLM.M.rows;let best=-1,bp=2;R.forEach((r,j)=>{if(r&&r.p<bp&&j>3){bp=r.p;best=j}});return best})()")
    bx=pg.locator("#tlc").bounding_box(); n=pg.evaluate("TLM.idx.length")
    for j in [k-12,k-8,k-4,k]:
        x=bx["x"]+(j+.5)/n*bx["width"]; pg.mouse.move(x,bx["y"]+45); pg.wait_for_timeout(250); shot(0.35 if j!=k else 2.6,'hover')
    pg.mouse.click(x,bx["y"]+45); pg.wait_for_timeout(700); shot(2.0,'pinned')
    # logit lens: step through tokens
    pg.mouse.move(5,5); pg.click("#modes button[data-m=lens]"); pg.wait_for_function("document.querySelectorAll('#lensv tr').length>10",timeout=60000)
    pg.evaluate("tlPick(16)"); pg.wait_for_timeout(500)
    pg.evaluate("(()=>{const v=document.getElementById('lensv');v.scrollTop=1e5;const m=document.querySelector('.mon');m.scrollTop=v.offsetTop+v.offsetHeight-560})()"); pg.wait_for_timeout(600); shot(1.6,'lens')
    for t in range(17,42): pg.evaluate(f"tlPick({t});document.getElementById('lensv').scrollTop=1e5"); pg.wait_for_timeout(200); shot(0.22,'lens')
    shot(1.4,'lens')
    # training view: a real adapter's ΔW
    pg.click("#modes button[data-m=train]"); pg.wait_for_function("deltaReal===true",timeout=60000); pg.wait_for_timeout(700)
    pg.evaluate("document.querySelector('.mon').scrollTop=document.querySelector('.net').offsetTop-12"); pg.wait_for_timeout(500); shot(1.0,'train')
    sw=pg.evaluate("(()=>{const s=document.getElementById('scroller');return s.scrollWidth-s.clientWidth})()")
    for k in range(1,11): pg.evaluate(f"document.getElementById('scroller').scrollLeft={sw}*{k}/10"); pg.wait_for_timeout(60); shot(0.08,'train')
    pg.wait_for_timeout(500); shot(3.2,'train')
    pg.evaluate("document.querySelector('.mon').scrollTop=0"); pg.click("#modes button[data-m=infer]"); pg.wait_for_timeout(900); shot(1.2,'end')
    b.close()
json.dump(frames,open(f'{OUT}/list.json','w'))
print(len(frames),'frames', round(sum(d for _,d,_ in frames),1),'s', {l:sum(1 for *_,x in frames if x==l) for l in dict.fromkeys(x for *_,x in frames)}); print("errors:",errs)
