import cv2, sys, json, glob, os, numpy as np
S=sys.argv[1]; sys.path.insert(0,S+'/work')
from detect import detect
from concurrent.futures import ProcessPoolExecutor
def run(f):
    n=os.path.basename(f).replace('VID-20260927-','').replace('.mp4','')
    out=S+f'/tracks/{n}.json'
    if os.path.exists(out): return n
    cap=cv2.VideoCapture(f); fps=cap.get(5)
    dual = n in ('WA0032','WA0034')
    res=[]
    while True:
        ok,fr=cap.read()
        if not ok: break
        if dual: fr=fr[:, :640]
        res.append(detect(fr))
    json.dump(dict(fps=fps,shape=list(fr.shape[:2]) if False else None,dets=res),open(out,'w'))
    return n
fs=[f for f in sorted(glob.glob(S+'/raw/*/*.mp4')) if not any(k in f for k in ('0109','0110','0111','164536'))]
with ProcessPoolExecutor(4) as ex:
    for n in ex.map(run,fs): print(n,flush=True)
