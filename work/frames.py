import sys, os, cv2, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from lib import *
# usage: frames.py out.jpg name:t[:pane] ...
out = sys.argv[1]
ims = []
for spec in sys.argv[2:]:
    p = spec.split(':')
    name, t = p[0], float(p[1]); pn = p[2] if len(p) > 2 else None
    fps, n, w, h = probe(name)
    cap = cv2.VideoCapture(src_path(name))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps))); ok, f = cap.read()
    f = pane(f, pn)
    f = cv2.resize(f, (int(f.shape[1] * 400 / f.shape[0]), 400))
    cv2.putText(f, spec, (5, 22), 0, 0.6, (0, 0, 0), 3); cv2.putText(f, spec, (5, 22), 0, 0.6, (0, 255, 255), 1)
    ims.append(f)
W = sum(i.shape[1] for i in ims)
row = np.hstack(ims)
if W > 2400:
    half = len(ims) // 2
    a = np.hstack(ims[:half]); b = np.hstack(ims[half:])
    wmax = max(a.shape[1], b.shape[1])
    a = np.pad(a, ((0, 0), (0, wmax - a.shape[1]), (0, 0))); b = np.pad(b, ((0, 0), (0, wmax - b.shape[1]), (0, 0)))
    row = np.vstack([a, b])
cv2.imwrite(out, row, [cv2.IMWRITE_JPEG_QUALITY, 88])
