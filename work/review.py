import os, sys, cv2, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from lib import S
from edl import timeline
# usage: review.py [shotdir] [ncols] [ids...]
shotdir = sys.argv[1] if len(sys.argv) > 1 else S + '/shots'
ncol = int(sys.argv[2]) if len(sys.argv) > 2 else 6
ids = sys.argv[3:]
tl, T = timeline()
rows = []
for s, t, n in tl:
    if ids and s['id'] not in ids:
        continue
    cap = cv2.VideoCapture(f"{shotdir}/{s['id']}.mp4")
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = np.linspace(0, N - 1, ncol).astype(int)
    ims = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i); ok, f = cap.read()
        if not ok:
            f = np.zeros((1920, 1080, 3), np.uint8)
        f = cv2.resize(f, (162, 288), interpolation=cv2.INTER_AREA)
        lab = f"{s['id']} {t + i / 24:.1f}"
        cv2.putText(f, lab, (3, 14), 0, 0.42, (0, 0, 0), 3); cv2.putText(f, lab, (3, 14), 0, 0.42, (0, 255, 255), 1)
        ims.append(f)
    rows.append(np.hstack(ims))
half = (len(rows) + 1) // 2
for k, part in enumerate([rows[:half], rows[half:]]):
    if part:
        cv2.imwrite(f'{S}/an/review_{k}.jpg', np.vstack(part), [cv2.IMWRITE_JPEG_QUALITY, 85])
