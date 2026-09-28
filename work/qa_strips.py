"""QA: for selected shots, find the moment of largest motion and show consecutive frames (full frame, downscaled)
plus the per-frame motion trace, to judge stutter / ghosting / snapping by eye.
usage: qa_strips.py film.mp4 fps SHOT [SHOT...]"""
import os, sys, cv2, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from lib import S
from edl import timeline
film, fps = sys.argv[1], int(sys.argv[2])
ids = sys.argv[3:]
tl, T = timeline(fps)
rows = []
for s, t, n in tl:
    if s['id'] not in ids:
        continue
    cap = cv2.VideoCapture(film); cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
    fr = []
    for k in range(n):
        ok, f = cap.read()
        if not ok:
            break
        fr.append(f)
    g = [cv2.cvtColor(cv2.resize(f, (135, 240)), cv2.COLOR_BGR2GRAY).astype(np.float32) for f in fr]
    mot = np.array([0] + [np.abs(g[i] - g[i - 1]).mean() for i in range(1, len(g))])
    k0 = int(np.clip(np.argmax(np.convolve(mot, np.ones(6), 'same')) - 3, 0, len(fr) - 8))
    tiles = [cv2.resize(fr[k], (216, 384), interpolation=cv2.INTER_AREA) for k in range(k0, k0 + 8)]
    for i, tt in enumerate(tiles):
        cv2.putText(tt, f"{s['id']} +{k0 + i}", (4, 16), 0, 0.45, (0, 0, 0), 3); cv2.putText(tt, f"{s['id']} +{k0 + i}", (4, 16), 0, 0.45, (0, 255, 255), 1)
    row = np.hstack(tiles)
    # motion trace under the row: per-frame change (should be smooth, no spikes/zeros)
    tr = np.zeros((60, row.shape[1], 3), np.uint8)
    m = mot / (mot.max() + 1e-6)
    pts = np.stack([np.linspace(0, row.shape[1] - 1, len(m)), 55 - m * 50], 1).astype(np.int32)
    cv2.polylines(tr, [pts], False, (0, 255, 0), 1)
    x0 = int(k0 / len(m) * row.shape[1]); x1 = int((k0 + 8) / len(m) * row.shape[1])
    cv2.rectangle(tr, (x0, 2), (x1, 58), (0, 200, 255), 1)
    rows.append(np.vstack([row, tr]))
cv2.imwrite(S + '/an/qa_strips.jpg', np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
print('ok', len(rows))
