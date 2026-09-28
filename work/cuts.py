"""Cut review: for every cut, the 2 frames before and 2 after (from the assembled film)."""
import os, sys, cv2, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from lib import S
from edl import timeline
film = sys.argv[1] if len(sys.argv) > 1 else S + '/out/preview.mp4'
tl, T = timeline()
cap = cv2.VideoCapture(film)
frames = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(cv2.resize(f, (135, 240), interpolation=cv2.INTER_AREA))
cuts = []
acc = 0
for s, t, n in tl[:-1]:
    acc += n
    cuts.append((acc, s['id']))
tiles = []
for c, sid in cuts:
    row = []
    for k in (c - 6, c - 1, c, c + 5):
        f = frames[min(max(k, 0), len(frames) - 1)].copy()
        if k == c:
            cv2.rectangle(f, (0, 0), (134, 239), (0, 255, 255), 2)
        row.append(f)
    r = np.hstack(row)
    cv2.putText(r, f'{sid}| {c / 24:.2f}s', (3, 14), 0, 0.45, (0, 0, 0), 3); cv2.putText(r, f'{sid}| {c / 24:.2f}s', (3, 14), 0, 0.45, (0, 255, 255), 1)
    tiles.append(r)
# 3 cut-groups per row
rows = []
for i in range(0, len(tiles), 3):
    grp = tiles[i:i + 3]
    while len(grp) < 3:
        grp.append(np.zeros_like(tiles[0]))
    rows.append(np.hstack([np.pad(g, ((0, 0), (0, 8), (0, 0))) for g in grp]))
cv2.imwrite(S + '/an/cuts.jpg', np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
print(len(frames), 'frames', len(cuts), 'cuts')
