"""Objective original-vs-enhanced motion metrics per shot.
For each shot: global image motion between consecutive frames (phase correlation at 270x480), then
  step_px   : 95th percentile of the per-frame displacement (px at 1080-wide scale) -> size of each visual 'jump'
  judder    : RMS of the frame-to-frame change of velocity, in px/s (uneven cadence, stutter, hand shake)
  dups      : frames identical to the previous one (frozen moments)
usage: compare.py old.mp4 old_fps_edl_module new.mp4 new_edl_module
"""
import os, sys, importlib, json
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(__file__))


def motion(path, t0, t1, fps):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t0 * fps)))
    n = int(round((t1 - t0) * fps))
    prev = None; win = None; v = []; dups = 0; diffs = []; lum = []
    for k in range(n):
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(f, (270, 480), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
        lum.append(float(g.mean()))
        h = g - cv2.GaussianBlur(g, (0, 0), 6)
        if win is None:
            win = cv2.createHanningWindow(h.shape[::-1], cv2.CV_32F)
        if prev is not None:
            (dx, dy), r = cv2.phaseCorrelate(prev[1], h, win)
            v.append((dx * 4, dy * 4))
            d = np.abs(g - prev[0]).mean()
            diffs.append(d)
        prev = (g, h)
    v = np.array(v)
    if len(v) < 4:
        return None
    sp = np.hypot(v[:, 0], v[:, 1])
    dups = int((np.array(diffs) < 0.15).sum())
    acc = np.diff(v * fps, axis=0)  # px/s change per frame
    judder = float(np.sqrt((acc ** 2).sum(1).mean()))
    L = np.array(lum)
    from scipy.ndimage import uniform_filter1d
    flick = float(np.sqrt(((L - uniform_filter1d(L, max(3, int(fps * 0.25)))) ** 2).mean()))
    return dict(step_px=round(float(np.percentile(sp, 95)), 1), judder=round(judder, 0), dups=dups, flicker=round(flick, 2))


if __name__ == '__main__':
    old, old_edl, new, new_edl = sys.argv[1:5]
    E1 = importlib.import_module(old_edl); E2 = importlib.import_module(new_edl)
    t1, _ = E1.timeline(); t2, _ = E2.timeline()
    f1 = getattr(E1, 'FPS_OUT', 24); f2 = getattr(E2, 'FPS_OUT', 24)
    m1 = {s['id']: (t, t + n / f1) for s, t, n in t1}
    m2 = {s['id']: (t, t + n / f2) for s, t, n in t2}
    rows = []
    print(f"{'shot':5s} | {'antes: salto px':>15s} {'judder':>8s} {'dup':>4s} {'flick':>6s} | {'depois: salto px':>16s} {'judder':>8s} {'dup':>4s} {'flick':>6s}")
    for sid in m2:
        b = motion(new, *m2[sid], f2)
        a = motion(old, *m1[sid], f1) if sid in m1 else None
        rows.append(dict(shot=sid, before=a, after=b))
        fa = f"{a['step_px']:15.1f} {a['judder']:8.0f} {a['dups']:4d} {a['flicker']:6.2f}" if a else f"{'(novo)':>15s} {'':8s} {'':4s} {'':6s}"
        print(f"{sid:5s} | {fa} | {b['step_px']:16.1f} {b['judder']:8.0f} {b['dups']:4d} {b['flicker']:6.2f}")
    json.dump(rows, open(os.environ.get('S', '.') + '/an/compare.json', 'w'), indent=1)
