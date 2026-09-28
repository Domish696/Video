"""Temporal forensics per source clip: duplicates, jumps, camera jitter, timestamp irregularity."""
import os, sys, json, glob, subprocess
import numpy as np, cv2
from scipy.ndimage import gaussian_filter1d
sys.path.insert(0, os.path.dirname(__file__))
from lib import S, src_path, probe


def pts(name):
    out = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'packet=pts_time',
                                   '-of', 'csv=p=0', src_path(name)]).decode().split()
    t = np.sort(np.array([float(x) for x in out if x not in ('', 'N/A')]))
    return t


def analyse(name, pane=None):
    cap = cv2.VideoCapture(src_path(name))
    fps = cap.get(cv2.CAP_PROP_FPS)
    prev = None; diffs = []; shifts = []; win = None
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if pane == 'L':
            f = f[:, :640]
        elif pane == 'R':
            f = f[:, 640:1280]
        g = cv2.cvtColor(cv2.resize(f, (f.shape[1] // 2, f.shape[0] // 2), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
        hpf = g - cv2.GaussianBlur(g, (0, 0), 8)
        if win is None:
            win = cv2.createHanningWindow(hpf.shape[::-1], cv2.CV_32F)
        if prev is not None:
            diffs.append(float(np.abs(g - prev[0]).mean()))
            (dx, dy), r = cv2.phaseCorrelate(prev[1], hpf, win)
            shifts.append((dx * 2, dy * 2, r))
        prev = (g, hpf)
    d = np.array(diffs); sh = np.array(shifts)
    # duplicates: near-zero change while the neighbourhood moves
    med = np.array([np.median(d[max(0, i - 4):i + 5]) for i in range(len(d))])
    dup = np.where((d < 0.35) & (med > 1.2))[0] + 1
    # jumps: displacement much larger than the local cadence (dropped capture frame)
    mag = np.hypot(sh[:, 0], sh[:, 1]); good = sh[:, 2] > 0.08
    mmed = np.array([np.median(mag[max(0, i - 4):i + 5]) for i in range(len(mag))])
    jump = np.where(good & (mag > 1.9 * mmed + 2.0) & (mag > 4))[0] + 1
    # camera jitter: high-frequency part of the measured path (only where correlation is trustworthy)
    path = np.cumsum(np.where(good[:, None], sh[:, :2], 0), 0)
    hf = path - np.stack([gaussian_filter1d(path[:, k], 5, mode='nearest') for k in range(2)], 1)
    jit = float(np.sqrt((hf[good] ** 2).sum(1).mean())) if good.any() else float('nan')
    t = pts(name)
    dt = np.diff(t) if len(t) > 2 else np.array([1 / fps])
    return dict(name=name + (':' + pane if pane else ''), fps=round(fps, 3), n=len(d) + 1, dup=dup.tolist(), jump=jump.tolist(),
                jitter_px=round(jit, 2), trust=round(float(good.mean()), 2), dt_min=round(float(dt.min()) * 1000, 1), dt_max=round(float(dt.max()) * 1000, 1),
                vfr=bool(dt.max() > 1.5 * np.median(dt)))


if __name__ == '__main__':
    names = sorted(set(os.path.basename(x).replace('VID-20260927-', '').replace('.mp4', '') for x in glob.glob(S + '/raw/*/*.mp4')))
    jobs = []
    for n in names:
        if n in ('WA0032', 'WA0034'):
            jobs += [(n, 'L'), (n, 'R')]
        else:
            jobs.append((n, None))
    from multiprocessing import Pool
    with Pool(4) as p:
        res = p.starmap(analyse, jobs)
    json.dump(res, open(S + '/an/temporal.json', 'w'), indent=1)
    for r in res:
        print(f"{r['name']:24s} fps {r['fps']:6.3f} n {r['n']:4d} dups {len(r['dup']):3d} {r['dup'][:8]} jumps {len(r['jump']):3d} {r['jump'][:6]} "
              f"jitter {r['jitter_px']:5.2f}px trust {r['trust']:.2f} dt {r['dt_min']}-{r['dt_max']}ms vfr={r['vfr']}")
