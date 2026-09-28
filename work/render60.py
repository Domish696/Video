"""60 fps renderer (fluidity pass).

For every output instant t (60 fps) of a shot:
  * find the two *real* source frames around t using their true decode timestamps (duplicates removed),
  * place a 9:16 virtual camera: slow subject/key path + removal of involuntary tremor (>~2 Hz) measured on the
    image content, keeping a share of it so it still feels hand-held,
  * reconstruct the in-between moment with RIFE v4.26 flow computed at source resolution, and transport the
    AI-upscaled versions of the two real frames along that flow (the upscaler only ever sees real frames, so the
    detail it adds is temporally stable instead of 'boiling' frame to frame),
  * add fresh fine grain + vignette at 60 fps.
usage: render60.py [--final] [--only S05,D1] [--outdir DIR] [--debug]
"""
import os, sys, json, time, argparse
import numpy as np, cv2
from scipy.ndimage import gaussian_filter1d
sys.path.insert(0, os.path.dirname(__file__))
from lib import S, src_path, probe, pane, load_dets, subject_series, fill_nan, shot_stats, make_grader, vignette, FFWriter, OUT_W, OUT_H
from edl import SHOTS, timeline, FPS_OUT
import rife

ap = argparse.ArgumentParser()
ap.add_argument('--final', action='store_true')
ap.add_argument('--only', default='')
ap.add_argument('--outdir', default=S + '/shots60')
ap.add_argument('--debug', action='store_true')
ap.add_argument('--fast', action='store_true', help='preview: RIFE flow at half scale')
ap.add_argument('--procs', type=int, default=1)
ap.add_argument('--nearest', action='store_true', help='framing preview: no interpolation')
args = ap.parse_args()
os.makedirs(args.outdir, exist_ok=True)
if args.final:
    from sr import upscale4
FRAME_TS = json.load(open(S + '/an/frame_ts.json'))


# ------------------------------------------------------------------ real frames with true timestamps
def load_segment(name, t0, t1):
    ts = np.array(FRAME_TS[name])
    idx = np.where((ts >= t0 - 0.25) & (ts <= t1 + 0.25))[0]
    i0, i1 = max(0, idx[0] - 1), min(len(ts) - 1, idx[-1] + 1)
    cap = cv2.VideoCapture(src_path(name))
    fr = []
    i = 0
    while i <= i1:
        ok, f = cap.read()
        if not ok:
            break
        if i >= i0:
            fr.append(f)
        i += 1
    tt = ts[i0:i0 + len(fr)]
    return fr, tt, i0


def dedup(frames, times):
    """Drop repeated frames (encoder/phone duplicates): near-zero change while the neighbourhood moves."""
    g = [cv2.cvtColor(cv2.resize(f, (f.shape[1] // 4, f.shape[0] // 4), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames]
    d = np.array([np.abs(g[i] - g[i - 1]).mean() for i in range(1, len(g))])
    keep = [0]
    for i in range(1, len(frames)):
        loc = np.median(d[max(0, i - 5):i + 4])
        if d[i - 1] < 0.35 and loc > 1.2:
            continue
        keep.append(i)
    return keep, len(frames) - len(keep)


def content_motion(frames):
    """Cumulative content displacement (px) and reliability, frame to frame, via phase correlation."""
    C = [np.zeros(2)]; R = [1.0]
    prev = None; win = None
    for f in frames:
        g = cv2.cvtColor(cv2.resize(f, (f.shape[1] // 2, f.shape[0] // 2), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
        g = g - cv2.GaussianBlur(g, (0, 0), 10)
        if win is None:
            win = cv2.createHanningWindow(g.shape[::-1], cv2.CV_32F)
        if prev is not None:
            (dx, dy), r = cv2.phaseCorrelate(prev, g, win)
            ok = r > 0.06 and abs(dx) < 60 and abs(dy) < 60
            C.append(C[-1] + (np.array([dx, dy]) * 2 if ok else 0))
            R.append(r)
        prev = g
    return np.array(C), np.array(R)


def tremor(times, C, tq, sigma_s=0.10):
    """High-frequency part of the content path, resampled at query times tq."""
    grid = np.arange(times[0], times[-1] + 1e-6, 1 / 240)
    if len(grid) < 8:
        return np.zeros((len(tq), 2))
    out = []
    for k in range(2):
        c = np.interp(grid, times, C[:, k])
        hf = c - gaussian_filter1d(c, sigma_s * 240, mode='nearest')
        out.append(np.interp(tq, grid, hf))
    return np.stack(out, 1)


def smooth_t(tq, v, sigma_s):
    grid = np.arange(tq[0], tq[-1] + 1e-6, 1 / 240)
    g = np.interp(grid, tq, v)
    g = gaussian_filter1d(g, sigma_s * 240, mode='nearest')
    return np.interp(tq, grid, g)


# ------------------------------------------------------------------ camera paths
def camera_path(s, name, frames, times, i0, tq, W, H, cw, ch):
    C, R = content_motion(frames)
    J = tremor(times, C, tq) * s.get('shake', 0.8)
    mode = s['mode']
    if mode == 'track':
        fps, dets = load_dets(name)
        cx, cy, box = subject_series(dets, len(dets), W, H)
        ft = np.array(FRAME_TS[name])[:len(dets)]
        ok = ~np.isnan(cx)
        if ok.sum() < 2:
            x = np.full(len(tq), W / 2); y = np.full(len(tq), H / 2); b = np.tile([W / 2 - 5, H / 2 - 5, W / 2 + 5, H / 2 + 5], (len(tq), 1))
        else:
            x = np.interp(tq, ft[ok], cx[ok]); y = np.interp(tq, ft[ok], cy[ok])
            b = np.stack([np.interp(tq, ft[ok], box[ok, k]) for k in range(4)], 1)
        if s.get('xkeys'):
            kt = np.array([k[0] for k in s['xkeys']]); kx = np.array([k[1] for k in s['xkeys']]) * W
            m = tq >= kt[0]
            x = np.where(m, np.interp(tq, kt, kx), x)
            b = np.where(m[:, None], np.stack([x - 6, y - 6, x + 6, y + 6], 1), b)
        base = smooth_t(tq, x + s.get('lead', 0.0) * cw, 0.38)
        mg = 0.07 * cw
        for it in range(3):
            left = base - cw / 2; right = base + cw / 2
            bw = b[:, 2] - b[:, 0]; fits = bw < cw - 2 * mg
            push = np.zeros_like(base)
            push = np.where(fits & (b[:, 0] < left + mg), b[:, 0] - (left + mg), push)
            push = np.where(fits & (b[:, 2] > right - mg), b[:, 2] - (right - mg), push)
            push = np.where(~fits, (b[:, 0] + b[:, 2]) / 2 - base, push)
            base = base + smooth_t(tq, push, 0.12)
        px = base + J[:, 0]
        py = smooth_t(tq, y, 0.7) + s.get('vbias', 0.0) * ch + J[:, 1]
    else:
        if s.get('cxkeys'):
            kx = np.interp(tq, [k[0] for k in s['cxkeys']], [k[1] * W for k in s['cxkeys']])
            kx = smooth_t(tq, kx, 0.15)
        else:
            kx = np.full(len(tq), s.get('cx', 0.5) * W)
        px = kx + J[:, 0]
        py = np.full(len(tq), s.get('cy', 0.5) * H) + J[:, 1]
    px = np.clip(px, cw / 2, W - cw / 2); py = np.clip(py, ch / 2, H - ch / 2)
    return px, py


# ------------------------------------------------------------------ helpers
def out_grid(ow, oh):
    X = (np.arange(ow, dtype=np.float32) + 0.5) / ow
    Y = (np.arange(oh, dtype=np.float32) + 0.5) / oh
    return np.meshgrid(X, Y)


def grain(img, rng, amt=0.010):
    h, w = img.shape[:2]
    n = rng.standard_normal((h // 2, w // 2)).astype(np.float32)
    n = cv2.resize(n, (w, h), interpolation=cv2.INTER_LINEAR)
    L = img.mean(2, keepdims=True)
    k = amt * (0.6 + 0.8 * L * (1 - L) * 4) / 1.4
    return np.clip(img + n[..., None] * k, 0, 1)


class RealFrameCache:
    """Upscaled (or not) graded regions of real frames, built lazily."""

    def __init__(self, frames, grader, regions, sc, s):
        self.frames, self.grader, self.regions, self.sc, self.s = frames, grader, regions, sc, s
        self.cache = {}

    def get(self, i):
        if i not in self.cache:
            x0, y0, x1, y1 = self.regions[i]
            g = self.grader(self.frames[i][y0:y1, x0:x1])
            if self.sc == 4:
                up = upscale4(g, denoise=self.s.get('sr_dn', 0.5))
                # de-ringing: keep the AI detail but forbid overshoot beyond the local range of a plain upscale
                lz = cv2.resize(g, (up.shape[1], up.shape[0]), interpolation=cv2.INTER_CUBIC)
                k5 = np.ones((5, 5), np.uint8)
                up = np.clip(up, cv2.erode(lz, k5) - 0.015, cv2.dilate(lz, k5) + 0.015)
                fm = self.s.get('face_mix', 0.0)
                if fm > 0:
                    up = up * (1 - fm) + cv2.resize(g, (up.shape[1], up.shape[0]), interpolation=cv2.INTER_LANCZOS4) * fm
                img = up.astype(np.float32)
            else:
                img = g.astype(np.float32)
            self.cache[i] = (img, x0, y0)
            for k in list(self.cache):
                if k < i - 2:
                    del self.cache[k]
        return self.cache[i]


def sample(entry, px, py, sc, interp):
    img, x0, y0 = entry
    mx = ((px - x0 + 0.5) * sc - 0.5).astype(np.float32)
    my = ((py - y0 + 0.5) * sc - 0.5).astype(np.float32)
    return cv2.remap(img, mx, my, interp, borderMode=cv2.BORDER_REPLICATE)


# ------------------------------------------------------------------ one layer (full frame or a pane)
def render_layer(s, name, which, t_start, n, ow, oh, zoom, final, report):
    fps_src, _, W0, H0 = probe(name)
    tq = s['t_in'] + np.arange(n) / FPS_OUT * s.get('speed', 1.0)
    fr_all, tt_all, i0 = load_segment(name, tq[0], tq[-1])
    fr_all = [pane(f, which) for f in fr_all]
    keep, ndup = dedup(fr_all, tt_all)
    frames = [fr_all[k] for k in keep]; times = tt_all[keep]
    H, W = frames[0].shape[:2]
    # crop size in source pixels (9:16 for full frames; panes keep their half-frame aspect)
    aspect = ow / oh
    ch = H * zoom; cw = ch * aspect
    if cw > W * zoom:
        cw = W * zoom; ch = cw / aspect
    px, py = camera_path(s, name, frames, times, i0, tq, W, H, cw, ch)
    # bracket every output instant between two real frames
    ia = np.clip(np.searchsorted(times, tq, side='right') - 1, 0, len(times) - 2)
    u = np.clip((tq - times[ia]) / (times[ia + 1] - times[ia]), 0, 1)
    # regions each real frame must provide (crop windows that use it + flow margin)
    mgn = 28
    reg = {}
    for k in range(n):
        box = (px[k] - cw / 2, py[k] - ch / 2, px[k] + cw / 2, py[k] + ch / 2)
        for i in (ia[k], ia[k] + 1):
            r = reg.get(i, [1e9, 1e9, -1e9, -1e9])
            reg[i] = [min(r[0], box[0]), min(r[1], box[1]), max(r[2], box[2]), max(r[3], box[3])]
    regions = {i: (int(max(0, np.floor(r[0] - mgn))), int(max(0, np.floor(r[1] - mgn))), int(min(W, np.ceil(r[2] + mgn))), int(min(H, np.ceil(r[3] + mgn))))
               for i, r in reg.items()}
    st = shot_stats(frames[::3])
    grader = make_grader(st, s.get('grade', {}))
    use_sr = final and not s.get('nosr') and s['mode'] != 'stack'
    sc = 4 if use_sr else 1
    cache = RealFrameCache(frames, grader, regions, sc, s)
    _gcache = {}

    def graded(i):
        if i not in _gcache:
            _gcache[i] = grader(frames[i])
            for kk in list(_gcache):
                if kk < i - 2:
                    del _gcache[kk]
        return _gcache[i]
    interp = cv2.INTER_CUBIC if sc == 4 else cv2.INTER_LANCZOS4
    GX, GY = out_grid(ow, oh)
    report.update(dict(real_frames=len(frames), duplicates_removed=ndup, max_gap_ms=round(float(np.diff(times).max()) * 1000, 1),
                       interpolated=int(((u > 0.03) & (u < 0.97)).sum()), frames_out=n))
    # crop-centre velocity (source px / s) to turn content motion into on-screen motion for the shutter
    vcx = np.gradient(px) * FPS_OUT; vcy = np.gradient(py) * FPS_OUT
    PX = np.arange(ow, dtype=np.float32)[None, :].repeat(oh, 0); PY = np.arange(oh, dtype=np.float32)[:, None].repeat(ow, 1)
    rscale = s.get('rife_scale', 0.5 if (args.fast or s['mode'] == 'stack') else 1.0)
    for k in range(n):
        spx = (px[k] - cw / 2 + GX * cw - 0.5).astype(np.float32)
        spy = (py[k] - ch / 2 + GY * ch - 0.5).astype(np.float32)
        a, b = ia[k], ia[k] + 1
        ra, rb = regions[a], regions[b]
        q = (min(ra[0], rb[0]), min(ra[1], rb[1]), max(ra[2], rb[2]), max(ra[3], rb[3]))
        ga = graded(a)[q[1]:q[3], q[0]:q[2]]; gb = graded(b)[q[1]:q[3], q[0]:q[2]]
        if args.nearest:
            img = sample(cache.get(a if u[k] < 0.5 else b), spx, spy, sc, interp)
            yield np.clip(img, 0, 1)
            continue
        fa, fb, m = rife.flows(ga, gb, float(np.clip(u[k], 0.001, 0.999)), scale=rscale)
        qx = (spx - q[0]).astype(np.float32); qy = (spy - q[1]).astype(np.float32)
        rm = lambda z: cv2.remap(z, qx, qy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        fax, fay, fbx, fby = rm(fa[..., 0]), rm(fa[..., 1]), rm(fb[..., 0]), rm(fb[..., 1])
        mm = rm(m)[..., None]
        if u[k] <= 0.02 or u[k] >= 0.98:      # (practically) a real frame: show it as captured
            img = sample(cache.get(a if u[k] <= 0.02 else b), spx, spy, sc, interp)
        else:
            A = sample(cache.get(a), spx + fax, spy + fay, sc, interp)
            B = sample(cache.get(b), spx + fbx, spy + fby, sc, interp)
            # anti-ghost: where the two transported frames disagree (occlusion, smoke, flow failure) do not
            # superimpose them - take the temporally nearer real frame there
            dis = cv2.GaussianBlur(np.abs(A - B).mean(2), (0, 0), 2.0)
            w = np.clip((dis - 0.07) / 0.08, 0, 1)[..., None]
            near = 1.0 if u[k] < 0.5 else 0.0
            mm = mm * (1 - w) + near * w
            img = A * mm + B * (1 - mm)
            report['ghost_guard_px'] = report.get('ghost_guard_px', 0) + int((w > 0.5).sum())
        # physically-plausible short shutter: blur along each pixel's on-screen velocity
        shutter = s.get('shutter', 0.5)
        if shutter > 0:
            dtab = max(times[b] - times[a], 1e-3)
            kx = ow / cw; ky = oh / ch
            dx = ((fbx - fax) / dtab - vcx[k]) * kx * shutter / FPS_OUT
            dy = ((fby - fay) / dtab - vcy[k]) * ky * shutter / FPS_OUT
            mag = np.sqrt(dx * dx + dy * dy)
            lim = np.minimum(1.0, 36.0 / np.maximum(mag, 1e-6))
            dx = (dx * lim).astype(np.float32); dy = (dy * lim).astype(np.float32)
            if float(np.percentile(mag, 99)) > 1.0:
                acc = np.zeros_like(img)
                for sft in (-0.5, -0.25, 0.0, 0.25, 0.5):
                    acc += cv2.remap(img, (PX + sft * dx).astype(np.float32), (PY + sft * dy).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                img = acc / 5.0
        if sc == 1:  # plain upscale: a gentle unsharp so it sits with the AI-upscaled shots
            bl = cv2.GaussianBlur(img, (0, 0), 1.1)
            img = np.clip(img + 0.30 * (img - bl), 0, 1)
        yield np.clip(img, 0, 1)
        if args.debug and k % 30 == 0:
            print(f"   {s['id']}{which or ''} {k}/{n} u={u[k]:.2f}", flush=True)


def render_shot(s, t_start, n):
    rng = np.random.default_rng(abs(hash(s['id'])) % 2 ** 32)
    vig = vignette(OUT_H, OUT_W, 0.10)
    path = f"{args.outdir}/{s['id']}.mp4"
    report = {}
    if s['mode'] == 'stack':
        ph = OUT_H // 2 - 3
        top = render_layer(s, s['src'], 'L', t_start, n, OUT_W, ph, 0.96, args.final, report)
        rep2 = {}
        bot = render_layer(dict(s, grade=dict(s.get('grade', {}), sat=1.02)), s['src'], 'R', t_start, n, OUT_W, ph, 0.96, args.final, rep2)
        gap = np.full((OUT_H - 2 * ph, OUT_W, 3), 0.03, np.float32)
        frames = (np.vstack([a, gap, b]) for a, b in zip(top, bot))
    else:
        zoom = s.get('zoom', 1.0)
        frames = render_layer(s, s['src'], None, t_start, n, OUT_W, OUT_H, zoom, args.final, report)
    wr = FFWriter(path, fps=FPS_OUT, crf=12, preset='fast')
    for img in frames:
        x = grain(img * vig, rng)
        wr.write((np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8))
    wr.close()
    json.dump(report, open(f"{args.outdir}/{s['id']}.json", 'w'))
    return path, report


def _job(j):
    s, t, n = j
    t0 = time.time()
    p, rep = render_shot(s, t, n)
    return f"{s['id']} {n}f {time.time() - t0:.0f}s {rep}"


if __name__ == '__main__':
    tl, T = timeline()
    only = set(args.only.split(',')) if args.only else None
    jobs = [(s, t, n) for s, t, n in tl if not only or s['id'] in only]
    if args.procs <= 1:
        for j in jobs:
            print(_job(j), flush=True)
    else:
        from multiprocessing import Pool
        with Pool(args.procs) as pool:
            for r in pool.imap_unordered(_job, sorted(jobs, key=lambda j: -j[2])):
                print(r, flush=True)
