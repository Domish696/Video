"""Render each EDL shot to a 1080x1920 24 fps intermediate.
usage: render.py [--final] [--only S05,S06] [--debug]
"""
import os, sys, json, time, argparse
import numpy as np, cv2
from scipy.ndimage import gaussian_filter1d
sys.path.insert(0, os.path.dirname(__file__))
from lib import *
from edl import SHOTS, timeline

ap = argparse.ArgumentParser()
ap.add_argument('--final', action='store_true')
ap.add_argument('--only', default='')
ap.add_argument('--debug', action='store_true')
ap.add_argument('--outdir', default=S + '/shots')
args = ap.parse_args()
os.makedirs(args.outdir, exist_ok=True)
if args.final:
    from sr import upscale4

DIS = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)


def interp(fa, fb, a):
    """Motion-compensated in-between frame (bidirectional DIS optical flow)."""
    if a < 0.08:
        return fa
    if a > 0.92:
        return fb
    ga = cv2.cvtColor(fa, cv2.COLOR_BGR2GRAY); gb = cv2.cvtColor(fb, cv2.COLOR_BGR2GRAY)
    fab = DIS.calc(ga, gb, None); fba = DIS.calc(gb, ga, None)
    h, w = ga.shape
    a = np.float32(a)
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    wa = cv2.remap(fa, (gx - a * fab[..., 0]).astype(np.float32), (gy - a * fab[..., 1]).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    wb = cv2.remap(fb, (gx - (1 - a) * fba[..., 0]).astype(np.float32), (gy - (1 - a) * fba[..., 1]).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return cv2.addWeighted(wa, float(1 - a), wb, float(a), 0)


def read_range(name, t0, t1):
    fps, n, w, h = probe(name)
    i0 = max(0, int(np.floor(t0 * fps)) - 1); i1 = min(n - 1, int(np.ceil(t1 * fps)) + 1)
    cap = cv2.VideoCapture(src_path(name))
    frames = []
    i = 0
    while i <= i1:
        ok, f = cap.read()
        if not ok:
            break
        if i >= i0:
            frames.append(f)
        i += 1
    return fps, i0, frames


def src_positions(s, nout, fps):
    sp = s.get('speed', 1.0)
    return np.array([(s['t_in'] + k / FPS * sp) * fps for k in range(nout)])  # fractional source frame index


def sample(frames, i0, pos, which=None):
    j = pos - i0
    a = int(np.floor(j)); fr = j - a
    a = int(np.clip(a, 0, len(frames) - 1)); b = min(a + 1, len(frames) - 1)
    fa = pane(frames[a], which); fb = pane(frames[b], which)
    if fr < 0.08 or a == b:
        return fa
    if fr > 0.92:
        return fb
    return interp(fa, fb, fr)


def crop_track(s, name, pos, W, H, cw, ch, which=None):
    fps, dets = load_dets(name)
    n = len(dets)
    cx, cy, box = subject_series(dets, n, W, H)
    idx = np.clip(np.round(pos).astype(int), 0, n - 1)
    x = cx[idx]; y = cy[idx]; b = box[idx]
    if np.isnan(x).all():
        x = np.full(len(idx), W / 2); y = np.full(len(idx), H / 2); b = np.tile([W / 2 - 5, H / 2 - 5, W / 2 + 5, H / 2 + 5], (len(idx), 1))
    x = fill_nan(x); y = fill_nan(y)
    for k in range(4):
        b[:, k] = fill_nan(b[:, k])
    if s.get('xkeys'):
        # manual operator override: from the first key on, follow the keyed position (normalised x)
        ts = pos / fps
        kt = np.array([k[0] for k in s['xkeys']]); kx = np.array([k[1] for k in s['xkeys']]) * W
        m = ts >= kt[0]
        x = np.where(m, np.interp(ts, kt, kx), x)
        half = cw / 2 - 0.08 * cw
        b = np.where(m[:, None], np.stack([x - 6, y - 6, x + 6, y + 6], 1), b)
    lead = s.get('lead', 0.0) * cw
    sig = s.get('smooth', 9.0)
    base = gaussian_filter1d(x + lead, sig, mode='nearest')
    fine = gaussian_filter1d(x + lead, 1.2, mode='nearest')
    path = base + s.get('lock', 0.55) * (fine - base)
    # keep the whole formation inside the frame with a margin
    m = 0.07 * cw
    for it in range(3):
        left = path - cw / 2; right = path + cw / 2
        push = np.zeros_like(path)
        bw = b[:, 2] - b[:, 0]
        fits = bw < cw - 2 * m
        push = np.where(fits & (b[:, 0] < left + m), b[:, 0] - (left + m), push)
        push = np.where(fits & (b[:, 2] > right - m), b[:, 2] - (right - m), push)
        push = np.where(~fits, (b[:, 0] + b[:, 2]) / 2 - path, push)
        path = path + gaussian_filter1d(push, 4, mode='nearest')
    path = gaussian_filter1d(path, 2.0, mode='nearest')
    px = np.clip(path, cw / 2, W - cw / 2)
    # vertical: only if there is room (zoom<1)
    if ch < H - 1:
        by = gaussian_filter1d(y, sig, mode='nearest') + s.get('vbias', 0.0) * ch
        py = np.clip(by, ch / 2, H - ch / 2)
    else:
        py = np.full(len(px), H / 2)
    return px, py, x, y, b


def crop_fixed(s, frames, i0, pos, W, H, cw, ch, which=None):
    # handheld jitter removal: smooth the measured camera path and counter-shift the window
    idx = np.clip(np.round(pos - i0).astype(int), 0, len(frames) - 1)
    sub = [pane(frames[i], which) for i in range(len(frames))]
    small = [cv2.resize(f, (W // 2, H // 2)) for f in sub]
    sh = global_shifts(small) * 2.0
    C = np.cumsum(sh, 0)  # content displacement
    Cs = np.stack([gaussian_filter1d(C[:, k], s.get('smooth', 10.0), mode='nearest') for k in range(2)], 1)
    corr = (C - Cs) * s.get('stab', 1.0)
    cx0 = s.get('cx', 0.5) * W; cy0 = s.get('cy', 0.5) * H
    if s.get('cxkeys'):
        ts = pos / probe(s['src'])[0]
        cx0 = np.interp(ts, [k[0] for k in s['cxkeys']], [k[1] * W for k in s['cxkeys']])
        cx0 = gaussian_filter1d(cx0, 4, mode='nearest')
    px = np.clip(cx0 + corr[idx, 0], cw / 2, W - cw / 2)
    py = np.clip(cy0 + corr[idx, 1], ch / 2, H - ch / 2)
    return px, py


def extract(frame, cx, cy, cw, ch, ow, oh, interp_flag=cv2.INTER_LANCZOS4):
    sx = ow / cw; sy = oh / ch
    M = np.array([[sx, 0, ow / 2 - sx * cx], [0, sy, oh / 2 - sy * cy]], np.float32)
    return cv2.warpAffine(frame, M, (ow, oh), flags=interp_flag, borderMode=cv2.BORDER_REFLECT)


def grain(img, rng, amt=0.010):
    h, w = img.shape[:2]
    n = rng.standard_normal((h // 2, w // 2)).astype(np.float32)
    n = cv2.resize(n, (w, h), interpolation=cv2.INTER_LINEAR)
    L = img.mean(2, keepdims=True)
    k = amt * (0.6 + 0.8 * L * (1 - L) * 4) / 1.4  # strongest in midtones, like film
    return np.clip(img + n[..., None] * k, 0, 1)


def finish(img01, rng, vig):
    x = img01 * vig
    x = grain(x, rng)
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)


def to_out(crop_bgr_u8, grader, final, sr_dn=0.5, face_mix=0.0):
    g = grader(crop_bgr_u8)  # float01 at crop resolution
    if final:
        up = upscale4(g, denoise=sr_dn)
        if face_mix > 0:
            lz = cv2.resize(g, (up.shape[1], up.shape[0]), interpolation=cv2.INTER_LANCZOS4)
            up = up * (1 - face_mix) + lz * face_mix
        out = cv2.resize(up, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
    else:
        out = cv2.resize(g, (OUT_W, OUT_H), interpolation=cv2.INTER_LANCZOS4)
        # preview: gentle unsharp to approximate the final crispness
        bl = cv2.GaussianBlur(out, (0, 0), 1.2)
        out = np.clip(out + 0.35 * (out - bl), 0, 1)
    return out


def render(s, t_start, nout):
    name = s['src']
    fps, n, W0, H0 = probe(name)
    pos = src_positions(s, nout, fps)
    t0 = pos.min() / fps - 0.6; t1 = pos.max() / fps + 0.6
    fps_, i0, frames = read_range(name, max(0, t0), t1)
    rng = np.random.default_rng(abs(hash(s['id'])) % 2 ** 32)
    vig = vignette(OUT_H, OUT_W, 0.10)
    path = f"{args.outdir}/{s['id']}.mp4"
    wr = FFWriter(path)
    mode = s['mode']
    used = [pane(frames[int(np.clip(round(p - i0), 0, len(frames) - 1))], 'L' if mode == 'stack' else None) for p in pos[::3]]
    if mode == 'stack':
        stR = shot_stats([pane(frames[int(np.clip(round(p - i0), 0, len(frames) - 1))], 'R') for p in pos[::3]])
        stL = shot_stats(used)
        gL = make_grader(stL, s.get('grade', {})); gR = make_grader(stR, dict(s.get('grade', {}), sat=1.04))
        ph = OUT_H // 2 - 3
        for k in range(nout):
            A = sample(frames, i0, pos[k], 'L'); B = sample(frames, i0, pos[k], 'R')
            outs = []
            for P, g in ((A, gL), (B, gR)):
                P = P[:, 1:639]  # trim pane edge artefacts
                sc = max(OUT_W / P.shape[1], ph / P.shape[0])  # cover the half-frame
                ww = int(np.ceil(P.shape[1] * sc)); hh = int(np.ceil(P.shape[0] * sc))
                x = cv2.resize(g(P), (ww, hh), interpolation=cv2.INTER_LANCZOS4)
                bl = cv2.GaussianBlur(x, (0, 0), 1.0); x = np.clip(x + 0.30 * (x - bl), 0, 1)
                y0 = (hh - ph) // 2; x0 = (ww - OUT_W) // 2
                outs.append(x[y0:y0 + ph, x0:x0 + OUT_W])
            gap = np.full((OUT_H - 2 * ph, OUT_W, 3), 0.03, np.float32)
            img = np.vstack([outs[0], gap, outs[1]])
            wr.write(finish(img, rng, vig))
        wr.close()
        return path
    which = None
    W, H = W0, H0
    if mode == 'native':
        ch = H * s.get('zoom', 1.0); cw = ch * 9 / 16
        if cw > W:
            cw = W; ch = cw * 16 / 9
        px = np.full(nout, W / 2); py = np.full(nout, H / 2)
    else:
        ch = H * s.get('zoom', 1.0); cw = ch * 9 / 16
        if mode == 'track':
            px, py, sx, sy, sb = crop_track(s, name, pos, W, H, cw, ch)
        else:
            px, py = crop_fixed(s, frames, i0, pos, W, H, cw, ch)
    st = shot_stats(used)
    grader = make_grader(st, s.get('grade', {}))
    cw_i = int(round(cw)); ch_i = int(round(ch))
    for k in range(nout):
        f = sample(frames, i0, pos[k], which)
        crop = extract(f, px[k], py[k], cw, ch, cw_i, ch_i, cv2.INTER_CUBIC)
        img = to_out(crop, grader, args.final, sr_dn=s.get('sr_dn', 0.5), face_mix=s.get('face_mix', 0.0))
        if args.debug and mode == 'track':
            sc = OUT_W / cw
            X = (sx[k] - (px[k] - cw / 2)) * sc; Y = (sy[k] - (py[k] - ch / 2)) * (OUT_H / ch)
            cv2.circle(img, (int(X), int(Y)), 12, (0, 0, 1), 2)
        wr.write(finish(img, rng, vig))
    wr.close()
    return path


def _job(a):
    s, t, n = a
    t0 = time.time()
    p = render(s, t, n)
    return f"{s['id']} {n}f {time.time() - t0:.1f}s -> {p}"


if __name__ == '__main__':
    tl, T = timeline()
    only = set(args.only.split(',')) if args.only else None
    jobs = [(s, t, n) for s, t, n in tl if not only or s['id'] in only]
    nproc = 1 if args.final else int(os.environ.get('NPROC', '4'))
    if nproc == 1:
        for j in jobs:
            print(_job(j), flush=True)
    else:
        from multiprocessing import Pool
        with Pool(nproc) as pool:
            for r in pool.imap_unordered(_job, sorted(jobs, key=lambda j: -j[2])):
                print(r, flush=True)
