"""Core helpers for the Campo de Marte edit: frame access, subject tracking, virtual 9:16 camera, grading."""
import os, json, glob, subprocess
import numpy as np, cv2
from scipy.ndimage import gaussian_filter1d

S = os.environ.get('S', '/tmp/claude-0/-home-user-Video/f8f0d442-cac9-55ab-9a30-19dc4913e7e5/scratchpad')
OUT_W, OUT_H, FPS = 1080, 1920, 24


def src_path(name):
    return [x for x in glob.glob(S + '/raw/*/*.mp4') if name in os.path.basename(x)][0]


def probe(name):
    p = src_path(name)
    cap = cv2.VideoCapture(p)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, n, w, h


def read_all(name):
    """Decode every frame sequentially (frame-accurate)."""
    cap = cv2.VideoCapture(src_path(name))
    fr = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(f)
    return fr


def pane(frame, which):
    if which == 'L':
        return frame[:, :640]
    if which == 'R':
        return frame[:, 640:1280]
    return frame


# ---------------------------------------------------------------- tracking
def load_dets(name):
    d = json.load(open(f'{S}/tracks/{name}.json'))
    return d['fps'], d['dets']


def subject_series(dets, n, W, H, prefer='cluster'):
    """Per-frame subject point + bbox from raw detections with temporal association.
    Returns arrays cx, cy (nan where missing), box (x0,y0,x1,y1)."""
    cx = np.full(n, np.nan); cy = np.full(n, np.nan)
    box = np.full((n, 4), np.nan)
    prev = None
    for i in range(min(n, len(dets))):
        D = [d for d in dets[i] if d['a'] >= 5]
        if not D:
            continue
        pts = np.array([[d['cx'], d['cy']] for d in D]); wts = np.array([d['score'] for d in D])
        if prev is not None:
            dist = np.hypot(pts[:, 0] - prev[0], pts[:, 1] - prev[1])
            keep = dist < 0.35 * W
            if keep.any():
                pts, wts, D = pts[keep], wts[keep], [d for d, k in zip(D, keep) if k]
        # dominant cluster: detections within 0.3W of the heaviest one
        j = int(np.argmax(wts))
        near = np.hypot(pts[:, 0] - pts[j, 0], pts[:, 1] - pts[j, 1]) < 0.30 * W
        pts, wts = pts[near], wts[near]; D = [d for d, k in zip(D, near) if k]
        c = (pts * wts[:, None]).sum(0) / wts.sum()
        cx[i], cy[i] = c
        box[i] = [min(d['x'] for d in D), min(d['y'] for d in D), max(d['x'] + d['w'] for d in D), max(d['y'] + d['h'] for d in D)]
        prev = c
    return cx, cy, box


def fill_nan(a):
    a = a.copy()
    idx = np.arange(len(a)); ok = ~np.isnan(a)
    if ok.sum() == 0:
        return None
    a[~ok] = np.interp(idx[~ok], idx[ok], a[ok])
    return a


# ---------------------------------------------------------------- camera motion (for non-tracked shots)
def global_shifts(frames, roi=None):
    """Content translation between consecutive frames via phase correlation."""
    sh = [(0.0, 0.0)]
    prev = None
    win = None
    for f in frames:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if roi is not None:
            y0, y1, x0, x1 = roi
            g = g[y0:y1, x0:x1]
        g = cv2.GaussianBlur(g, (0, 0), 1.0)
        g = g - cv2.GaussianBlur(g, (0, 0), 12)  # high-pass: texture, not exposure
        if win is None:
            win = cv2.createHanningWindow(g.shape[::-1], cv2.CV_32F)
        if prev is not None:
            (dx, dy), r = cv2.phaseCorrelate(prev, g, win)
            if r < 0.05 or abs(dx) > 60 or abs(dy) > 60:
                dx, dy = 0.0, 0.0
            sh.append((dx, dy))
        prev = g
    return np.array(sh[:len(frames)])


# ---------------------------------------------------------------- grading
def srgb_to_lin(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(x):
    x = np.clip(x, 0, None)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def shot_stats(frames):
    """Robust per-shot statistics used to normalise exposure and white balance."""
    smp = np.concatenate([cv2.resize(f, (160, int(160 * f.shape[0] / f.shape[1]))).reshape(-1, 3) for f in frames[::max(1, len(frames) // 12)]]).astype(np.float32) / 255
    lum = smp @ np.array([0.114, 0.587, 0.299], np.float32)
    lo, hi = np.percentile(lum, 0.7), np.percentile(lum, 99.6)
    # white balance from bright, low-saturation pixels (clouds / haze)
    mx = smp.max(1); mn = smp.min(1); sat = (mx - mn) / (mx + 1e-4)
    sel = (lum > np.percentile(lum, 70)) & (sat < 0.22)
    if sel.sum() < 50:
        sel = lum > np.percentile(lum, 80)
    wb = smp[sel].mean(0)  # B,G,R
    return dict(lo=float(lo), hi=float(hi), wb=wb.tolist(), mean=float(lum.mean()))


def make_grader(stats, p):
    """Returns f(bgr_uint8)->bgr_float01 implementing: WB -> levels -> dehaze/contrast -> look."""
    wb = np.array(stats['wb'], np.float32)
    g = wb[1]
    # neutralise the bright neutrals, keep a hint of afternoon warmth
    target = np.array([0.99, 1.0, 1.006], np.float32)  # B,G,R: neutral clouds, the faintest afternoon warmth
    gains = (g / np.maximum(wb, 1e-3)) * target
    s = p.get('wb_strength', 0.6)
    gains = 1 + (gains - 1) * s
    # Levels are only allowed to *nudge*: sky-only frames have a tiny luminance range (the aircraft are a
    # handful of pixels), and a full auto-stretch there explodes contrast and saturation.
    lo = float(np.clip(stats['lo'] * 0.5, 0.0, 0.06))
    hi = float(np.clip(stats['hi'] + 0.02, 0.90, 1.0))
    # gentle exposure normalisation towards a common daylight mid-level
    auto = float(np.clip(np.log2(0.60 / max(stats['mean'], 1e-3)) * 0.5, -0.10, 0.30))
    expo = p.get('expo', 0.0) + auto
    sat = p.get('sat', 1.04)
    con = p.get('contrast', 0.15)
    dehaze = p.get('dehaze', 0.0)
    warm = p.get('warm', 0.0)

    def f(img):
        x = img.astype(np.float32) / 255.0
        x = x * gains[None, None, :]
        x = (x - lo) / max(hi - lo, 0.2)
        if dehaze > 0:
            # subtract a fraction of the veiling haze estimated from a heavily blurred min-channel
            dc = cv2.erode(x.min(2), np.ones((15, 15), np.uint8))
            dc = cv2.GaussianBlur(dc, (0, 0), 25)
            veil = np.clip(dc, 0, 1)[..., None] * dehaze
            x = (x - veil) / np.maximum(1 - veil, 0.35)
        if expo:
            x = x * (2 ** expo)
        x = np.clip(x, 0, 1.2)
        # filmic contrast on luminance only (keeps hue)
        L = x @ np.array([0.114, 0.587, 0.299], np.float32)
        Lc = np.clip(L, 0, 1)
        curve = Lc + con * 2 * Lc * (1 - Lc) * (2 * Lc - 1)  # gentle S-curve: deeper darks, brighter upper mids
        # soft shoulder for highlights
        curve = np.where(curve > 0.85, 0.85 + (curve - 0.85) * 0.75 + 0.0, curve)
        ratio = (curve + 1e-4) / (L + 1e-4)
        x = x * ratio[..., None]
        # saturation with vibrance (protect already saturated + skin)
        L2 = (x @ np.array([0.114, 0.587, 0.299], np.float32))[..., None]
        chroma = x - L2
        cs = np.abs(chroma).max(2, keepdims=True)
        vib = sat + (sat - 1) * 0.8 * (1 - np.clip(cs * 4, 0, 1))
        x = L2 + chroma * vib
        if warm:
            x = x * np.array([1 - warm, 1.0, 1 + warm], np.float32)[None, None, :]
        # split tone: very subtle cool shadows / warm highlights
        L3 = np.clip(x @ np.array([0.114, 0.587, 0.299], np.float32), 0, 1)[..., None]
        sh = (1 - L3) ** 2; hl = L3 ** 2
        x = x + sh * np.array([0.010, 0.003, -0.005], np.float32) + hl * np.array([-0.006, 0.0, 0.004], np.float32)
        return np.clip(x, 0, 1)
    return f


def vignette(h, w, strength=0.10):
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((x - w / 2) / (w / 2)) ** 2 * 0.6 + ((y - h / 2) / (h / 2)) ** 2 * 0.6)
    v = 1 - strength * np.clip(r - 0.35, 0, None) ** 2 / 0.55
    return v[..., None].astype(np.float32)


class FFWriter:
    def __init__(self, path, w=OUT_W, h=OUT_H, fps=FPS, crf=14, preset='slow'):
        self.p = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{w}x{h}', '-r', str(fps), '-i', '-',
                                   '-c:v', 'libx264', '-preset', preset, '-crf', str(crf), '-pix_fmt', 'yuv420p', '-color_primaries', 'bt709',
                                   '-color_trc', 'bt709', '-colorspace', 'bt709', path], stdin=subprocess.PIPE)

    def write(self, img):
        self.p.stdin.write(np.ascontiguousarray(img).tobytes())

    def close(self):
        self.p.stdin.close(); self.p.wait()
