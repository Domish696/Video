"""Mix: real location sound per shot (with J/L overlaps), the PA announcer lines, and the score with
section automation + voice-keyed ducking. Output: $S/mix/final_audio.wav (48 kHz stereo, -14 LUFS, -1 dBTP)."""
import os, sys, json
import numpy as np, soundfile as sf, pyloudnorm as pyln
from scipy.signal import resample_poly
sys.path.insert(0, os.path.dirname(__file__))
from synth import hp, lp, bp, make_ir, reverb, sos_filter, SR, ns
from edl import timeline

S = os.environ['S']
os.makedirs(S + '/mix', exist_ok=True)
tl, T = timeline()
st = {s['id']: (s, t, n / 24) for s, t, n in tl}
L = ns(T + 1.0)
_cache = {}


def src(name):
    if name not in _cache:
        x, sr = sf.read(f'{S}/audio48/{name}.wav', dtype='float32')
        if x.ndim == 1:
            x = np.stack([x, x], 1)
        _cache[name] = x.astype(np.float64)
    return _cache[name]


def clip(name, a, b):
    x = src(name)
    i, j = ns(max(a, 0)), ns(b)
    y = x[i:j]
    if a < 0:
        y = np.concatenate([np.zeros((ns(-a), 2)), y])
    return y.copy()


def fades(y, fin, fout, shape='eqp'):
    n = len(y)
    a, b = ns(fin), ns(fout)
    if a > 0:
        r = np.linspace(0, 1, min(a, n))
        y[:len(r)] *= (np.sin(r * np.pi / 2) if shape == 'eqp' else r)[:, None]
    if b > 0:
        r = np.linspace(1, 0, min(b, n))
        y[n - len(r):] *= (np.sin(r * np.pi / 2) if shape == 'eqp' else r)[:, None]
    return y


def clean(y, hpf=70):
    """remove handling/wind rumble below the engines' fundamental, tame harsh phone-mic top end"""
    out = np.stack([hp(y[:, c], hpf, 0.6) for c in range(2)], 1)
    out = np.stack([lp(out[:, c], 11000, 0.6) for c in range(2)], 1)
    return out


AMB = np.zeros((L, 2)); VO = np.zeros((L, 2))


def place(buf, y, t, gain_db):
    i = ns(t)
    if i < 0:
        y = y[-i:]; i = 0
    j = min(L, i + len(y))
    buf[i:j] += y[:j - i] * 10 ** (gain_db / 20)


# ---------------------------------------------------------------- per-shot location sound
#            gain dB, pre(J), post(L), fade-in, fade-out
AUD = dict(S01=(0.0, 0.0, 1.3, 0.05, 1.3), S02=None, S03=(-12, 0.1, 0.1, 0.2, 0.2), S04=(-14, 0.1, 0.1, 0.2, 0.3),
           S05=(-3, 0.6, 0.15, 1.4, 0.3), S06=(0.0, 0.3, 0.15, 0.3, 0.3), S07=(-2, 0.15, 0.15, 0.3, 0.3),
           S08=(-3, 0.15, 0.15, 0.25, 0.3), S09=(-3, 0.15, 0.15, 0.25, 0.3), S10=(-4, 0.15, 0.15, 0.3, 0.3),
           S11=(-7, 0.15, 0.1, 0.3, 0.2), S12=(-7, 0.1, 0.05, 0.2, 0.1), S13=(1.0, 0.03, 0.2, 0.03, 0.3),
           S14=(-1, 0.15, 3.3, 0.25, 0.9), S15=(-10, 0.1, 0.1, 0.2, 0.2), S16=(-2, 0.1, 0.1, 0.2, 0.25),
           S17=(-4, 0.1, 0.1, 0.2, 0.25), S18=(-7, 0.1, 0.0, 0.3, 1.6))
for sid, a in AUD.items():
    if a is None:
        continue
    s, t, d = st[sid]
    g, pre, post, fi, fo = a
    y = clip(s['src'], s['t_in'] - pre, s['t_in'] + d + post)
    y = clean(y)
    y = fades(y, fi, fo)
    place(AMB, y, t - pre, g)

# ---------------------------------------------------------------- announcer lines (PA of the Esquadrilha da Fumaça)
def vo(name, a, b, t, g, fi=0.25, fo=0.35):
    y = clip(name, a, b)
    y = clean(y, 90)
    # presence: gently lift the intelligibility band
    pres = np.stack([bp(y[:, c], 2400, 0.8) for c in range(2)], 1)
    y = y + 0.35 * pres
    y = fades(y, fi, fo)
    place(VO, y, t, g)
    return (t, t + b - a)


s02 = st['S02'][1]
vo_win = []
vo_win.append(vo('WA0052', 14.10, 24.45, s02 + 0.5, 0.0, fi=0.4, fo=0.5))          # "...a Esquadrilha da Fumaça, com seus sete aviões ... inicia a demonstração número 4194 aqui na cidade de São Paulo."
t_s13 = st['S13'][1]
vo_win.append(vo('WA0098', 2.62, 5.62, t_s13 - 3.0, 1.5, fi=0.12, fo=0.2))          # "vamos ouvir os motores da Esquadrilha da Fumaça, A-29 Super Tucano."
s14, t14, d14 = st['S14']
vo_win.append((t14 + (15.10 - s14['t_in']), t14 + (17.0 - s14['t_in'])))             # "Uma salva de palmas..." (inside S14's L-cut)

# glue: a short shared room so clips from different phones sit in one space
ir = make_ir(0.9, 0.01, 6000, early=True, seed=11)
AMB = reverb(AMB, ir, 0.10)

# ---------------------------------------------------------------- score + automation
M = np.zeros((L, 2))
mus, sr = sf.read(f'{S}/music/music.wav', dtype='float64')
n = min(L, len(mus)); M[:n] = mus[:n]
ev = json.load(open(f'{S}/music/events.json'))
TD, TP = ev['T_DROP'], ev['T_PEAK']
tt = np.arange(L) / SR
TO = ev['T_OPEN']
# 1) the location sound steps back where the score leads, and forward where the real sound is the point
amb_keys = [(0, 0), (TO + 0.3, 0), (TO + 1.0, -4), (TD - 3.2, -4), (TD - 2.8, -2), (TD - 0.1, -2), (TD, 1.5), (TP - 0.3, 1.5), (TP + 0.2, -5),
            (TP + 4, -5), (TP + 4.4, 0), (T - 3.5, -1), (T, -3)]
aa = 10 ** (np.interp(tt, [k[0] for k in amb_keys], [k[1] for k in amb_keys]) / 20)
AMB *= aa[:, None]

# 2) score level set by *measurement in the band a phone speaker reproduces* (300 Hz - 6 kHz)
from scipy.signal import butter, sosfilt
_sos = butter(4, [300, 6000], btype='band', fs=SR, output='sos')
def band_db(x, a, b):
    y = sosfilt(_sos, x[ns(a):ns(b)].mean(1))
    return 20 * np.log10(np.sqrt((y ** 2).mean()) + 1e-9)
# (t0, t1, reference, target dB of score relative to the reference)
SECT = [(0.0, 3.85, 'amb', -5), (3.9, 4.6, 'amb', +2), (4.8, TO - 0.3, 'vo', -10), (TO, ev['T_VERT'], 'amb', +4),
        (ev['T_VERT'], ev['T_FORM'], 'amb', +3), (ev['T_FORM'], TD - 3.2, 'amb', +5), (TD - 2.9, TD - 0.1, 'vo', -10),
        (TD, TD + 3.0, 'amb', -14), (TD + 3.0, TP - 0.05, 'amb', -5), (TP, TP + 4.0, 'amb', +8),
        (TP + 4.3, TP + 7.3, 'amb', -3), (TP + 7.3, T - 0.9, 'amb', +3)]
gains = []
for a, b, ref, tgt in SECT:
    R = AMB if ref == 'amb' else VO
    g = tgt - (band_db(M, a, b) - band_db(R, a, b))
    gains.append((a, b, float(np.clip(g, -20, 24))))
    print(f'  score {a:5.1f}-{b:5.1f} vs {ref}: gain {g:+5.1f} dB')
keys = []
for a, b, g in gains:
    keys += [(a + 0.02, g), (b - 0.02, g)]
keys = sorted(keys)
kt = np.array([k[0] for k in keys]); kv = np.array([k[1] for k in keys])
auto = 10 ** (np.interp(tt, kt, kv) / 20)
auto = np.convolve(auto, np.ones(ns(0.12)) / ns(0.12), mode='same')  # soft corners
# voice-keyed ducking (only where an announcer line is actually speaking)
vmask = np.zeros(L)
for a, b in vo_win:
    vmask[ns(a):ns(b)] = 1
env = np.sqrt(np.convolve((VO ** 2).mean(1), np.ones(ns(0.05)) / ns(0.05), mode='same'))
act = np.clip((20 * np.log10(env + 1e-9) + 42) / 12, 0, 1) * vmask
k = ns(0.35)
act = np.convolve(act, np.ones(k) / k, mode='same')
duck = 10 ** (-3.0 * act / 20)
M *= (auto * duck)[:, None]

mix = AMB + VO + M
mix = np.stack([hp(mix[:, c], 25) for c in range(2)], 1)
# fade out tail to digital silence exactly at the end
e = np.ones(L); i0, i1 = ns(T - 0.9), ns(T)
e[i0:i1] = np.linspace(1, 0, i1 - i0) ** 1.4; e[i1:] = 0
mix *= e[:, None]


# ---------------------------------------------------------------- master: gentle glue compression + true-peak limiter + loudness
def compress(x, thr_db=-20, ratio=2.0, att=0.01, rel=0.25):
    lvl = np.sqrt(np.convolve((x ** 2).mean(1), np.ones(ns(0.02)) / ns(0.02), mode='same')) + 1e-9
    db = 20 * np.log10(lvl)
    over = np.maximum(db - thr_db, 0)
    gr = over * (1 - 1 / ratio)
    # smooth gain reduction (attack/release)
    g = np.zeros_like(gr); a_, r_ = np.exp(-1 / ns(att)), np.exp(-1 / ns(rel)); cur = 0.0
    for i in range(0, len(gr), 64):
        v = gr[i]
        cur = a_ ** 64 * cur + (1 - a_ ** 64) * v if v > cur else r_ ** 64 * cur + (1 - r_ ** 64) * v
        g[i:i + 64] = cur
    return x * 10 ** (-g / 20)[:, None]


def limiter(x, ceiling_db=-1.0, look=0.004, rel=0.08):
    c = 10 ** (ceiling_db / 20)
    up = resample_poly(x, 4, 1, axis=0)  # true-peak estimate
    pk = np.abs(up).max(1).reshape(-1, 4).max(1)[:len(x)]
    pk = np.pad(pk, (0, len(x) - len(pk)))
    need = np.minimum(1, c / np.maximum(pk, 1e-9))
    la = ns(look)
    from scipy.ndimage import minimum_filter1d
    need = minimum_filter1d(need, 2 * la + 1)
    g = np.ones_like(need); cur = 1.0; r_ = np.exp(-1 / ns(rel))
    for i in range(len(need)):
        v = need[i]
        cur = v if v < cur else r_ * cur + (1 - r_) * v
        g[i] = cur
    return x * g[:, None]


mix = compress(mix)
meter = pyln.Meter(SR)
lufs = meter.integrated_loudness(mix[:ns(T)])
mix *= 10 ** ((-14.0 - lufs) / 20)
mix = limiter(mix, -1.0)
lufs2 = meter.integrated_loudness(mix[:ns(T)])
sf.write(f'{S}/mix/final_audio.wav', mix[:ns(T)].astype(np.float32), SR, subtype='FLOAT')
sf.write(f'{S}/mix/stem_amb.wav', AMB[:ns(T)].astype(np.float32), SR)
sf.write(f'{S}/mix/stem_vo.wav', VO[:ns(T)].astype(np.float32), SR)
sf.write(f'{S}/mix/stem_music.wav', M[:ns(T)].astype(np.float32), SR)
print(f'loudness before {lufs:.1f} LUFS -> after {lufs2:.1f} LUFS, peak {20 * np.log10(np.abs(mix).max()):.2f} dBFS')
