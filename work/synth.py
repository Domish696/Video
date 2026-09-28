"""Small synthesis toolkit (48 kHz, numpy + numba) used to score the film."""
import numpy as np
from numba import njit
from scipy.signal import fftconvolve, butter, sosfilt

SR = 48000
RNG = np.random.default_rng(27092026)


def mtof(m):
    return 440.0 * 2 ** ((m - 69) / 12.0)


def ns(sec):
    return int(round(sec * SR))


# ------------------------------------------------------------------ oscillators
def saw(freq, n, phase0=0.0):
    """PolyBLEP band-limited saw. freq may be scalar or array (len n)."""
    f = np.broadcast_to(np.asarray(freq, np.float64), (n,)).astype(np.float64)
    dt = f / SR
    ph = (phase0 + np.cumsum(dt)) % 1.0
    y = 2 * ph - 1
    m1 = ph < dt
    t = ph[m1] / dt[m1]
    y[m1] -= t + t - t * t - 1
    m2 = ph > 1 - dt
    t = (ph[m2] - 1) / dt[m2]
    y[m2] -= t * t + t + t + 1
    return y


def sine(freq, n, phase0=0.0):
    f = np.broadcast_to(np.asarray(freq, np.float64), (n,))
    return np.sin(2 * np.pi * (phase0 + np.cumsum(f) / SR))


def env_adsr(n, a, d, s, r, hold=None):
    """hold = samples of note-on (default n - release)."""
    A, D, R = ns(a), ns(d), ns(r)
    hold = n - R if hold is None else hold
    e = np.zeros(n)
    t = np.arange(n)
    e = np.where(t < A, t / max(A, 1), e)
    dd = (t >= A) & (t < A + D)
    e = np.where(dd, 1 - (1 - s) * (t - A) / max(D, 1), e)
    e = np.where((t >= A + D) & (t < hold), s, e)
    lvl = np.interp(hold, [0, A, A + D, n], [0, 1, s, s]) if hold < A + D else s
    rr = t >= hold
    e = np.where(rr, lvl * np.clip(1 - (t - hold) / max(R, 1), 0, 1), e)
    return e


# ------------------------------------------------------------------ filters
@njit(cache=True)
def svf(x, cutoff, q, mode):
    """Chamberlin/Simper SVF with per-sample cutoff. mode 0=LP 1=BP 2=HP."""
    n = x.shape[0]
    y = np.zeros(n)
    ic1 = 0.0; ic2 = 0.0
    k = 1.0 / q
    for i in range(n):
        g = np.tan(np.pi * min(cutoff[i], 0.45 * 48000.0) / 48000.0)
        a1 = 1.0 / (1.0 + g * (g + k)); a2 = g * a1; a3 = g * a2
        v3 = x[i] - ic2
        v1 = a1 * ic1 + a2 * v3
        v2 = ic2 + a2 * ic1 + a3 * v3
        ic1 = 2 * v1 - ic1; ic2 = 2 * v2 - ic2
        if mode == 0:
            y[i] = v2
        elif mode == 1:
            y[i] = v1
        else:
            y[i] = x[i] - k * v1 - v2
    return y


def lp(x, fc, q=0.707):
    return svf(np.ascontiguousarray(x, np.float64), np.broadcast_to(np.asarray(fc, np.float64), x.shape).copy(), q, 0)


def hp(x, fc, q=0.707):
    return svf(np.ascontiguousarray(x, np.float64), np.broadcast_to(np.asarray(fc, np.float64), x.shape).copy(), q, 2)


def bp(x, fc, q=1.0):
    return svf(np.ascontiguousarray(x, np.float64), np.broadcast_to(np.asarray(fc, np.float64), x.shape).copy(), q, 1)


def sos_filter(x, kind, f, order=2):
    sos = butter(order, f, btype=kind, fs=SR, output='sos')
    return sosfilt(sos, x, axis=0)


# ------------------------------------------------------------------ instruments (return mono or stereo arrays)
def felt_piano(m, dur, vel=0.5, B=0.00032):
    f0 = mtof(m)
    n = ns(dur + 2.8)
    t = np.arange(n) / SR
    y = np.zeros(n)
    tau0 = 2.6 * (261.6 / f0) ** 0.35
    bright = 0.35 + 0.65 * vel
    for k in range(1, 18):
        fk = k * f0 * np.sqrt(1 + B * k * k)
        if fk > 9000:
            break
        amp = (1.0 / k ** 1.25) * np.exp(-(k - 1) * (0.55 - 0.35 * bright))
        tau = tau0 / (1 + 0.45 * (k - 1))
        for dtn in (-0.0008, 0.0008):
            y += 0.5 * amp * np.exp(-t / tau) * np.sin(2 * np.pi * fk * (1 + dtn) * t + RNG.uniform(0, 6.28))
    # soft felt hammer thump
    thump = RNG.standard_normal(ns(0.012)) * np.exp(-np.arange(ns(0.012)) / ns(0.003))
    y[:len(thump)] += 0.05 * vel * lp(thump, 900)
    # damper
    rel = ns(dur)
    y[rel:] *= np.exp(-np.arange(n - rel) / ns(0.28))
    a = np.minimum(1, np.arange(n) / ns(0.004))
    return lp(y * a * vel, 2500 + 3000 * vel)


def supersaw(m, dur, amp=0.2, detune=13.0, voices=7, cutoff=2400, attack=0.45, release=1.0, vib=0.0045):
    f0 = mtof(m)
    n = ns(dur + release)
    L = np.zeros(n); R = np.zeros(n)
    t = np.arange(n) / SR
    for v in range(voices):
        c = (v - (voices - 1) / 2) / ((voices - 1) / 2) * detune
        fr = f0 * 2 ** (c / 1200) * (1 + vib * np.sin(2 * np.pi * (4.6 + 0.3 * v) * t + v))
        s = saw(fr, n, RNG.uniform())
        pan = (v / (voices - 1)) * 2 - 1
        L += s * np.sqrt((1 - pan) / 2); R += s * np.sqrt((1 + pan) / 2)
    e = env_adsr(n, attack, 0.3, 0.9, release, hold=ns(dur))
    fc = cutoff * (0.55 + 0.45 * np.minimum(1, t / max(attack * 1.5, 0.05)))
    L = lp(L, fc, 0.6) * e; R = lp(R, fc, 0.6) * e
    return np.stack([L, R], 1) * amp / np.sqrt(voices)


def brass(m, dur, amp=0.25, swell=0.5, peak=1800):
    f0 = mtof(m)
    n = ns(dur + 0.6)
    t = np.arange(n) / SR
    fr = f0 * (1 + 0.003 * np.sin(2 * np.pi * 5.2 * t) * np.minimum(1, t / 0.6))
    s = 0.6 * saw(fr, n) + 0.4 * saw(fr * 1.003, n, 0.3)
    e = env_adsr(n, swell, 0.4, 0.8, 0.5, hold=ns(dur))
    fc = 250 + peak * e ** 1.5
    y = lp(s, fc, 0.9) * e
    y = np.tanh(y * 1.6) / 1.6
    return y * amp


def pluck(m, dur=0.25, amp=0.2, bright=2600):
    f0 = mtof(m)
    n = ns(dur + 0.3)
    t = np.arange(n) / SR
    s = 0.7 * saw(f0, n) + 0.3 * saw(f0 * 2.0005, n)
    fc = 180 + bright * np.exp(-t / 0.07)
    e = np.exp(-t / (dur * 0.6)) * np.minimum(1, t / 0.002)
    return lp(s, fc, 1.2) * e * amp


def sub(m, dur, amp=0.35, attack=0.05, release=0.4):
    f0 = mtof(m)
    n = ns(dur + release)
    y = sine(f0, n) + 0.12 * sine(2 * f0, n)
    e = env_adsr(n, attack, 0.1, 1.0, release, hold=ns(dur))
    return np.tanh(1.3 * y * e) * amp


def bass_harm(m, dur, amp=0.12, attack=0.01, release=0.2, cutoff=1500):
    """Audible-on-a-phone bass: saturated saw an octave up, so the bassline survives small speakers
    (they cannot reproduce the fundamental, but the ear reconstructs it from the harmonics)."""
    f0 = mtof(m + 12)
    n = ns(dur + release)
    t = np.arange(n) / SR
    s = saw(f0, n) + 0.5 * saw(f0 * 1.004, n, 0.37)
    e = env_adsr(n, attack, 0.15, 0.75, release, hold=ns(dur))
    y = lp(s, cutoff * (0.6 + 0.4 * np.exp(-t / 0.12)), 0.9) * e
    return np.tanh(2.2 * y) / 2.2 * amp


def crack(amp=0.5):
    """The mid/high 'snap' of an impact, the part a phone speaker actually plays."""
    n = ns(0.9); t = np.arange(n) / SR
    y = hp(noise(n), 1200, 0.7) * np.exp(-t / 0.09)
    y += bp(noise(n), 450, 1.2) * np.exp(-t / 0.18) * 0.8
    y += sine(180 * (1 + 0.6 * np.exp(-t / 0.02)), n) * np.exp(-t / 0.12) * 0.6
    return np.tanh(y * 1.5) * amp


def whoosh(dur=0.7, amp=0.25, peak=0.62):
    """Air moving past, peaking at `peak` * dur (placed so the peak lands on a cut)."""
    n = ns(dur); t = np.arange(n) / SR
    tp = peak * dur
    env = np.where(t < tp, (t / tp) ** 2.2, np.exp(-(t - tp) / (0.18 * dur)))
    fc = 500 * np.where(t < tp, 2 ** (3.2 * t / tp), 2 ** (3.2 - 1.5 * (t - tp) / (dur - tp + 1e-6)))
    y = bp(noise(n), fc, 1.6) * env
    return y * amp


def harp(m, amp=0.25):
    f0 = mtof(m)
    n = ns(2.5)
    t = np.arange(n) / SR
    y = np.zeros(n)
    for k in range(1, 9):
        y += np.exp(-t / (1.4 / k)) * np.sin(2 * np.pi * k * f0 * t) / k ** 1.6
    return y * np.minimum(1, t / 0.0015) * amp


# ------------------------------------------------------------------ percussion / fx
def noise(n):
    return RNG.standard_normal(n)


def kick(amp=0.8):
    n = ns(0.6); t = np.arange(n) / SR
    f = 44 + 120 * np.exp(-t / 0.035)
    y = sine(f, n) * np.exp(-t / 0.28)
    click = hp(noise(n), 3000) * np.exp(-t / 0.004) * 0.25
    return np.tanh((y + click) * 1.5) * amp


def taiko(amp=0.9, pitch=1.0):
    n = ns(1.6); t = np.arange(n) / SR
    f = (62 + 70 * np.exp(-t / 0.05)) * pitch
    body = sine(f, n) * np.exp(-t / 0.55)
    skin = bp(noise(n), 420 * pitch, 1.2) * np.exp(-t / 0.07) * 0.9
    return np.tanh((body + skin) * 1.2) * amp


def tom(amp=0.6, pitch=1.0):
    n = ns(0.8); t = np.arange(n) / SR
    f = (110 + 90 * np.exp(-t / 0.04)) * pitch
    return (sine(f, n) * np.exp(-t / 0.22) + bp(noise(n), 900 * pitch, 1.5) * np.exp(-t / 0.03) * 0.5) * amp


def snare(amp=0.5):
    n = ns(0.5); t = np.arange(n) / SR
    y = bp(noise(n), 3200, 0.7) * np.exp(-t / 0.13) + 0.6 * sine(190 * (1 + 0.3 * np.exp(-t / 0.01)), n) * np.exp(-t / 0.06)
    return y * amp


def clap(amp=0.4):
    n = ns(0.5); t = np.arange(n) / SR
    e = np.zeros(n)
    for d in (0.0, 0.011, 0.022):
        e += np.exp(-np.maximum(t - d, 0) / 0.007) * (t >= d)
    e += 0.4 * np.exp(-t / 0.12)
    return bp(noise(n), 1500, 0.9) * e * amp


def hat(amp=0.15, open_=False):
    n = ns(0.4 if open_ else 0.08); t = np.arange(n) / SR
    return hp(noise(n), 7500, 0.8) * np.exp(-t / (0.12 if open_ else 0.025)) * amp


def shaker(amp=0.08):
    n = ns(0.12); t = np.arange(n) / SR
    return bp(noise(n), 6000, 0.7) * np.sin(np.pi * np.minimum(t / 0.1, 1)) ** 2 * amp


def cymbal(dur=2.5, amp=0.25):
    n = ns(dur); t = np.arange(n) / SR
    y = hp(noise(n), 5000, 0.6)
    for fm in (3150, 4270, 5820, 7310, 8890):
        y += 0.15 * np.sin(2 * np.pi * fm * t + RNG.uniform(0, 6))
    return y * np.exp(-t / (dur / 3.5)) * amp


def swell(dur=2.0, amp=0.3):
    """reverse cymbal ending at t=dur."""
    return cymbal(dur, amp)[::-1] * np.linspace(0, 1, ns(dur)) ** 1.5


def riser(dur=2.0, amp=0.25, f0=300, f1=7000):
    n = ns(dur); t = np.arange(n) / SR
    fc = f0 * (f1 / f0) ** (t / dur)
    y = bp(noise(n), fc, 2.5) * (t / dur) ** 2
    y += 0.25 * saw(220 * 2 ** (t / dur), n) * (t / dur) ** 3 * 0.3
    return y * amp


def boom(amp=1.0):
    n = ns(4.0); t = np.arange(n) / SR
    f = 28 + 40 * np.exp(-t / 0.35)
    y = sine(f, n) * np.exp(-t / 1.4)
    y += lp(noise(n), 180 + 600 * np.exp(-t / 0.05)) * np.exp(-t / 0.4) * 0.8
    y += taiko(0.7, 0.8)[:n] if ns(1.6) >= n else np.pad(taiko(0.7, 0.8), (0, n - ns(1.6)))
    return np.tanh(y * 1.8) * amp


def heartbeat(amp=0.5):
    n = ns(0.7); t = np.arange(n) / SR
    y = sine(48 + 20 * np.exp(-t / 0.03), n) * np.exp(-t / 0.12)
    t2 = np.maximum(t - 0.22, 0)
    y += 0.6 * sine(44 + 15 * np.exp(-t2 / 0.03), n) * np.exp(-t2 / 0.12) * (t >= 0.22)
    return y * amp


def tremolo_strings(m, dur, amp=0.12, rate=12.0):
    s = supersaw(m, dur, amp=1.0, detune=9, voices=5, cutoff=3200, attack=dur * 0.8, release=0.3)
    n = s.shape[0]; t = np.arange(n) / SR
    trem = 0.65 + 0.35 * np.sin(2 * np.pi * rate * t)
    return s * trem[:, None] * amp


# ------------------------------------------------------------------ reverb
def make_ir(t60=2.4, pre=0.02, damp=5500, early=True, seed=7):
    rng = np.random.default_rng(seed)
    n = ns(t60 * 1.2)
    t = np.arange(n) / SR
    env = np.exp(-6.9 * t / t60)
    ir = np.zeros((n + ns(pre), 2))
    for c in range(2):
        x = rng.standard_normal(n) * env
        # darker tail: progressive lowpass by crossfading two filtered copies
        dark = sos_filter(x, 'low', 1800)
        brt = sos_filter(x, 'low', damp)
        w = np.clip(t / (t60 * 0.6), 0, 1)
        ir[ns(pre):, c] = brt * (1 - w) + dark * w
        if early:
            for d, g in [(0.007, 0.5), (0.013, 0.35), (0.021, 0.3), (0.034, 0.22), (0.047, 0.18)]:
                ir[ns(pre) + ns(d + 0.002 * c), c] += g * (1 if rng.uniform() > 0.5 else -1)
    ir /= np.sqrt((ir ** 2).sum(0).mean())
    return ir


def reverb(x, ir, wet=0.3):
    if x.ndim == 1:
        x = np.stack([x, x], 1)
    y = np.stack([fftconvolve(x[:, c], ir[:, c])[:x.shape[0]] for c in range(2)], 1)
    return x * (1 - wet) + y * wet


def pan(x, p):
    """mono -> stereo constant power, p in [-1,1]."""
    a = (p + 1) * np.pi / 4
    return np.stack([x * np.cos(a), x * np.sin(a)], 1)
