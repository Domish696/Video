"""Original score for the Campo de Marte film. A major, 120 BPM (beat 0.5 s, bar 2 s).
The key comes from the aircraft themselves: the A-29 propeller hum measured in the clips sits at
~110 Hz (A2) with a secondary D3, so the score is built on an A pedal that the real engines reinforce.
Writes stems to $S/music/*.npy (48 kHz stereo float32) and a mixdown music.wav."""
import os, sys, json
import numpy as np, soundfile as sf
sys.path.insert(0, os.path.dirname(__file__))
from synth import *
from synth import bass_harm, crack, whoosh
from edl import timeline

S = os.environ['S']
OUT = S + '/music'
os.makedirs(OUT, exist_ok=True)

tl, TOTAL = timeline()
start = {s['id']: t for s, t, n in tl}
T_OPEN, T_SOLO, T_PAIR, T_VERT, T_CROSS, T_FORM, T_DROP = (start[k] for k in ('S05', 'S06', 'S07', 'S08', 'S09', 'S10', 'S13'))
T_PEAK = T_DROP + 5.0
T_AFTER = start['S15']
T_END = TOTAL
s06 = [s for s, t, n in tl if s['id'] == 'S06'][0]
COUNTS = [T_SOLO + (x - s06['t_in']) for x in (3.35, 4.15, 5.70)]  # narrator: "um", "dois", "três"

L = ns(T_END + 4.0)
BUS = {k: np.zeros((L, 2)) for k in ('pad', 'piano', 'lead', 'bass', 'drums', 'fx', 'pulse')}


def put(bus, x, t, gain=1.0, p=0.0):
    if x.ndim == 1:
        x = pan(x, p)
    i = ns(t)
    if i >= L:
        return
    j = min(L, i + len(x))
    BUS[bus][i:j] += x[:j - i] * gain


V = {
    'A5': (33, [45, 52, 57, 64]),
    'Aadd9': (33, [57, 64, 69, 71, 73]),
    'A': (33, [57, 61, 64, 69]),
    'A/C#': (37, [57, 61, 64, 69]),
    'E/G#': (44, [56, 59, 64, 68]),
    'E': (40, [56, 59, 64, 68]),
    'Esus4': (40, [57, 59, 64, 69]),
    'F#m': (42, [57, 61, 66, 69]),
    'F#m7': (42, [57, 61, 64, 66]),
    'D': (38, [57, 62, 66, 69]),
    'Dadd9': (38, [57, 62, 64, 66]),
}
D = T_DROP
CH = [(4.0, 6.0, 'Aadd9'), (6.0, 8.0, 'E/G#'), (8.0, 10.0, 'F#m7'), (10.0, 12.0, 'Dadd9'), (12.0, 13.0, 'A/C#'), (13.0, 14.0, 'Esus4'),
      (14.0, 16.0, 'D'), (16.0, 18.0, 'A'), (18.0, 20.0, 'E'), (20.0, 22.0, 'F#m'), (22.0, 24.0, 'D'), (24.0, 26.0, 'A/C#'),
      (26.0, 28.0, 'Esus4'), (28.0, 29.5, 'E'), (29.5, 31.0, 'F#m'), (31.0, 33.0, 'D'),
      (33.0, 35.0, 'A'), (35.0, 37.0, 'E/G#'), (37.0, 39.0, 'F#m'), (39.0, 41.0, 'D'), (41.0, D, 'Esus4'),
      (T_PEAK, T_PEAK + 2, 'D'), (T_PEAK + 2, T_PEAK + 4, 'A'), (T_PEAK + 4, T_PEAK + 6, 'E'), (T_PEAK + 6, T_PEAK + 8, 'F#m'),
      (T_PEAK + 8, T_PEAK + 9, 'D'), (T_PEAK + 9, T_END + 1.5, 'Aadd9')]


def section(t):
    if t < 4.0: return 'intro'
    if t < T_OPEN: return 'theme'
    if t < T_VERT: return 'open'
    if t < T_FORM: return 'build'
    if t < D: return 'climax'
    if t < T_PEAK: return 'drop'
    if t < T_PEAK + 4: return 'chorus'
    return 'outro'


# ------------------------------------------------------------------ INTRO: drone + heartbeat + swell into the KC-390 overhead
dr = sub(33, 4.3, amp=0.30, attack=2.2, release=0.6)
put('bass', dr, 0.0)
put('pad', supersaw(45, 4.2, amp=0.10, cutoff=420, attack=2.5, release=1.2, detune=8), 0.0)
put('pad', supersaw(52, 4.2, amp=0.07, cutoff=520, attack=2.5, release=1.2, detune=8), 0.0)
put('fx', supersaw(81, 3.6, amp=0.020, cutoff=6000, attack=2.0, release=1.5, voices=5), 0.4)  # sky shimmer
for k, t in enumerate((1.0, 2.0, 3.0)):
    put('drums', heartbeat(0.30 + 0.12 * k), t)
put('fx', swell(2.4, 0.40), 4.0 - 2.4)
put('fx', riser(3.0, 0.22, 200, 6000), 1.0)
# IMPACT at the overhead moment
put('drums', boom(0.95), 4.0)
put('drums', crack(0.55), 4.0)
put('fx', cymbal(3.5, 0.36), 4.0, p=0.1)
for m in (45, 52, 57):
    put('lead', brass(m, 1.2, amp=0.10, swell=0.03, peak=1500), 4.0, p=(m - 52) / 20)

# ------------------------------------------------------------------ PADS / BASS across the chord chart
for t0, t1, name in CH:
    b, notes = V[name]
    sec = section(t0)
    if sec in ('theme',):
        amp, cut, att = 0.070, 2400, 0.9
    elif sec == 'open':
        amp, cut, att = 0.095, 3600, 0.45
    elif sec == 'build':
        amp, cut, att = 0.100, 4200, 0.35
    elif sec == 'climax':
        amp, cut, att = 0.110, 5000, 0.25
    elif sec == 'chorus':
        amp, cut, att = 0.125, 5800, 0.06
    else:
        amp, cut, att = 0.080, 3000, 0.6
    dur = t1 - t0
    for i, m in enumerate(notes):
        put('pad', supersaw(m, dur, amp=amp, cutoff=cut, attack=att, release=0.9), t0, p=(i - 1.5) / 3)
    # bass
    if sec == 'theme':
        put('bass', sub(b + 12, dur, amp=0.16, attack=0.3, release=0.8), t0)
        put('bass', bass_harm(b, dur, amp=0.05, attack=0.3, release=0.8, cutoff=900), t0)
    elif sec in ('open', 'build'):
        bar = t0
        while bar < t1 - 1e-6:
            put('bass', sub(b + 12, min(1.1, t1 - bar), amp=0.22, attack=0.02, release=0.25), bar)
            put('bass', bass_harm(b, min(1.1, t1 - bar), amp=0.11, release=0.2), bar)
            if bar + 1.25 < t1:
                put('bass', sub(b + 12, 0.6, amp=0.18, attack=0.02, release=0.2), bar + 1.25)
                put('bass', bass_harm(b, 0.6, amp=0.09, release=0.15), bar + 1.25)
            bar += 2.0
    elif sec in ('climax', 'chorus'):
        k = t0
        while k < t1 - 1e-6:  # driving eighths, octave jumps
            m = b + 12 if int(round((k - t0) / 0.25)) % 4 != 2 else b + 24
            put('bass', sub(m, 0.22, amp=0.20, attack=0.005, release=0.06), k)
            put('bass', bass_harm(m - 12, 0.2, amp=0.10, attack=0.004, release=0.05, cutoff=1900), k)
            k += 0.25
        put('bass', sub(b, t1 - t0, amp=0.16, attack=0.02, release=0.3), t0)
    elif sec == 'outro':
        put('bass', sub(b + 12, dur, amp=0.18, attack=0.2, release=1.5), t0)
        put('bass', bass_harm(b, dur, amp=0.05, attack=0.2, release=1.5, cutoff=900), t0)

# ------------------------------------------------------------------ PIANO: the "memory" theme under the announcer
mel_theme = [(4.5, 76, 1.0), (5.5, 73, 0.5), (6.0, 71, 1.0), (7.0, 68, 0.5), (7.5, 69, 0.5), (8.0, 69, 1.0), (9.0, 73, 0.5), (9.5, 76, 0.5),
             (10.0, 78, 1.5), (11.5, 76, 0.5), (12.0, 76, 0.5), (12.5, 73, 0.5), (13.0, 71, 1.0)]
for t, m, d in mel_theme:
    put('piano', felt_piano(m, d, vel=0.42 + 0.05 * RNG.standard_normal()), t + RNG.normal(0, 0.006), p=0.15)
for t0, t1, name in CH:
    if not (4.0 <= t0 < T_OPEN):
        continue
    b, notes = V[name]
    arp = [b + 12, notes[0], notes[1], notes[2]]
    k = 0; t = t0
    while t < t1 - 1e-6:
        put('piano', felt_piano(arp[k % 4], 0.45, vel=0.22 + 0.04 * RNG.standard_normal()), t + RNG.normal(0, 0.008), p=-0.2)
        k += 1; t += 0.5
# outro piano: echo of the motif, then a last bell-like A
mel_out = [(T_PEAK + 4.0, 71, 1.0), (T_PEAK + 5.0, 76, 1.0), (T_PEAK + 6.0, 73, 1.5), (T_PEAK + 7.5, 69, 0.5), (T_PEAK + 8.0, 78, 1.0),
           (T_PEAK + 9.0, 76, 2.0), (T_PEAK + 10.5, 81, 2.5)]
for t, m, d in mel_out:
    put('piano', felt_piano(m, d, vel=0.40), t, p=0.15)
for t0, t1, name in CH:
    if t0 < T_PEAK + 4.0:
        continue
    b, notes = V[name]
    arp = [b + 12, notes[0], notes[1], notes[2], notes[1], notes[0]]
    k = 0; t = t0
    while t < min(t1, T_END - 0.5) - 1e-6:
        put('piano', felt_piano(arp[k % len(arp)], 0.45, vel=0.20), t + RNG.normal(0, 0.008), p=-0.2)
        k += 1; t += 0.5

# ------------------------------------------------------------------ OPENING: the show starts
put('drums', taiko(0.75), T_OPEN)
put('drums', crack(0.35), T_OPEN)
put('fx', cymbal(3.0, 0.26), T_OPEN, p=-0.2)
put('fx', swell(1.2, 0.30), T_OPEN - 1.2)
for m in (50, 57):
    put('lead', brass(m, 1.8, amp=0.08, swell=0.35, peak=1400), T_OPEN)
# pulse: eighth-note plucks (sidechained later by kicks)
for t0, t1, name in CH:
    if t0 < 12.0 or t0 >= D:
        continue
    b, notes = V[name]
    seq = [notes[0], notes[2], notes[1] + 12, notes[2]]
    t = t0; k = 0
    while t < t1 - 1e-6:
        amp = 0.06 if t0 < T_OPEN else 0.13
        put('pulse', pluck(seq[k % 4], 0.22, amp=amp * (1.25 if k % 2 == 0 else 1.0), bright=2600 if t0 < T_OPEN else 4400), t, p=0.35 if k % 2 else -0.35)
        k += 1; t += 0.25
# theme statement on strings+horns
theme = [(0.0, 78, 1.0), (1.0, 76, 0.5), (1.5, 74, 0.5), (2.0, 73, 1.5), (3.5, 76, 0.5), (4.0, 71, 1.0), (5.0, 76, 1.0), (6.0, 73, 2.0)]
for dt, m, d in theme:
    put('lead', supersaw(m, d, amp=0.075, cutoff=4800, attack=0.10, release=0.5, voices=5, detune=8), T_OPEN + dt)
    put('fx', harp(m + 12, 0.07), T_OPEN + dt)
    put('lead', brass(m - 12, d, amp=0.05, swell=0.2, peak=1200), T_OPEN + dt)
# the narrator counts the four-point roll -> harp answers
for t, m in zip(COUNTS, (69, 73, 76)):
    put('fx', harp(m, 0.30), t, p=0.3)
    put('fx', harp(m + 12, 0.08), t + 0.004, p=-0.3)
# percussion for the opening
bar = T_OPEN
while bar < T_VERT - 1e-6:
    put('drums', taiko(0.45), bar)
    bar += 2.0
t = 18.0
while t < T_VERT - 1e-6:
    put('drums', shaker(0.05 + 0.02 * (int(t * 2) % 2)), t)
    t += 0.25
t = T_PAIR
while t < T_VERT - 1e-6:
    put('drums', kick(0.45), t)
    t += 1.0

# ------------------------------------------------------------------ BUILD: vertical climb + crossing
climb = [52, 57, 59, 64, 69, 71, 76, 81, 83, 88]
for i, m in enumerate(climb * 1):
    put('fx', harp(m, 0.12 + 0.01 * i), T_VERT + 0.2 * i, p=-0.5 + i / 9)
put('lead', supersaw(64, 3.0, amp=0.05, cutoff=3500, attack=1.5, release=0.8, voices=5), T_VERT)
put('lead', supersaw(71, 3.0, amp=0.04, cutoff=3500, attack=1.5, release=0.8, voices=5), T_VERT)
t = T_VERT
while t < T_FORM - 1e-6:
    put('drums', kick(0.55), t)
    if abs(((t - T_VERT) % 2.0) - 1.0) < 1e-6:
        put('drums', snare(0.30), t); put('drums', clap(0.2), t)
    t += 1.0
t = T_VERT
while t < T_FORM - 1e-6:
    put('drums', hat(0.07), t)
    t += 0.25
put('drums', taiko(0.7), T_CROSS)
put('fx', cymbal(2.0, 0.12), T_CROSS, p=0.3)
t = T_FORM - 1.5
k = 0
while t < T_FORM - 1e-6:
    put('drums', tom(0.25 + 0.05 * k, 1.0 + 0.08 * (k % 3)), t, p=-0.4 + 0.1 * k)
    k += 1; t += 0.125
put('fx', riser(2.0, 0.26), T_FORM - 2.0)
put('fx', swell(1.5, 0.34), T_FORM - 1.5)

# ------------------------------------------------------------------ CLIMAX: formations grow in the frame
put('drums', boom(0.55), T_FORM)
put('drums', crack(0.45), T_FORM)
put('fx', cymbal(3.0, 0.34), T_FORM)
bar = T_FORM
while bar < 41.0 - 1e-6:
    for off, what in ((0.0, 'k'), (0.75, 'k'), (1.0, 's'), (1.5, 'k')):
        tt = bar + off
        if tt >= 41.0:
            break
        if what == 'k':
            put('drums', kick(0.6), tt)
        else:
            put('drums', snare(0.34), tt); put('drums', clap(0.22), tt)
    put('drums', taiko(0.55), bar)
    bar += 2.0
t = T_FORM
while t < 41.0 - 1e-6:
    put('drums', hat(0.05 + 0.02 * (int(round(t * 4)) % 2)), t, p=0.25)
    t += 0.125
chorus_mel = [(0.0, 81, 1.0), (1.0, 78, 0.5), (1.5, 76, 0.5), (2.0, 76, 1.5), (3.5, 73, 0.5), (4.0, 71, 1.0), (5.0, 76, 1.0),
              (6.0, 73, 1.0), (7.0, 76, 0.5), (7.5, 81, 0.5)]
for dt, m, d in chorus_mel:
    put('lead', supersaw(m - 12, d, amp=0.080, cutoff=5200, attack=0.08, release=0.5, voices=5, detune=9), T_FORM + dt)
    put('lead', supersaw(m, d, amp=0.035, cutoff=6000, attack=0.08, release=0.5, voices=5, detune=9), T_FORM + dt)
    put('lead', brass(m - 24, d, amp=0.055, swell=0.15, peak=1500), T_FORM + dt)
for t0, t1, name in CH:
    if T_FORM <= t0 < 41.0:
        b, notes = V[name]
        put('lead', brass(b + 24, t1 - t0 - 0.1, amp=0.06, swell=0.5, peak=1600), t0)
# the announcer asks to listen to the engines: pull the music back, inhale, silence
put('pad', supersaw(64, D - 41.0, amp=0.05, cutoff=900, attack=0.3, release=0.25), 41.0)
put('fx', swell(1.4, 0.18), D - 1.4)
put('drums', taiko(0.4), 41.0)

# ------------------------------------------------------------------ DROP: only a whisper of score under the real engines
put('bass', sub(33, T_PEAK - D, amp=0.07, attack=1.0, release=0.2), D)
put('pad', tremolo_strings(76, T_PEAK - D - 1.2, amp=0.07, rate=13), D + 1.2)
put('pad', tremolo_strings(81, T_PEAK - D - 1.8, amp=0.05, rate=13), D + 1.8)
put('fx', swell(2.2, 0.40), T_PEAK - 2.2)
put('fx', riser(2.5, 0.22, 300, 8000), T_PEAK - 2.5)

# ------------------------------------------------------------------ CHORUS: back in at the closest pass
put('drums', boom(1.0), T_PEAK)
put('drums', crack(0.65), T_PEAK)
put('fx', cymbal(4.0, 0.42), T_PEAK)
for m in (50, 57, 62):
    put('lead', brass(m, 1.9, amp=0.09, swell=0.04, peak=1900), T_PEAK, p=(m - 57) / 15)
for dt, m, d in chorus_mel[:7]:
    put('lead', supersaw(m, d, amp=0.090, cutoff=6000, attack=0.05, release=0.6, voices=7, detune=10), T_PEAK + dt)
    put('fx', harp(m + 12, 0.08), T_PEAK + dt)
    put('lead', brass(m - 12, d, amp=0.06, swell=0.1, peak=1700), T_PEAK + dt)
bar = T_PEAK
while bar < T_PEAK + 4.0 - 1e-6:
    for off, what in ((0.0, 'k'), (0.75, 'k'), (1.0, 's'), (1.5, 'k')):
        tt = bar + off
        if what == 'k':
            put('drums', kick(0.65), tt)
        else:
            put('drums', snare(0.36), tt); put('drums', clap(0.24), tt)
    put('drums', taiko(0.65), bar)
    bar += 2.0
t = T_PEAK
while t < T_PEAK + 4.0 - 1e-6:
    put('drums', hat(0.06), t, p=0.25)
    t += 0.125
put('drums', taiko(0.5), T_PEAK + 4.0)
put('fx', cymbal(3.0, 0.12), T_PEAK + 4.0)
# final chord
put('fx', cymbal(3.0, 0.07), T_PEAK + 9.0, p=-0.3)

# ------------------------------------------------------------------ cut accents: make the edit audible
cut_times = [t for s, t, n in tl][1:]
for s, t, n in tl[1:]:
    sid = s['id']
    if t in (T_OPEN, T_FORM, T_PEAK, D):
        continue  # these already carry big hits (or, for the drop, silence)
    if sid in ('S02',):
        put('fx', whoosh(1.0, 0.30, 0.55), t - 0.55, p=0.2)      # the KC-390 tears past into the crowd
    elif sid in ('S06', 'S07', 'S08', 'S09', 'S11', 'S12', 'S14', 'S18'):
        put('fx', whoosh(0.6, 0.20, 0.7), t - 0.42, p=(-0.3 if int(t * 2) % 2 else 0.3))
        put('drums', taiko(0.40), t)
    elif sid in ('S15', 'S16', 'S17'):
        put('fx', harp(76 if sid == 'S15' else (73 if sid == 'S16' else 69), 0.12), t)
# ------------------------------------------------------------------ sidechain duck (pads/pulse breathe with the kicks)
hits = []
for tt in np.arange(T_PAIR, T_VERT, 1.0): hits.append(tt)
for tt in np.arange(T_VERT, T_FORM, 1.0): hits.append(tt)
bar = T_FORM
while bar < 41.0:
    hits += [bar, bar + 0.75, bar + 1.5]; bar += 2.0
bar = T_PEAK
while bar < T_PEAK + 4.0:
    hits += [bar, bar + 0.75, bar + 1.5]; bar += 2.0
duck = np.ones(L)
for h in hits:
    i = ns(h); n = ns(0.35)
    if i + n < L:
        duck[i:i + n] = np.minimum(duck[i:i + n], 1 - 0.45 * np.exp(-np.arange(n) / ns(0.09)))
for b in ('pulse', 'pad'):
    BUS[b] *= duck[:, None]

# ------------------------------------------------------------------ reverbs + stems
ir_hall = make_ir(2.8, 0.025, 6000)
ir_room = make_ir(1.4, 0.012, 7000, seed=3)
wet = dict(pad=0.35, piano=0.30, lead=0.30, bass=0.0, drums=0.16, fx=0.38, pulse=0.22)
irs = dict(pad=ir_hall, piano=ir_hall, lead=ir_hall, drums=ir_room, fx=ir_hall, pulse=ir_room)
gain = dict(pad=1.1, piano=1.15, lead=1.3, bass=1.0, drums=1.25, fx=1.0, pulse=1.35)
mix = np.zeros((L, 2))
for b, x in BUS.items():
    y = reverb(x, irs[b], wet[b]) if wet[b] > 0 else x
    if b != 'bass':
        y = np.stack([hp(y[:, 0], 28), hp(y[:, 1], 28)], 1)
    y = y * gain[b]
    np.save(f'{OUT}/{b}.npy', y.astype(np.float32))
    mix += y
# tail fade after the end
e = np.ones(L); i0 = ns(T_END - 1.4); i1 = ns(T_END + 0.3)
e[i0:i1] = np.linspace(1, 0, i1 - i0) ** 1.6; e[i1:] = 0
mix *= e[:, None]
pk = np.abs(mix).max()
mix = mix / pk * 0.89
sf.write(f'{OUT}/music.wav', mix.astype(np.float32), SR)
json.dump(dict(T_OPEN=T_OPEN, T_SOLO=T_SOLO, T_VERT=T_VERT, T_CROSS=T_CROSS, T_FORM=T_FORM, T_DROP=D, T_PEAK=T_PEAK, COUNTS=COUNTS, END=T_END, peak=float(pk)),
          open(f'{OUT}/events.json', 'w'), indent=1)
print('music done', T_END, 'peak', pk)
