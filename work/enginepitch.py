import numpy as np, librosa, os
S = os.environ['S']
segs = [('WA0037', 0, 4.7), ('WA0038', 0, 6.5), ('WA0082', 1, 7), ('WA0078', 4, 8), ('WA0044', 12, 19),
        ('WA0046', 16, 24), ('WA0070', 8, 16), ('WA0102', 10, 16), ('WA0060', 1, 7), ('WA0064', 5, 9.9), ('WA0035', 3, 9)]
tot = np.zeros(12)
names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
for n, a, b in segs:
    y, sr = librosa.load(f'{S}/audio/{n}.wav', sr=16000, offset=a, duration=b - a)
    yh = librosa.effects.harmonic(y, margin=3)
    C = librosa.feature.chroma_cqt(y=yh, sr=sr, hop_length=512, fmin=librosa.note_to_hz('C2'), n_octaves=6)
    c = C.mean(1)
    tot += c / c.sum()
    Sx = np.abs(librosa.stft(yh, n_fft=8192, hop_length=2048)).mean(1)
    f = np.fft.rfftfreq(8192, 1 / sr)
    m = (f > 60) & (f < 2000)
    idx = np.argsort(Sx[m])[-6:][::-1]
    pk = [(round(f[m][i], 1), librosa.hz_to_note(f[m][i])) for i in idx]
    print(n, 'chroma top:', [names[i] for i in np.argsort(c)[-3:][::-1]], 'peaks:', pk)
print('TOTAL', [(names[i], round(tot[i] / tot.sum(), 3)) for i in np.argsort(tot)[::-1]])
