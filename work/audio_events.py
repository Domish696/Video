import numpy as np, librosa, os
S = os.environ['S']


def env(name, a, b, hop=0.02):
    y, sr = librosa.load(f'{S}/audio/{name}.wav', sr=16000, offset=a, duration=b - a)
    r = librosa.feature.rms(y=y, frame_length=int(0.05 * sr), hop_length=int(hop * sr))[0]
    return 20 * np.log10(r + 1e-6), y, sr


# KC-390 overhead swoosh: broadband peak (HP filtered) in WA0035
db, y, sr = env('WA0035', 5.0, 9.5)
yh = librosa.effects.preemphasis(y, coef=0.97)
r = librosa.feature.rms(y=yh, frame_length=800, hop_length=320)[0]
t = 5.0 + np.arange(len(r)) * 0.02
print('KC390 HF-energy peak at', t[np.argmax(r)], ' top5', np.round(t[np.argsort(r)[-5:]], 2))
# speech onsets in WA0082 narration (counting) 2.5-7.0: band 300-3000 Hz onset strength
for name, a, b in [('WA0082', 2.4, 7.2), ('WA0046', 16.5, 24.0), ('WA0098', 1.0, 18.0), ('WA0052', 13.5, 24.8), ('WA0034', 9.5, 11.8)]:
    y, sr = librosa.load(f'{S}/audio/{name}.wav', sr=16000, offset=a, duration=b - a)
    S_ = np.abs(librosa.stft(y, n_fft=512, hop_length=160))
    f = librosa.fft_frequencies(sr=sr, n_fft=512)
    band = S_[(f > 300) & (f < 3000)].sum(0)
    on = librosa.onset.onset_detect(onset_envelope=librosa.onset.onset_strength(S=librosa.amplitude_to_db(S_[(f > 300) & (f < 3000)]), sr=sr, hop_length=160), sr=sr, hop_length=160, units='time', backtrack=False)
    lv = 20 * np.log10(band + 1e-6)
    tt = a + np.arange(len(lv)) * 0.01
    # print a coarse loudness profile (0.25 s bins)
    bins = np.arange(a, b, 0.25)
    prof = [lv[(tt >= x) & (tt < x + 0.25)].mean() for x in bins]
    print(name, 'onsets:', np.round(a + on, 2))
    print('   prof', ' '.join(f'{x:.2f}:{p:.0f}' for x, p in zip(bins, prof)))
