"""AI source separation (Demucs htdemucs, two stems: voice / everything else) for every audio range the mix uses."""
import os, sys, json, subprocess
import numpy as np, soundfile as sf
sys.path.insert(0, os.path.dirname(__file__))
from edl import timeline, FPS_OUT

S = os.environ['S']
OUT = S + '/sep'
tl, T = timeline()
need = {}
for s, t, n in tl:
    a, b = s['t_in'] - 1.5, s['t_in'] + n / FPS_OUT + 3.8
    need.setdefault(s['src'], []).append((a, b))
need.setdefault('WA0052', []).append((13.0, 25.5))   # announcer: opening line
need.setdefault('WA0098', []).append((1.5, 6.5))     # announcer: "vamos ouvir os motores"
jobs = []
for name, rngs in need.items():
    rngs.sort()
    merged = []
    for a, b in rngs:
        a = max(0.0, a)
        if merged and a <= merged[-1][1] + 0.5:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    for a, b in merged:
        jobs.append((name, round(a, 2), round(b, 2)))
index = []
for name, a, b in jobs:
    tag = f'{name}_{a:.2f}_{b:.2f}'
    wav = f'{OUT}/in/{tag}.wav'
    os.makedirs(f'{OUT}/in', exist_ok=True)
    if not os.path.exists(f'{OUT}/htdemucs/{tag}/vocals.wav'):
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', str(a), '-t', str(b - a), '-i', f'{S}/audio48/{name}.wav', '-ac', '2', '-ar', '44100', wav], check=True)
        subprocess.run([sys.executable, '-m', 'demucs', '-n', 'htdemucs', '--two-stems=vocals', '-j', '1', '-d', 'cpu', '-o', OUT, wav], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    index.append(dict(name=name, a=a, b=b, voice=f'{OUT}/htdemucs/{tag}/vocals.wav', rest=f'{OUT}/htdemucs/{tag}/no_vocals.wav'))
    print('separated', tag, flush=True)
json.dump(index, open(f'{OUT}/index.json', 'w'), indent=1)
