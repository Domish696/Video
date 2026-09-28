import numpy as np, librosa, glob, os, itertools, scipy.signal as ss
S=os.environ['S']
fs=sorted(glob.glob(S+'/audio/*.wav'))
feats={}
for f in fs:
    y,sr=librosa.load(f,sr=16000)
    M=librosa.feature.melspectrogram(y=y,sr=sr,n_fft=1024,hop_length=320,n_mels=24,fmin=100,fmax=5000)  # 20ms hop
    L=np.log(M+1e-6)
    # spectral flux-ish: positive diff
    e=np.log(M.sum(0)+1e-6)
    k=50
    eh=e-np.convolve(e,np.ones(k)/k,mode='same')
    eh=eh[k:-k] if len(eh)>3*k else eh
    feats[os.path.basename(f)[:-4]]=eh
def ncc(a,b,minov=200):
    # pearson corr for every lag; lag = start of b relative to a (frames)
    best=(-1,0,0)
    vals=[]
    for lag in range(-(len(b)-minov), len(a)-minov+1):
        a0=max(0,lag); b0=max(0,-lag)
        n=min(len(a)-a0,len(b)-b0)
        if n<minov: vals.append(0); continue
        x=a[a0:a0+n]; y=b[b0:b0+n]
        r=np.corrcoef(x,y)[0,1]
        vals.append(r)
        if r>best[0]: best=(r,lag,n)
    vals=np.array(vals)
    z=(best[0]-vals.mean())/(vals.std()+1e-9)
    return best,z
names=list(feats)
res=[]
for a,b in itertools.combinations(names,2):
    (r,lag,n),z=ncc(feats[a],feats[b])
    res.append((r,z,a,b,lag*0.02,n*0.02))
res.sort(reverse=True)
for x in res[:30]:
    print(f'r={x[0]:.2f} z={x[1]:5.1f} {x[2]:>20s} {x[3]:>20s}  b starts at a+{x[4]:7.2f}s  overlap {x[5]:.1f}s')
