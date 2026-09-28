import cv2, sys, numpy as np, glob
S=sys.argv[1]; name=sys.argv[2]; t0=float(sys.argv[3]); t1=float(sys.argv[4]); step=float(sys.argv[5]) if len(sys.argv)>5 else 0.25
f=[x for x in glob.glob(S+'/raw/*/*.mp4') if name in x][0]
cap=cv2.VideoCapture(f); fps=cap.get(5)
ims=[]; t=t0
while t<=t1:
    cap.set(1,int(round(t*fps))); ok,fr=cap.read()
    if not ok: break
    h,w=fr.shape[:2]; tw=240 if w>h else 135
    im=cv2.resize(fr,(tw,int(h*tw/w)),interpolation=cv2.INTER_AREA)
    cv2.putText(im,f'{t:.2f}',(3,14),0,0.45,(0,0,0),3); cv2.putText(im,f'{t:.2f}',(3,14),0,0.45,(0,255,255),1)
    ims.append(im); t+=step
cols=8 if ims[0].shape[1]>ims[0].shape[0] else 12
th,tw=ims[0].shape[:2]; rows=(len(ims)+cols-1)//cols
sh=np.zeros((rows*th,cols*tw,3),np.uint8)
for i,im in enumerate(ims):
    r,c=divmod(i,cols); sh[r*th:r*th+th,c*tw:c*tw+tw]=im
cv2.imwrite(f'{S}/an/strip_{name}_{t0:g}.jpg',sh,[cv2.IMWRITE_JPEG_QUALITY,88])
