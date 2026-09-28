import cv2, numpy as np
KER=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(45,45))
def ground_mask(g):
    h,w=g.shape
    s=cv2.resize(g,(w//4,h//4),interpolation=cv2.INTER_AREA).astype(np.float32)
    mu=cv2.blur(s,(5,5)); sq=cv2.blur(s*s,(5,5)); sd=np.sqrt(np.maximum(sq-mu*mu,0))
    tex=((sd>9)|(s<70)).astype(np.uint8)
    tex=cv2.morphologyEx(tex,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    n,lab,st,_=cv2.connectedComponentsWithStats(tex,8)
    gm=np.zeros_like(tex)
    hh=tex.shape[0]
    for i in range(1,n):
        x,y,ww,h2,a=st[i]
        if y+h2>=hh-1 and a>30: gm[lab==i]=1
    gm=cv2.dilate(gm,np.ones((5,5),np.uint8))
    return cv2.resize(gm,(w,h),interpolation=cv2.INTER_NEAREST)
def detect(frame, rel=0.28, minarea=5):
    h,w=frame.shape[:2]
    g=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    small=cv2.resize(g,(w//2,h//2),interpolation=cv2.INTER_AREA)
    clo=cv2.morphologyEx(small,cv2.MORPH_CLOSE,KER)
    clo=cv2.GaussianBlur(clo,(0,0),3)
    clo=cv2.resize(clo,(w,h),interpolation=cv2.INTER_LINEAR).astype(np.float32)
    gf=g.astype(np.float32)
    d=(clo-gf)/np.maximum(clo,1)
    gm=ground_mask(g)
    m=((d>rel)&(clo>95)&(gm==0)).astype(np.uint8)
    m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
    n,lab,st,cen=cv2.connectedComponentsWithStats(m,8)
    out=[]
    for i in range(1,n):
        x,y,ww,hh,a=st[i]
        if a<minarea: continue
        region=(lab[y:y+hh,x:x+ww]==i)
        dd=d[y:y+hh,x:x+ww]
        mx=float(dd[region].max())
        # separate dark cores from smoke
        cth=max(0.42,0.72*mx)
        if mx<0.40: continue   # nothing dark enough -> smoke only
        core=((dd>cth)&region).astype(np.uint8)
        core=cv2.dilate(core,np.ones((3,3),np.uint8))
        n2,l2,s2,c2=cv2.connectedComponentsWithStats(core,8)
        for j in range(1,n2):
            xx,yy,w2,h2,a2=s2[j]
            if a2<4: continue
            X,Y=x+xx,y+yy
            if Y+h2>=h-2: continue
            dk=float(dd[l2==j].mean())
            out.append(dict(cx=float(x+c2[j][0]),cy=float(y+c2[j][1]),x=int(X),y=int(Y),w=int(w2),h=int(h2),a=int(a2),dark=dk,score=dk*a2))
    return out
