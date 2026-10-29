import cv2, numpy as np
from src.morphology import fill_holes
lab = cv2.cvtColor(cv2.imread('C:\\Blood Project\\TXL-PBC_Dataset-master\\TXL-PBC\\images\\test\\7f96e833a77559fb8fe76152b2b76660.png'), cv2.COLOR_BGR2LAB)
a = lab[:,:,1]
_, m = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
c = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
f = fill_holes(c)
print('Filled Mean:', np.mean(f))

boxes=[]
lines=open('C:\\Blood Project\\TXL-PBC_Dataset-master\\TXL-PBC\\labels\\test\\7f96e833a77559fb8fe76152b2b76660.txt').readlines()
for l in lines:
    p = l.strip().split()
    if not p: continue
    cx,cy,w,h=map(float,p[1:5])
    x1=max(0,int((cx-w/2)*360))
    y1=max(0,int((cy-h/2)*363))
    x2=min(360,int((cx+w/2)*360))
    y2=min(363,int((cy+h/2)*363))
    boxes.append((x1,y1,x2,y2))

r=[np.count_nonzero(f[y1:y2,x1:x2]==255)/((x2-x1)*(y2-y1)) for x1,y1,x2,y2 in boxes if x2>x1 and y2>y1]
print('Cov Mean:', np.mean(r))
