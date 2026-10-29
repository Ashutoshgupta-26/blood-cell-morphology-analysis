import cv2, numpy as np
from src.morphology import fill_holes
img = cv2.imread('C:\\Blood Project\\TXL-PBC_Dataset-master\\TXL-PBC\\images\\test\\7f96e833a77559fb8fe76152b2b76660.png')
g = img[:,:,1]
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

boxes=[]
lines=open('C:\\Blood Project\\TXL-PBC_Dataset-master\\TXL-PBC\\labels\\test\\7f96e833a77559fb8fe76152b2b76660.txt').readlines()
for l in lines:
    p = l.strip().split()
    if not p: continue
    cx,cy,w,h=map(float,p[1:5])
    x1=max(0,int((cx-w/2)*img.shape[1]))
    y1=max(0,int((cy-h/2)*img.shape[0]))
    x2=min(img.shape[1],int((cx+w/2)*img.shape[1]))
    y2=min(img.shape[0],int((cy+h/2)*img.shape[0]))
    boxes.append((x1,y1,x2,y2))

for t in [165, 170, 175, 180, 185, 190, 195]:
    _, m = cv2.threshold(g, t, 255, cv2.THRESH_BINARY_INV)
    c = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    f = fill_holes(c)
    
    # Also evaluate if cells merged (components)
    num, _ = cv2.connectedComponents(f)
    
    r=[np.count_nonzero(f[y1:y2,x1:x2]==255)/((x2-x1)*(y2-y1)) for x1,y1,x2,y2 in boxes if x2>x1 and y2>y1]
    print(f"Thresh {t}: Cov = {np.mean(r):.3f}, Comp = {num-1}")
