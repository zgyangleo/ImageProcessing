import numpy as np
import cv2

def diffimage(src_1,src_2):
    src_1 = src_1.astype(np.int)
    src_2 = src_2.astype(np.int)
    diff = abs(src_1 - src_2)
    return diff.astype(np.uint8)

def convertImage(src):
    return cv2.cvtColor(src,cv2.COLOR_RGB2GRAY)

def diffimage(src_1,src_2):
    src_1 = src_1.astype(np.int)
    src_2 = src_2.astype(np.int)
    diff = abs(src_1 - src_2)
    return diff.astype(np.uint8)

def barMinus(src_1,src_2):
    src_1 = convertImage(src_1)
    src_2 = convertImage(src_2)
    return np.sum(diffimage(src_1,src_2))
    
keyfrmae = 1
cap = cv2.VideoCapture('monitor.avi')
ret, lastframe = cap.read()
l  = []
while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break
    if barMinus(lastframe,frame)/frame.size>2: 
        cv2.imwrite("barminus"+str(keyfrmae)+".jpg",frame)
        keyfrmae+=1
    lastframe = frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()