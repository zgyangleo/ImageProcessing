import numpy as np
import cv2
import uuid

def diffimage(src_1,src_2):
    src_1 = src_1.astype(np.int)
    src_2 = src_2.astype(np.int)
    diff = abs(src_1 - src_2)
    return diff.astype(np.uint8)

keyfrmae = 1
cap = cv2.VideoCapture('monitor.avi')
ret, lastframe = cap.read()
while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break
    if np.sum(diffimage(lastframe,frame))/frame.size > 10:
        cv2.imwrite("frameminus"+str(keyfrmae)+".jpg",frame)
        keyfrmae+=1
    lastframe = frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()