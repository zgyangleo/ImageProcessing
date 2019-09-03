import numpy as np
import cv2
import uuid
def convertImage(image):
    image = cv2.resize(image,(32,32),interpolation=cv2.INTER_CUBIC)
    image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    image = image.astype(np.float32)
    return image

def cal_dct(image):
    image = convertImage(image)
    from copy import deepcopy
    dct_matrix = deepcopy(image)
    dct_1 = cv2.dct(cv2.dct(dct_matrix))[:8,:8]
    img_list = dct_1.flatten()
    avg = np.mean(img_list)
    img_list = ['0' if i<avg else '1' for i in img_list]
    #将二进制转化为16进制
    fig = ""
    for i in range(0,64,4):
        num = hex(int("".join(img_list[i:i+4]),2))
        fig += num[2:]
    return fig

def phash(image1,image2):
    def hammingDist(s1, s2):
        assert len(s1) == len(s2)
        return sum([ch1 != ch2 for ch1, ch2 in zip(s1, s2)])
    s1 = cal_dct(image1)
    s2 = cal_dct(image2)
    return hammingDist(s1,s2)

keyfrmae = 1
cap = cv2.VideoCapture('monitor.avi')
ret, lastframe = cap.read()
while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break
    if phash(lastframe,frame)>8:
        cv2.imwrite("phash"+str(keyfrmae)+".jpg",frame)
        keyfrmae+=1
    lastframe = frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()