import numpy as np
import cv2
import uuid
import os
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

VIDEO_ROOT = '/.../video/'
FRAME_ROOT = '/.../frame/'
video_list = os.listdir(VIDEO_ROOT)
for video_name in video_list:
    keyframe = 1
    file_name = video_name[:-5] + '/'
    os.makedirs(FRAME_ROOT+file_name)
#    print(type(file_name))
#    print(file_name)
#    print(FRAME_ROOT)
    keyfrmae = 1
    cap = cv2.VideoCapture(VIDEO_ROOT+video_name)
    ret, lastframe = cap.read()
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        if phash(lastframe,frame)>7:
            cv2.imwrite(FRAME_ROOT+file_name+str(keyfrmae)+".jpg",frame)
            keyfrmae+=1
        lastframe = frame
#        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
'''
    video_path = FRAME_ROOT + file_name
    videoremove_list = os.listdir(video_path)
    print(len(videoremove_list))
    if len(videoremove_list) < 8:
        for name in videoremove_list:
            os.remove(os.path.join(video_path, name))
        cap = cv2.VideoCapture(VIDEO_ROOT+video_name)
        ret, lastframe = cap.read()
        while(cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break
            if phash(lastframe,frame)>1:
                cv2.imwrite(FRAME_ROOT+file_name+str(keyfrmae)+".jpg",frame)
                keyfrmae+=1
            lastframe = frame
#        cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
'''        
