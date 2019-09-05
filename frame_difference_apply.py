import numpy as np
import cv2
import uuid
import os

VIDEO_ROOT = '/.../video/'
FRAME_ROOT = '/../frame/'

def diffimage(src_1,src_2):
    src_1 = src_1.astype(np.int)
    src_2 = src_2.astype(np.int)
    diff = abs(src_1 - src_2)
    return diff.astype(np.uint8)

video_list = os.listdir(VIDEO_ROOT)
for video_name in video_list:
    keyframe = 1
    file_name = video_name[:-5] + '/'
    os.makedirs(FRAME_ROOT+file_name)
#    print(type(file_name))
#    print(file_name)
#    print(FRAME_ROOT)
    cap = cv2.VideoCapture(VIDEO_ROOT+video_name)
    ret, lastframe = cap.read()
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        if np.sum(diffimage(lastframe,frame))/frame.size > 12:
            
            cv2.imwrite(FRAME_ROOT+file_name+str(keyframe)+'.jpg',frame)
#            print(FRAME_ROOT+file_name+str(keyframe)+'.jpg')
            keyframe+=1
        lastframe = frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break            

    cap.release()
    cv2.destroyAllWindows()

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
            if np.sum(diffimage(lastframe,frame))/frame.size > 1:
            
                cv2.imwrite(video_path+str(keyframe)+'.jpg',frame)
#                print(FRAME_ROOT+file_name+str(keyframe)+'.jpg')
                keyframe+=1
            lastframe = frame
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()