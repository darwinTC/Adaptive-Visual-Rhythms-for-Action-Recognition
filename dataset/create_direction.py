import os
import sys
import glob
import argparse
import numpy as np
import cv2
from multiprocessing import Pool, current_process
from skimage.feature import hog
from skimage import data, color, exposure
from itertools import product
list_class = []
first_data ={}
second_data = {}

def run(enum, video):
    '''
    Determines the direction of movement of a given video.
    '''

    vid_name = video.split('/')[-1].split('.')[0]
    vid_class = video.split('/')[3]
    # map each video with its respective video class
    first_data[vid_name]=vid_class

    cap = cv2.VideoCapture(video)
    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.3,
                           minDistance = 7,
                           blockSize = 7 )
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # Create some random colors
    color = np.random.randint(0,255,(100,3))
    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    kernel = np.ones((5,5),np.float32)/25
    old_gray = cv2.medianBlur(cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY),5)
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    ejex = 1
    ejey = 0
    while 1:
        ret,frame = cap.read()
        if not ret:
            break
        frame_gray = cv2.medianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),5)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        try:
            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        except cv2.error:
            break

        if (p1 is None) or (p0 is None):
            break
        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]
        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            ejex += abs(a-c)
            ejey += abs(b-d)
            mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)   
        img = cv2.add(frame,mask)

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)

    # save b each video its respective direction (1=ejey, 2=ejex)
    if ejex>ejey:
        second_data[vid_name] = 2
    else:
        second_data[vid_name] = 1
    
    dir_ = 'x' if ejex > ejey else 'y'
    print(str(enum)+' : name vide: '+vid_name+' => direction '+dir_)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Determine the direction of movement by class of a given dataset')
    parser.add_argument('--src_dir', type=str, default='./UCF-101',
                        help='path to the video data')

    parser.add_argument('--num_worker', type=int, default=8)
    parser.add_argument('--num_gpu', type=int, default=2, help='number of GPU')
    parser.add_argument('--ext', type=str, default='avi', choices=['avi','mp4'],
                        help='video file extensions')

    args = parser.parse_args()
    src_path = args.src_dir
    num_worker = args.num_w orker
    ext = args.ext
    NUM_GPU = args.num_gpu

    vid_list = sorted(glob.glob(src_path+'/*/*.'+ext))

    for i, vid in enumerate(vid_list):
        run(i, vid)
    
    #np.save("direction_video_class.npy", first_data)
    #np.save("direction_video_rhythm.npy", second_data)

    #first_data = np.load('direction_video_class.npy').item()
    #second_data = np.load('direction_video_rhythm.npy').item()
    dic_final = {}
    list_name_vid = sorted([video.split('/')[-1].split('.')[0] for video in vid_list])
    list_class =sorted(set([video.split('/')[3] for video in vid_list]))

    for vid in list_name_vid:
        if first_data[vid] in dic_final:
            if second_data[vid]==2:
                dic_final[first_data[vid]][0]+=1
            else:
                dic_final[first_data[vid]][1]+=1        
        else:
            if second_data[vid]==2:
                dic_final[first_data[vid]]=[1,0]
            else:
                dic_final[first_data[vid]]=[0,1]
    direction = []
    for i,vid in enumerate(list_class):
        idx = '2' if dic_final[vid][0]>dic_final[vid][1] else '1'
        direction.append(idx)

    open('directionHMDB.txt', 'w').writelines(direction)

