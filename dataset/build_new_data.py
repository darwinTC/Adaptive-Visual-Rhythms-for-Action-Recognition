"""
    author : darwinTC (darwin.ttito.c@gmail.com)
    data : 20/04/2018
    description : methods to create the visual rhythm for each video of UCF101 and
                  HMDB51 dataset
"""
import os
import sys
import glob
import argparse
import numpy as np
import cv2
from multiprocessing import Pool, current_process


def load_video(file_name, height = 240, width = 320, flag = False):
    '''
        Load Video
        width  : 320
        height : 240
    '''
    vidcap = cv2.VideoCapture(file_name)
    cnt = 0
    frames = []
    success, img = vidcap.read()
    while success:
        if not flag:
            img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        img = np.float32(img)
        frames.append(img)
        success, img = vidcap.read()

    missing_frames = 0
    number_frames = len(frames)
    # complete the number of frame missing   
    if width > len(frames):
        missing_frames = width - number_frames;
    iterator = missing_frames // number_frames
    aditional = missing_frames % number_frames if missing_frames % number_frames != 0 else 0
        
    new_frames = []
    for frame in frames:
        new_frames.append(frame)
        for i in range(0,iterator):
            new_frames.append(frame)
        if aditional !=0:
            new_frames.append(frame)
            aditional -= 1
    return new_frames

def visual_rhythm_diagonal_down_top(frames, path, vid_name, height = 240, width = 320):
    '''
        create visual rhythm images from the diagonal of each frame
        0 0 0 4       1
        0 0 3 0       2
        0 2 0 0 ==>   3
        1 0 0 0       4
    '''
    rhythm_red = np.array([]).reshape(0,1)  
    rhythm_green = np.array([]).reshape(0,1)
    rhythm_blue = np.array([]).reshape(0,1)
    is_begin = True
    for frame in frames:
        r_mean = np.vstack(np.diagonal(np.flip(frame[:,:,0],0)))
        g_mean = np.vstack(np.diagonal(np.flip(frame[:,:,1],0)))
        b_mean = np.vstack(np.diagonal(np.flip(frame[:,:,2],0)))
        if is_begin :
            rhythm_red = np.vstack((rhythm_red,r_mean)) 
            rhythm_green = np.vstack((rhythm_green,g_mean)) 
            rhythm_blue = np.vstack((rhythm_blue,b_mean))
            is_begin = False
        else:
            rhythm_red = np.concatenate((rhythm_red,r_mean), axis=1) 
            rhythm_green = np.concatenate((rhythm_green,g_mean), axis=1) 
            rhythm_blue = np.concatenate((rhythm_blue,b_mean), axis=1)
    
    if rhythm_red.shape[0] > height :
        rhythm_red = cv2.resize(rhythm_red,(width,height), cv2.INTER_LINEAR) 
        rhythm_green = cv2.resize(rhythm_green,(width,height), cv2.INTER_LINEAR) 
        rhythm_blue = cv2.resize(rhythm_blue,(width,height), cv2.INTER_LINEAR) 

    if os.path.isfile(path%(4)):
        os.remove(path%(4))
    print("Creating vertical visual rhythm to video : {0}.jpg".format(vid_name))
    image_rhythm = np.dstack((rhythm_red,rhythm_green ,rhythm_blue)) 
    cv2.imwrite(path % (4), image_rhythm)
    #path2 = "../visual_rhythm2/"+vid_name+".jpg"
    #cv2.imwrite(path2, image_rhythm)     

def visual_rhythm_diagonal_top_down(frames, path, vid_name, height = 240, width = 320):
    '''
        create visual rhythm images from the diagonal of each frame
        1 0 0 0        1
        0 2 0 0        2
        0 0 3 0  ==>   3
        0 0 0 4        4
    '''
    rhythm_red = np.array([]).reshape(0,1)  
    rhythm_green = np.array([]).reshape(0,1)
    rhythm_blue = np.array([]).reshape(0,1)
    is_begin = True
    for frame in frames:
        r_mean = np.vstack(np.diagonal(frame[:,:,0]))
        g_mean = np.vstack(np.diagonal(frame[:,:,1]))
        b_mean = np.vstack(np.diagonal(frame[:,:,2]))
        if is_begin :
            rhythm_red = np.vstack((rhythm_red,r_mean)) 
            rhythm_green = np.vstack((rhythm_green,g_mean)) 
            rhythm_blue = np.vstack((rhythm_blue,b_mean))
            is_begin = False
        else:
            rhythm_red = np.concatenate((rhythm_red,r_mean), axis=1) 
            rhythm_green = np.concatenate((rhythm_green,g_mean), axis=1) 
            rhythm_blue = np.concatenate((rhythm_blue,b_mean), axis=1)
    
    if rhythm_red.shape[0] > height :
        rhythm_red = cv2.resize(rhythm_red,(width,height), cv2.INTER_LINEAR) 
        rhythm_green = cv2.resize(rhythm_green,(width,height), cv2.INTER_LINEAR) 
        rhythm_blue = cv2.resize(rhythm_blue,(width,height), cv2.INTER_LINEAR) 

    if os.path.isfile(path%(3)):
        os.remove(path%(3))
    print("Creating vertical visual rhythm to video : {0}.jpg".format(vid_name))
    image_rhythm = np.dstack((rhythm_red,rhythm_green ,rhythm_blue)) 
    cv2.imwrite(path % (3), image_rhythm)
    #path2 = "../visual_rhythm/"+vid_name+".jpg"
    #cv2.imwrite(path2, image_rhythm)   
   
def create_visual_rhythm_mean_horizontal_vertical(frames, path, vid_name, height = 240, width = 240):   
    '''
        Create horizontal and vertical visual rhythm image from a video
    '''        
    rhythm_red = np.array([]).reshape(0,1)  
    rhythm_green = np.array([]).reshape(0,1)
    rhythm_blue = np.array([]).reshape(0,1)
    is_begin = True
    for frame in frames:
        a= cv2.resize(frame[:,:,0],(width,height), cv2.INTER_LINEAR) 
        b = cv2.resize(frame[:,:,1],(width,height), cv2.INTER_LINEAR) 
        c = cv2.resize(frame[:,:,2],(width,height), cv2.INTER_LINEAR)
        r_mean = np.vstack((np.mean(a,axis=1)+np.mean(a,axis=0))/2)
        g_mean = np.vstack((np.mean(b,axis=1)+np.mean(b,axis=0))/2)
        b_mean = np.vstack((np.mean(c,axis=1)+np.mean(c,axis=0))/2)
        if is_begin :
            rhythm_red = np.vstack((rhythm_red,r_mean)) 
            rhythm_green = np.vstack((rhythm_green,g_mean)) 
            rhythm_blue = np.vstack((rhythm_blue,b_mean))
            is_begin = False
        else:
            rhythm_red = np.concatenate((rhythm_red,r_mean), axis=1) 
            rhythm_green = np.concatenate((rhythm_green,g_mean), axis=1) 
            rhythm_blue = np.concatenate((rhythm_blue,b_mean), axis=1)
    #print(rhythm_red.shape)
    
    if rhythm_red.shape[1] > width :
        rhythm_red = cv2.resize(rhythm_red,(width,height), cv2.INTER_LINEAR) 
        rhythm_green = cv2.resize(rhythm_green,(width,height), cv2.INTER_LINEAR) 
        rhythm_blue = cv2.resize(rhythm_blue,(width,height), cv2.INTER_LINEAR)
        
    if os.path.isfile(path%(5)):
        os.remove(path%(5))
    print("Creating horizontal visual rhythm to video : {0}.jpg".format(vid_name))
    image_rhythm = np.dstack((rhythm_red,rhythm_green ,rhythm_blue)) 
    cv2.imwrite(path % (5), image_rhythm)
    #path2 = "../visual-rhythm/"+vid_name+".jpg"
    #cv2.imwrite(path2, image_rhythm)
 
def create_visual_rhythm_mean_horizontal(frames, path, vid_name, height = 240, width = 320):   
    '''
        Create horizontal visual rhythm image from a video
    '''
    rhythm_red = np.array([]).reshape(0,1)  
    rhythm_green = np.array([]).reshape(0,1)
    rhythm_blue = np.array([]).reshape(0,1)
    is_begin = True
    for frame in frames:
        r_mean = np.vstack(np.mean(frame[:,:,0],axis=1))
        g_mean = np.vstack(np.mean(frame[:,:,1],axis=1))
        b_mean = np.vstack(np.mean(frame[:,:,2],axis=1))
        if is_begin :
            rhythm_red = np.vstack((rhythm_red,r_mean)) 
            rhythm_green = np.vstack((rhythm_green,g_mean)) 
            rhythm_blue = np.vstack((rhythm_blue,b_mean))
            is_begin = False
        else:
            rhythm_red = np.concatenate((rhythm_red,r_mean), axis=1) 
            rhythm_green = np.concatenate((rhythm_green,g_mean), axis=1) 
            rhythm_blue = np.concatenate((rhythm_blue,b_mean), axis=1)
    #print(rhythm_red.shape)
    
    if rhythm_red.shape[1] > width :
        rhythm_red = cv2.resize(rhythm_red,(width,height), cv2.INTER_LINEAR) 
        rhythm_green = cv2.resize(rhythm_green,(width,height), cv2.INTER_LINEAR) 
        rhythm_blue = cv2.resize(rhythm_blue,(width,height), cv2.INTER_LINEAR)

    if os.path.isfile(path%(1)):
        os.remove(path%(1))
    print("Creating horizontal visual rhythm to video : {0}.jpg".format(vid_name))
    image_rhythm = np.dstack((rhythm_red,rhythm_green ,rhythm_blue)) 
    cv2.imwrite(path % (1), image_rhythm)
    #path2 = "../visual-rhythm/"+vid_name+".jpg"
    #cv2.imwrite(path2, image_rhythm)

def create_visual_rhythm_mean_vertical(frames, path, vid_name, height = 240, width = 320):   
    '''
        Create vertical visual rhythm image from a video
    '''
    rhythm_red = np.array([])
    rhythm_green = np.array([])
    rhythm_blue = np.array([])
    is_begin = True
    for frame in frames:
        r_mean = np.mean(frame[:,:,0],axis=0)
        g_mean = np.mean(frame[:,:,1],axis=0)
        b_mean = np.mean(frame[:,:,2],axis=0)

        rhythm_red = np.vstack((rhythm_red,r_mean)) if rhythm_red.size else r_mean 
        rhythm_green = np.vstack((rhythm_green,g_mean)) if rhythm_green.size else g_mean 
        rhythm_blue = np.vstack((rhythm_blue,b_mean)) if rhythm_blue.size else b_mean 
    
    if rhythm_red.shape[0] > height :
        rhythm_red = cv2.resize(rhythm_red,(width,height), cv2.INTER_LINEAR) 
        rhythm_green = cv2.resize(rhythm_green,(width,height), cv2.INTER_LINEAR) 
        rhythm_blue = cv2.resize(rhythm_blue,(width,height), cv2.INTER_LINEAR) 

    if os.path.isfile(path%(2)):
        os.remove(path%(2))
    print("Creating vertical visual rhythm to video : {0}.jpg".format(vid_name))
    image_rhythm = np.dstack((rhythm_red,rhythm_green ,rhythm_blue)) 
    cv2.imwrite(path % (2), image_rhythm)
    #path2 = "../visual_rhythm2/"+vid_name+".jpg"
    #cv2.imwrite(path2, image_rhythm)

def create_visual_rhythm_gray_scale(frames, height = 240, width = 320):
    rhythm = np.array([])
    for frame in frames:
        row_mean = np.mean(frame,axis=0)
        rhythm = np.vstack((rhythm,row_mean)) if rhythm.size else row_mean
    
    if rhythm.shape[0] > height :
        rhythm = cv2.resize(rhythm,(width,height), cv2.INTER_LINEAR)
    return rhythm

def create_image_gradients(frames, out_full_path, path, vid_name, height = 240, width = 320):
    labels = ['x','y']    
    gradientX = []
    gradientY = []
    gradientXX = []
    gradientYY = []
    for index, img in enumerate(frames):
        sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
        gradientX.append(sobelx)
        sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
        gradientY.append(sobely)
        sobelxx = cv2.Sobel(sobelx,cv2.CV_64F,1,0,ksize=3)
        gradientXX.append(sobelxx)
        sobelyy = cv2.Sobel(sobely,cv2.CV_64F,0,1,ksize=3)
        gradientYY.append(sobelyy)
    
    rhythm_x = create_visual_rhythm_gray_scale(gradientX)
    rhythm_y = create_visual_rhythm_gray_scale(gradientY)
    print("Creating : {}".format(vid_name))
    #cv2.imwrite(path % (7), rhythm_x)
    #cv2.imwrite(path % (8), rhythm_y)
    path2 = "../gradients/"+vid_name+".jpg"
    cv2.imwrite(path2, rhythm_y)
    
    '''
    for index, img in enumerate(frames):
        for index_, label in enumerate(labels):
            path_image = path.format(out_full_path, label) % (index)
            if os.path.isfile(path_image):
                print("remove : "+path_image)
                os.remove(path_image)         
    '''

def run_create_images(vid_item):
    '''
        Create visual rhythm and gradients images
    '''
    vid_path = vid_item[0]
    vid_id = vid_item[1]
    vid_name = vid_path.split('/')[-1].split('.')[0]
    out_full_path = os.path.join(out_path, vid_name)
    try:
        os.mkdir(out_full_path)
    except OSError:
        pass
    current = current_process()
    dev_id = (int(current._identity[0]) - 1) % NUM_GPU
    visual_rhythm_path = '{}/visual_rhythm_%05d.jpg'.format(out_full_path)
    frames = load_video(vid_path, width = 320, flag = (modality != 'gradients'))
    if modality == 'rhythm-H':
        create_visual_rhythm_mean_horizontal(frames, visual_rhythm_path,vid_name)
    elif modality == 'rhythm-V':    
        create_visual_rhythm_mean_vertical(frames, visual_rhythm_path,vid_name)
    elif modality == 'rhythm-HV':
        create_visual_rhythm_mean_horizontal_vertical(frames, visual_rhythm_path,vid_name)    
    elif modality == 'gradients':
        gradients_path = '{}/gradient_{}_%05d.jpg'
        create_image_gradients(frames, out_full_path, visual_rhythm_path,vid_name)
    elif modality == 'rhythm-DTD':
        visual_rhythm_diagonal_top_down(frames, visual_rhythm_path,vid_name)
    elif modality == 'rhythm-DDT':
        visual_rhythm_diagonal_down_top(frames, visual_rhythm_path,vid_name)
    return True;

def create_train_test_files(path):
    '''
        Create files to train and test
    '''
    path_train = path + '/ucf101/train_flow_split{:d}.txt'
    path_test = path + '/ucf101/val_flow_split{:d}.txt'

    path_train_rhythm = path + '/ucf101/train_rhythm_split{:d}.txt'
    path_test_rhythm = path + '/ucf101/val_rhythm_split{:d}.txt'

    def line2rec(line):
        items = line.strip().split(' ')
        return "{0} {1} {2}".format(items[0],1,items[2])
    splits = []
    for i in range(1, 4):
        train_list = [line2rec(x) for x in open(path_train.format(i))]
        test_list = [line2rec(x) for x in open(path_test.format(i))]
        print(path_train.format(i))
        new_train = open(path_train_rhythm.format(i),'w')
        new_test = open(path_test_rhythm.format(i),'w')
        for line in train_list:
            new_train.write("{}\n".format(line))
        for line in test_list:
            new_test.write("{}\n".format(line))
        new_train.close()
        new_test.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="extract visual rhythm ")
    parser.add_argument("--src_dir", type=str, default='./UCF-101',
                        help='path to the video data')
    parser.add_argument("--out_dir", type=str, default='./ucf101_frames',
                        help='path to store visual thythm')
    parser.add_argument("--new_width", type=int, default=0, help='resize image width')
    parser.add_argument("--new_height", type=int, default=0, help='resize image height')

    parser.add_argument("--num_worker", type=int, default=8)
    parser.add_argument("--num_gpu", type=int, default=2, help='number of GPU')
    parser.add_argument("--out_format", type=str, default='dir', choices=['dir','zip'],
                        help='path to the visual rhythm')
    parser.add_argument("--ext", type=str, default='avi', choices=['avi','mp4'],
                        help='video file extensions')
    parser.add_argument('--modality', '-m', metavar='MODALITY', default='rhythm-H',
                        choices=["rhythm-H", "rhythm-V", "rhythm-HV", "gradients", "rhythm-DTD", "rhythm-DDT"],
                        help='modality: rhythm-H | rhythm-V | rhythm-HV | gradients | rhythm-DTD | rhythm-DDT ')
    parser.add_argument('--type_gradient', '-tg', metavar='GRADIENT', default='gradient_x',
                        choices=["gradient_x", "gradient_y"],
                        help='modality: gradient_x | gradient_y')

    args = parser.parse_args()
    out_path = args.out_dir
    src_path = args.src_dir
    num_worker = args.num_worker
    out_format = args.out_format
    ext = args.ext
    modality = args.modality
    new_size = (args.new_width, args.new_height)
    NUM_GPU = args.num_gpu

    if not os.path.isdir(out_path):
        print("creating folder: "+out_path)
        os.makedirs(out_path)

    vid_list = glob.glob(src_path+'/*/*.'+ext)
    print(vid_list)
    pool = Pool(num_worker)
    pool.map(run_create_images, zip(vid_list, range(len(vid_list))))
