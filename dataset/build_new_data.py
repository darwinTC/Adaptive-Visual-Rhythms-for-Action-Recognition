'''
    author : darwinTC (darwin.ttito.c@gmail.com)
    data : 20/04/2018
    description : methods to create several type of visual rhythm to videos of UCF101 and
                  HMDB51 dataset
'''
import os
import sys
import glob
import argparse
import numpy as np
import cv2
from multiprocessing import Pool, current_process
from skimage.feature import hog
from skimage import data, color, exposure

def complete_frames(frames, height = 240, width = 320):
    '''
        Increase the number of frames to be able to obtain the desired
        dimensions of an image.
    '''
    missing_frames = 0
    number_frames = len(frames)
    # complete the number of frame missing   
    if width > number_frames:
        missing_frames = width - number_frames
    iterator = missing_frames // number_frames
    aditional = missing_frames % number_frames
        
    new_frames = []
    for frame in frames:
        for i in range(0,iterator+1):
            new_frames.append(frame)
        if aditional !=0:
            new_frames.append(frame)
            aditional -= 1
    return new_frames  

def load_video(file_name, height = 240, width = 320, flag = False):
    '''
        Load Video
        width  : 320
        height : 240
    '''
    vidcap = cv2.VideoCapture(file_name)
    frames = []
    success, img = vidcap.read()
    while success:
        if not flag:
            img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        img = np.float32(img)
        frames.append(img)
        success, img = vidcap.read()

    # return the current frames if we need to calculate the gradient or HOG images
    if not flag:
        return frames
    else:
        return complete_frames(frames, height, width)

def visual_rhythm_diagonal_down_top(frames, path, vid_name, height = 240, width = 320):
    '''
        Create a column for the visual rhythm image from each frame of any video, where each value of
        this column is a value of the frame' diagonal (from bottom to top and left to right).

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
    print('Creating vertical visual rhythm to video : {0}.avi'.format(vid_name))
    image_rhythm = np.dstack((rhythm_red,rhythm_green ,rhythm_blue)) 
    cv2.imwrite(path % (4), image_rhythm)
    #path2 = '../visual_rhythm2/'+vid_name+'.jpg'
    #cv2.imwrite(path2, image_rhythm)     

def visual_rhythm_diagonal_top_down(frames, path, vid_name, height = 240, width = 320):
    '''
        Create a column for the visual rhythm image from each frame of any video, where each value of
        this column is a value of the frame' diagonal (from top to bottom and left to right).

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
    print('Creating vertical visual rhythm to video : {0}.avi'.format(vid_name))
    image_rhythm = np.dstack((rhythm_red,rhythm_green ,rhythm_blue)) 
    cv2.imwrite(path % (3), image_rhythm)
    #path2 = '../visual_rhythm/'+vid_name+'.jpg'
    #cv2.imwrite(path2, image_rhythm)   
   
def create_visual_rhythm_mean_horizontal_vertical(frames, path, vid_name, height = 240, width = 240):   
    '''
        Create a column for the visual rhythm image from each frame of any video, where each value of
        this column is the mean of values of its respective row and column.

        1 2 3    mean(mean(1,2,3), mean(1,4,7)) ..
        4 5 6 => mean(mean(4,5,6), mean(2,5,8)) ..
        7 8 9    mean(mean(7,8,9), mean(3,6,9)) .. 
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
    
    if rhythm_red.shape[1] > width :
        rhythm_red = cv2.resize(rhythm_red,(width,height), cv2.INTER_LINEAR) 
        rhythm_green = cv2.resize(rhythm_green,(width,height), cv2.INTER_LINEAR) 
        rhythm_blue = cv2.resize(rhythm_blue,(width,height), cv2.INTER_LINEAR)
        
    if os.path.isfile(path%(5)):
        os.remove(path%(5))
    print('Creating horizontal visual rhythm to video : {0}.avi'.format(vid_name))
    image_rhythm = np.dstack((rhythm_red,rhythm_green ,rhythm_blue)) 
    cv2.imwrite(path % (5), image_rhythm)
    #path2 = '../visual-rhythm/'+vid_name+'.jpg'
    #cv2.imwrite(path2, image_rhythm)
 
def create_visual_rhythm_mean_horizontal(frames, path, vid_name, height = 240, width = 320):   
    '''
        Create a column for the visual rhythm image from each frame of any video, where each value of
        this column is the mean of values of its respective row. 

        1 2 3    mean(1,2,3) ...
        4 5 6 => mean(4,5,6) ...
        7 8 9    mean(7,8,9) ...
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
    
    if rhythm_red.shape[1] > width :
        rhythm_red = cv2.resize(rhythm_red,(width,height), cv2.INTER_LINEAR) 
        rhythm_green = cv2.resize(rhythm_green,(width,height), cv2.INTER_LINEAR) 
        rhythm_blue = cv2.resize(rhythm_blue,(width,height), cv2.INTER_LINEAR)

    if os.path.isfile(path%(1)):
        os.remove(path%(1))
    print('Creating horizontal visual rhythm to video : {0}.avi'.format(vid_name))
    image_rhythm = np.dstack((rhythm_red,rhythm_green ,rhythm_blue)) 
    cv2.imwrite(path % (1), image_rhythm)
    #path2 = '../visual-rhythm/'+vid_name+'.jpg'
    #cv2.imwrite(path2, image_rhythm)

def create_visual_rhythm_mean_vertical(frames, path, vid_name, height = 240, width = 320):   
    '''
        Create a row for the visual rhythm image from each frame of any video,  where each value of
        this row is the mean of values of its respective column. 

        1 2 3        .           .           .
        4 5 6 => mean(1,4,7) mean(2,5,8) mean(3,6,9)
        7 8 9        .           .           .
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
    print('Creating vertical visual rhythm to video : {0}.avi'.format(vid_name))
    image_rhythm = np.dstack((rhythm_red,rhythm_green ,rhythm_blue)) 
    cv2.imwrite(path % (2), image_rhythm)
    #path2 = '../visual_rhythm2/'+vid_name+'.jpg'
    #cv2.imwrite(path2, image_rhythm)

def create_visual_rhythm_gray_scale(frames, height = 240, width = 320):
    rhythm = np.array([])
    for frame in frames:
        row_mean = np.mean(frame,axis=0)
        rhythm = np.vstack((rhythm,row_mean)) if rhythm.size else row_mean
    
    if rhythm.shape[0] > height :
        rhythm = cv2.resize(rhythm,(width,height), cv2.INTER_LINEAR)
    return rhythm

def create_visual_rhythm_with_gradients(frames, path, vid_name, height = 240, width = 320):
    '''
        First, is created the gradients(X,Y,XX,YY) of each video' frame and this new frames are store
        in an array, next is created visual rhythm images for each new amount of frame.
    '''
    labels = ['x','y']    
    gradientX = []
    gradientY = []
    gradientXX = []
    gradientYY = []
    for index, img in enumerate(frames):
        gradientX.append(cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3))
        gradientY.append(cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3))
        gradientXX.append(cv2.Sobel(sobelx,cv2.CV_64F,1,0,ksize=3))
        gradientYY.append(cv2.Sobel(sobely,cv2.CV_64F,0,1,ksize=3))
    
    rhythm_x = create_visual_rhythm_gray_scale(gradientX)
    rhythm_y = create_visual_rhythm_gray_scale(gradientY)
    print('Creating : {}'.format(vid_name))
    #cv2.imwrite(path % (7), rhythm_x)
    #cv2.imwrite(path % (8), rhythm_y)
    path2 = '../gradients/'+vid_name+'.jpg'
    cv2.imwrite(path2, rhythm_y)

def create_images_hog(frames, path, vid_name, height = 240, width = 320):
    '''
        This method create HOG images from each frame of any video.
    '''
    new_frames=[]
    for i, image in enumerate(frames):          
        __, hog_image = hog(image, orientations=8, pixels_per_cell=(8, 8),
                            cells_per_block=(1, 1), visualise=True)
        print('Creating HOG to video : {}.avi'.format(vid_name))
        hog_image = hog_image*50
        new_frames.append(hog_image)
        cv2.imwrite(path % (i+1), hog_image)
    # create visual rhythm from the hog images    
    #print('Creating visual rhythm images to video '+vid_name)
    #VR = create_visual_rhythm_gray_scale(new_frames)
    #cv2.imwrite('../hog/'+vid_name+'.jpg',VR)    

def obtain_previous_optical_flow_images(out_full_path):
    '''
        Method to obtain the previous optical flow images generated
        and stored in the folde 'nameDS'_frame, where nameDS is the
        name of the dataset.
    '''
    out_full_path_ = glob.escape(out_full_path)
    path_images = [elem.split('/')[-1] for elem in glob.glob(os.path.join(out_full_path_, '*'))]

    flow_x = sorted([os.path.join(out_full_path,elem) for elem in path_images if elem[:6]=='flow_x'])
    flow_y = sorted([os.path.join(out_full_path,elem) for elem in path_images if elem[:6]=='flow_y'])

    # read images from flow_x and flow_y
    flow_x = [cv2.cvtColor(cv2.imread(dir_img),cv2.COLOR_RGB2GRAY) for dir_img in flow_x]
    flow_y = [cv2.cvtColor(cv2.imread(dir_img),cv2.COLOR_RGB2GRAY) for dir_img in flow_y]    
    
    return flow_x, flow_y

def create_visual_rhythm_from_optical_flow(visual_rhythm_path, out_full_path, vid_name, height = 240, width = 320):
    '''
        Create visual rhythm with previous optical flow images
    '''
    flow_x, flow_y = obtain_previous_optical_flow_images(out_full_path)
    
    new_flow_x = complete_frames(flow_x)
    new_flow_y = complete_frames(flow_y)
    
    RV_flow_x = create_visual_rhythm_gray_scale(new_flow_x)
    RV_flow_y = create_visual_rhythm_gray_scale(new_flow_y)
    
    print('creating visual rhythm images to video : ' + vid_name)
    cv2.imwrite('../RV_flow/'+vid_name+'_flow_x.jpg',RV_flow_x)
    cv2.imwrite('../RV_flow/'+vid_name+'_flow_y.jpg',RV_flow_x)

def create_HOG_from_optical_flow(out_full_path, vid_name):
    '''
        This method create HOG images from previous optical flow images, this is
        because the optical images contain very relevant information of the actor
        excluing the background, so taked this images is better than the RGB one.
    '''
    flow_x, flow_y = obtain_previous_optical_flow_images(out_full_path)  

    print('Creating images to video: '+vid_name)
    for i in range(len(flow_x)):
        __, img_x = hog(flow_x[i], orientations=8, pixels_per_cell=(8, 8),
                                cells_per_block=(1, 1), visualise=True)
        __, img_y = hog(flow_y[i], orientations=8, pixels_per_cell=(8, 8),
                                cells_per_block=(1, 1), visualise=True)
        img_x = img_x*200
        img_y = img_y*200
        cv2.imwrite(os.path.join(out_full_path,'hog_x_%05d.jpg')%(i+1),img_x)
        cv2.imwrite(os.path.join(out_full_path,'hog_y_%05d.jpg')%(i+1),img_y)

def run_create_images(vid_item):
    '''
        Create several type of visual rhythm, gradients and hog images
        rhythm-H : horizontal visual rhythm
        rhythm-V : vertical visual rhythm
        rhythm-HV : mean(horizontal ,vertical) visual rhythm
        rhythm-DTD : diagonal visual rhythm (from top to down)
        rhythm-DDT : diagonal visual rhythm (from down to top)
        rhythm-OF : visual rhythm - optical flow
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
    frames = load_video(vid_path, width = 320, flag = (modality[0:6] == 'rhythm'))
    if modality == 'rhythm-H':
        create_visual_rhythm_mean_horizontal(frames, visual_rhythm_path,vid_name)
    elif modality == 'rhythm-V':    
        create_visual_rhythm_mean_vertical(frames, visual_rhythm_path,vid_name)
    elif modality == 'rhythm-HV':
        create_visual_rhythm_mean_horizontal_vertical(frames, visual_rhythm_path,vid_name)    
    elif modality == 'rhythm-DTD':
        visual_rhythm_diagonal_top_down(frames, visual_rhythm_path,vid_name)
    elif modality == 'rhythm-DDT':
        visual_rhythm_diagonal_down_top(frames, visual_rhythm_path,vid_name)
    elif modality == 'gradients':
        gradients_path = out_full_path+'/gradient_%05d.jpg'
        create_visual_rhythm_with_gradients(frames, gradients_path,vid_name)
    elif modality =='hog':
        hog_path = out_full_path+'/hog_%05d.jpg'
        create_images_hog(frames, hog_path, vid_name)
    elif modality == 'rhythm-OF':
        create_visual_rhythm_from_optical_flow(visual_rhythm_path, out_full_path, vid_name)
    elif modality == 'HOG-OF':
        create_HOG_from_optical_flow(out_full_path, vid_name)
    return True;

def create_train_test_files(path):
    '''
        Create train and test files to visual rhythm
    '''
    path_train = path + '/ucf101/train_flow_split{:d}.txt'
    path_test = path + '/ucf101/val_flow_split{:d}.txt'

    path_train_rhythm = path + '/ucf101/train_rhythm_split{:d}.txt'
    path_test_rhythm = path + '/ucf101/val_rhythm_split{:d}.txt'

    def line2rec(line):
        items = line.strip().split(' ')
        return '{0} {1} {2}'.format(items[0],1,items[2])
    splits = []
    for i in range(1, 4):
        train_list = [line2rec(x) for x in open(path_train.format(i))]
        test_list = [line2rec(x) for x in open(path_test.format(i))]
        print(path_train.format(i))
        new_train = open(path_train_rhythm.format(i),'w')
        new_test = open(path_test_rhythm.format(i),'w')
        for line in train_list:
            new_train.write('{}\n'.format(line))
        for line in test_list:
            new_test.write('{}\n'.format(line))
        new_train.close()
        new_test.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='extract visual rhythm ')
    parser.add_argument('--src_dir', type=str, default='./UCF-101',
                        help='path to the video data')
    parser.add_argument('--out_dir', type=str, default='./ucf101_frames',
                        help='path to store visual thythm')
    parser.add_argument('--new_width', type=int, default=0, help='resize image width')
    parser.add_argument('--new_height', type=int, default=0, help='resize image height')

    parser.add_argument('--num_worker', type=int, default=8)
    parser.add_argument('--num_gpu', type=int, default=2, help='number of GPU')
    parser.add_argument('--ext', type=str, default='avi', choices=['avi','mp4'],
                        help='video file extensions')
    parser.add_argument('--modality', '-m', metavar='MODALITY', default='rhythm-H',
                        choices=['rhythm-H', 'rhythm-V', 'rhythm-HV', 'rhythm-DTD', 'rhythm-OF', 'rhythm-DDT', 'gradients','hog', 'HOG-OF'],
                        help='modality: rhythm-H | rhythm-V | rhythm-HV | rhythm-DTD | rhythm-OF | rhythm-DDT | gradients | hog | HOF-OF')
    parser.add_argument('--type_gradient', '-tg', metavar='GRADIENT', default='gradient_x',
                        choices=['gradient_x', 'gradient_y'],
                        help='modality: gradient_x | gradient_y')

    args = parser.parse_args()
    out_path = args.out_dir
    src_path = args.src_dir
    num_worker = args.num_worker
    ext = args.ext
    modality = args.modality
    new_size = (args.new_width, args.new_height)
    NUM_GPU = args.num_gpu

    if not os.path.isdir(out_path):
        print('creating folder: '+out_path)
        os.makedirs(out_path)

    vid_list = glob.glob(src_path+'/*/*.'+ext)
    pool = Pool(num_worker)
    pool.map(run_create_images, zip(vid_list, range(len(vid_list))))
