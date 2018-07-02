import torch.utils.data as data

import os
import sys
import random
import numpy as np
import cv2
from pipes import quote

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def make_dataset(root, source):
    '''
        Method to obtain the path, duration(number of frame) and target of each
        video of the dataset from a file(train or test) that contain this detalls
        in each line.
    '''
    if not os.path.exists(source):
        print('Setting file %s for the dataset doesnt exist.' % (source))
        sys.exit()
    else:
        clips = []
        with open(source) as split_f:
            data = split_f.readlines()
            for line in data:
                line_info = line.split()
                clip_path = os.path.join(root, line_info[0])
                duration = int(line_info[1])
                target = int(line_info[2])
                item = (clip_path, duration, target)
                clips.append(item)
    return clips

def color(is_color):
    if is_color:
        cv_read_flag = cv2.IMREAD_COLOR         # > 0
    else:
        cv_read_flag = cv2.IMREAD_GRAYSCALE     # = 0    
    return cv_read_flag
    
def read_single_segment(path, offsets, new_height, new_width, new_length, is_color, name_pattern, modality, index):
    '''
        Takes visual_rhythm, history_motion or RGB images, one frame by video,
        and this correspond to some specific images(type of visual_rhythm or
        history_motion). 
    '''
    cv_read_flag = color(is_color)
    interpolation = cv2.INTER_LINEAR    
    sampled_list = []
    frame_name = name_pattern % (index)
    frame_path = os.path.join(path, frame_name)
    cv_img_origin = cv2.imread(frame_path, cv_read_flag)
    if cv_img_origin is None:
       print('Could not load file %s' % (frame_path))
       sys.exit()
       # TODO: error handling here
    if new_width > 0 and new_height > 0:
        # use OpenCV3, use OpenCV2.4.13 may have error
        cv_img = cv2.resize(cv_img_origin, (new_width, new_height), interpolation)
    else:
        cv_img = cv_img_origin

    if modality == "rgb":
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        sampled_list.append(cv_img)
    else:    
        sampled_list.append(np.expand_dims(cv_img, 2))

    clip_input = np.concatenate(sampled_list, axis=2)
    return clip_input 

def read_multiple_segment(path, offsets, new_height, new_width, new_length, is_color, name_pattern):
    '''
        Concatenates 10 consecutive optical flow images for (flow_x and flow_y),
        or (HOG_x and HOG_y) having 20 images concatenate by video in the end.
    '''
    cv_read_flag = color(is_color)
    interpolation = cv2.INTER_LINEAR
    sampled_list = []

    offset = offsets[0]
    for length_id in range(1, new_length+1):
        frame_name_x = name_pattern % ('x', length_id + offset)
        frame_path_x = os.path.join(path, frame_name_x)
        cv_img_origin_x = cv2.imread(frame_path_x, cv_read_flag)
        frame_name_y = name_pattern % ('y', length_id + offset)
        frame_path_y = os.path.join(path, frame_name_y)
        cv_img_origin_y = cv2.imread(frame_path_y, cv_read_flag)
        if cv_img_origin_x is None or cv_img_origin_y is None:
           print('Could not load file %s or %s' % (frame_path_x, frame_path_y))
           sys.exit()
           # TODO: error handling here
        if new_width > 0 and new_height > 0:
            cv_img_x = cv2.resize(cv_img_origin_x, (new_width, new_height), interpolation)
            cv_img_y = cv2.resize(cv_img_origin_y, (new_width, new_height), interpolation)
        else:
            cv_img_x = cv_img_origin_x
            cv_img_y = cv_img_origin_y
        sampled_list.append(np.expand_dims(cv_img_x, 2))
        sampled_list.append(np.expand_dims(cv_img_y, 2))

    clip_input = np.concatenate(sampled_list, axis=2)
    return clip_input


class dataset(data.Dataset):

    def __init__(self,
                 root,
                 source,
                 phase,
                 modality,
                 name_pattern=None,
                 is_color=True,
                 num_segments=1,
                 new_length=1,
                 new_width=0,
                 new_height=0,
                 transform=None,
                 target_transform=None,
                 video_transform=None,
                 approach_VR = 3):
        classes, class_to_idx = find_classes(root) 
        clips = make_dataset(root, source)
        
        if len(clips) == 0:
            raise(RuntimeError('Found 0 video clips in subfolders of: ' + root + '\n'
                               'Check your data directory.'))

        self.root = root
        self.source = source
        self.phase = phase
        self.modality = modality
        self.dataset = source.split('/')[3]
        self.visual_rhythm_approach = approach_VR

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.clips = clips
        self.direction =[]
        if name_pattern:
            self.name_pattern = name_pattern
        else:
            if self.modality == 'rgb':
                self.name_pattern = 'img_%05d.jpg'
            elif self.modality == 'rhythm':
                self.name_pattern = 'visual_rhythm_%05d.jpg'
                # recover the direction by class
                self.direction = [int(line.rstrip('\n')) for line in open('./datasets/settings/'+self.dataset+'/direction.txt')]
            elif self.modality == 'flow':
                self.name_pattern = 'flow_%s_%05d.jpg'
            elif self.modality == 'hog':
                self.name_pattern = 'hog_%s_%05d.jpg'
            elif self.modality == 'history':
                self.name_pattern = 'history_motion_%05d.jpg'

        self.is_color = is_color
        self.num_segments = num_segments
        self.new_length = new_length
        self.new_width = new_width
        self.new_height = new_height

        self.transform = transform
        self.target_transform = target_transform
        self.video_transform = video_transform

    def __getitem__(self, index):
        path, duration, target = self.clips[index]
        average_duration = int(duration / self.num_segments)
        offsets = []

        for seg_id in range(self.num_segments):
            if self.phase == 'train':
                if average_duration >= self.new_length:
                    offset = random.randint(0, average_duration - self.new_length)
                    # No +1 because randint(a,b) return a random integer N such that a <= N <= b.
                    offsets.append(offset + seg_id * average_duration)
                else:
                    offsets.append(0)
            elif self.phase == 'val':
                if average_duration >= self.new_length:
                    offsets.append(int((average_duration - self.new_length + 1)/2 + seg_id * average_duration))
                else:
                    offsets.append(0)
            else:
                print('Only phase train and val are supported.')

        if self.modality == 'rgb':
            clip_input = read_single_segment(path,
                                        offsets,
                                        self.new_height,
                                        self.new_width,
                                        self.new_length,
                                        self.is_color,
                                        self.name_pattern,
                                        self.modality,
                                        1 + offsets[0]  #duration if self.phase == 'train' else  1 + offsets[0] 
                                        )
        elif self.modality == 'rhythm' or self.modality == 'history':
            print(self.visual_rhythm_approach)
            clip_input = read_single_segment(path,
                                        offsets,
                                        self.new_height,
                                        self.new_width,
                                        self.new_length,
                                        self.is_color,
                                        self.name_pattern,
                                        self.modality,
                                        self.direction[target] if self.visual_rhythm_approach==3 else self.visual_rhythm_approach #type of visual rhythm image
                                        )            
        elif self.modality == 'flow' or self.modality == 'hog':
            clip_input = read_multiple_segment(path,
                                        offsets,
                                        self.new_height,
                                        self.new_width,
                                        self.new_length,
                                        self.is_color,
                                        self.name_pattern
                                        )
        else:
            print('No such modality %s' % (self.modality))

        if self.transform is not None:
            clip_input = self.transform(clip_input)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.video_transform is not None:
            clip_input = self.video_transform(clip_input)

        return clip_input, target


    def __len__(self):
        return len(self.clips)
