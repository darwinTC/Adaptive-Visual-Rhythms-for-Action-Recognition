
import argparse
import os
import glob
import random
import fnmatch

def parse_directory(path, rgb_prefix='img_', flow_x_prefix='flow_x_', flow_y_prefix='flow_y_'):
    """
    Parse directories holding extracted frames from standard benchmarks
    """
    print('parse frames under folder {}'.format(path))
    frame_folders = glob.glob(os.path.join(path, '*'))

    def count_files(directory, prefix_list):
        lst = os.listdir(directory)
        cnt_list = [len(fnmatch.filter(lst, x+'*')) for x in prefix_list]
        return cnt_list

    rgb_counts = {}
    flow_counts = {}
    for i,f in enumerate(frame_folders):
        all_cnt = count_files(f, (rgb_prefix, flow_x_prefix, flow_y_prefix))
        k = f.split('/')[-1]
        rgb_counts[k] = all_cnt[0]
        x_cnt = all_cnt[1]
        y_cnt = all_cnt[2]
        if x_cnt != y_cnt:
            raise ValueError('x and y direction have different number of flow images. video: '+f)
        flow_counts[k] = x_cnt
        if i % 200 == 0:
            print('{} videos parsed'.format(i))
    
    print('frame folder analysis done')
    return rgb_counts, flow_counts


def build_split_list(split_tuple, frame_info, split_idx, shuffle=False):
    split = split_tuple[split_idx]

    def build_set_list(set_list):
        rgb_list, flow_list = list(), list()
        for item in set_list:
            rgb_cnt = frame_info[0][item[0]]
            flow_cnt = frame_info[1][item[0]]
            rgb_list.append('{} {} {}\n'.format(item[0], rgb_cnt, item[1]))
            flow_list.append('{} {} {}\n'.format(item[0], flow_cnt, item[1]))
        if shuffle:
            random.shuffle(rgb_list)
            random.shuffle(flow_list)
        return rgb_list, flow_list
    print('here')
    train_rgb_list, train_flow_list = build_set_list(split[0])
    test_rgb_list, test_flow_list = build_set_list(split[1])
    return (train_rgb_list, test_rgb_list), (train_flow_list, test_flow_list)


def parse_dataset_splits(dataset = 'ucf101_splits'):
    print(dataset)
    class_ind = [x.strip().split() for x in open(dataset+'/classInd.txt')]
    print(class_ind)
    class_mapping = {x[1]:int(x[0])-1 for x in class_ind}

    def line2rec(line):
        items = line.strip().split('/')
        label = class_mapping[items[0]]
        vid = items[1].split('.')[0]
        return vid, label

    splits = []
    for i in xrange(1, 4):
        train_list = [line2rec(x) for x in open(dataset+'/trainlist{:02d}.txt'.format(i))]
        test_list = [line2rec(x) for x in open(dataset+'/testlist{:02d}.txt'.format(i))]
        splits.append((train_list, test_list))
    return splits

def create_files_HMDB51(path_file='hmdb51_splits'):
    """
        Create train and test files standard
        author : Darwin TC
        date: 03/05/2018
    """
    def line2rec(line, splits):
        detail_line = line.split(' ')
        number_split = splits[-1:]
        return (detail_line[0],detail_line[1], int(number_split))

    def create_splits():
        """
            Read the initial traing and test files of the HMDB51 dataset to afterward create
            just 3 files for test and train, sorting of a standard way similar to UCF101 dataset
        """        
        # load split file
        class_files = glob.glob(path_file+'/*split*.txt')
        class_list = set([x[x.find('/')+1:x.find('_test')] for x in class_files])
        class_dict = {x: str(i+1) for i,x in enumerate(class_list)}
        splits_train = [[],[],[]]
        splits_test = [[],[],[]]
        for name in class_files:
            part_name = name.split('/')[-1][:-4].split('_')
            test_list = [line2rec(x, part_name[2] if len(part_name)==3 else part_name[3]) for x in open(name)]
            for line in test_list:
                name_class = part_name[0] if len(part_name)==3 else (part_name[0]+'_'+part_name[1])
                name_clip = line[0][:line[0].find('.avi')]
                number_split = line[2]
                type_data = line[1] # is a data train or test(1,2)
                new_data = name_class+'/'+name_clip+'\n'
                if type_data == '1':
                    splits_train[number_split-1].append(new_data)
                elif type_data == '2':
                    splits_test[number_split-1].append(new_data)
        return class_dict, splits_train, splits_test
    
    def create_files(class_names, train, test, out_path ='.'):
        """
           method to create the initial files of class names, train and test
        """        
        list_label = ['{} {}\n'.format(v,k) for k,v in class_names.items()] 
        open(os.path.join(out_path, 'classInd.txt'), 'w').writelines(list_label)
        for i in range(1,4):
            open(os.path.join(out_path, 'trainlist{:02d}.txt'.format(i)), 'w').writelines(train[i-1])
            open(os.path.join(out_path, 'testlist{:02d}.txt'.format(i)), 'w').writelines(test[i-1])
            
    #create list of class, train and test
    class_names, train, test = create_splits()
    for i in range(3):
        train[i]=sorted(train[i])
        test[i]=sorted(test[i])
    create_files(class_names,train, test)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ucf101', choices=['ucf101', 'hmdb51'])
    parser.add_argument('--frame_path', type=str, default='./ucf101_frames',
                        help="root directory holding the frames")
    parser.add_argument('--out_list_path', type=str, default='./settings')

    parser.add_argument('--rgb_prefix', type=str, default='img_',
                        help="prefix of RGB frames")
    parser.add_argument('--flow_x_prefix', type=str, default='flow_x',
                        help="prefix of x direction flow images")
    parser.add_argument('--flow_y_prefix', type=str, default='flow_y',
                        help="prefix of y direction flow images", )

    parser.add_argument('--num_split', type=int, default=3,
                        help="number of split building file list")
    parser.add_argument('--shuffle', action='store_true', default=False)

    args = parser.parse_args()

    dataset = args.dataset
    frame_path = args.frame_path
    rgb_p = args.rgb_prefix
    flow_x_p = args.flow_x_prefix
    flow_y_p = args.flow_y_prefix
    num_split = args.num_split
    out_path = args.out_list_path
    shuffle = args.shuffle

    out_path = os.path.join(out_path,dataset)
    if not os.path.isdir(out_path):
        print("creating folder: "+out_path)
        os.makedirs(out_path)

    # operation
    print('processing dataset {}'.format(dataset))
    split_tp = parse_dataset_splits(dataset+'_splits')

    f_info = parse_directory(frame_path, rgb_p, flow_x_p, flow_y_p)  
    
    print('writing list files for training/testing')
    for i in xrange(max(num_split, len(split_tp))):
        lists = build_split_list(split_tp, f_info, i, shuffle)
        open(os.path.join(out_path, 'train_rgb_split{}.txt'.format(i + 1)), 'w').writelines(lists[0][0])
        open(os.path.join(out_path, 'val_rgb_split{}.txt'.format(i + 1)), 'w').writelines(lists[0][1])
        open(os.path.join(out_path, 'train_flow_split{}.txt'.format(i + 1)), 'w').writelines(lists[1][0])
        open(os.path.join(out_path, 'val_flow_split{}.txt'.format(i + 1)), 'w').writelines(lists[1][1])

