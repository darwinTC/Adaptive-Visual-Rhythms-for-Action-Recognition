"""
    author : DarwinTC (darwin.ttito.c@gmail.com)
    data : 20/04/2018
"""
import pandas as pd
import argparse
import itertools
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch Two-Stream Action Recognition - Test')
parser.add_argument('-s', '--split', default=-1, type=int, metavar='S',
                        help='which split of data to work on (default: 1)')

def xfrange(start, stop, step):
    """
        New definition to range for a loop
    """
    i = 0
    while start + i * step <= stop:
        yield start + i * step
        i += 1

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    cm = cm if normalize else cm.astype(int)
    thresh = cm.max() / 2.
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def create_confusion_matrix(data_complete, ground_truth, class_names):
    """
    This function creates two files .csv contain the confusion
    matrices without and with normalization
    """
    matrix = np.zeros((101,101)).astype(int)
    for i, data in enumerate(data_complete):
        index = np.argmax(data)
        matrix[ground_truth[i],index] += 1
    
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(matrix, classes=class_names,
                      title='Confusion matrix, without normalization')
    
    #plt.show()
    print(matrix)
    matrix_ = pd.DataFrame(matrix)
    matrix_.to_csv("./confusion_matrix.csv")
 
   
    matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")
    matrix = np.round(matrix,2)
    print(matrix)
    matrix = pd.DataFrame(matrix)
    matrix.to_csv("./confusion_matrix2.csv")

def obtain_acurracy(data_complete, ground_truth, text='rgb'):
    """
    This function calculates the value of the acurracy of a
    certain test(rgb, visual rhythm or optical flow)
    """
    match_count = 0
    for i, data in enumerate(data_complete):
        index = np.argmax(data)
        if ground_truth[i]==index:
            match_count += 1
    text = "Accuracy for "+text+" : %4.4f"
    acurracy = float(match_count)/len(data_complete)
    return text, acurracy

def print_acurracy(data, ground_truth, text = 'rgb'):
    """
    This function shows the acurracy
    """
    tex_, acurracy_ = obtain_acurracy(data, ground_truth, text)
    print(tex_ % (acurracy_))

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def fusion_three_net(first_data, second_data, third_data,ground_truth,
                     class_name, upper_bound = 10, step = 0.5):
    """
    This function calculates the best parameters configuration and
    acurracy for the fusion of three network(rgb, rhythm and flow)
    """
    best_parameters = (0,0,0,0)
    text = ''
    # search the best parameters in the fusion process
    for i in xfrange(0, upper_bound, step):
        for j in xfrange(0,upper_bound, step):
            for z in xfrange(0,upper_bound, step):
                new_data = i*first_data + j*second_data + z*third_data
                text,acurracy = obtain_acurracy(new_data, ground_truth,'fusion')
                if acurracy > best_parameters[3]:
                    best_parameters = (i,j,z,acurracy)
     
    # print the acurracy of the fusion data with the best parameters configuracion
    text = "Parameters : (%0.1f, %0.1f, %0.1f), " + text
    print(text % best_parameters)

def fusion_two_net(first_data, second_data, ground_truth, 
                   class_name, first_text ='rgb', second_text = 'rhythm', 
                   upper_bound = 10, step = 0.5):
    """
    This function calculates the best parameter configuration and
    acurracy for two certain network of the three used
    """
    #result=np.array([]);
    best_parameters = (0,0,0)
    text = ''
    # search the best parameters in the fusion process
    for i in xfrange(0,upper_bound,step):
        for j in xfrange(0,upper_bound,step):
            new_data = i*first_data + j*second_data
            text,acurracy = obtain_acurracy(new_data, ground_truth,'fusion')
            if acurracy > best_parameters[2]:
                best_parameters = (i,j,acurracy)
                result=new_data
    #np.save("ucf101_rhythm_union.npy", result) 
    # print the acurracy of the fusion date with the best parameters configuration
    text = "Parameters : (%0.1f, %0.1f), " + text
    print(text%(best_parameters))

def obtain_ground_truth(path_file):
    file_ = open(path_file, "r")
    lines_file = file_.readlines() 

    ground_truth = []
    class_name = [None]*101
    for line in lines_file:
        line_info = line.split(' ')
        line_name = line.split('_')
        ground_truth.append(int(line_info[2]))
        class_name[int(line_info[2])]=line_name[1]
    
    class_name = np.array(class_name)
    np.array(ground_truth)
    return class_name, np.array(ground_truth)

def final_fusion_two_network(data1, data2, ground_truth, label1, label2, upper_bound=10, step=0.5):
    best_parameters = (0,0,0)
    text = ''
    # search the best parameters in the fusion process
    for i in xfrange(0, upper_bound, step):
        for j in xfrange(0,upper_bound, step):
                new_data = []                
                for k in range(0,3):
                    new_data.append(i*data1[k] + j*data2[k])

                text,acurracy1 = obtain_acurracy(new_data[0], ground_truth[0],'fusion')
                text,acurracy2 = obtain_acurracy(new_data[1], ground_truth[1],'fusion')
                text,acurracy3 = obtain_acurracy(new_data[2], ground_truth[2],'fusion')
                acurracy = (acurracy1+acurracy2+acurracy3)/3
                if acurracy > best_parameters[2]:
                    best_parameters = (i,j,acurracy)
     
    # print the acurracy of the fusion data with the best parameters configuracion
    text = "Parameters to ("+label1+", "+label2+") : (%0.1f, %0.1f), " + text
    print(text % best_parameters)

def final_fusion_three_network(data_rgb, data_flow, data_rhythm, ground_truth,upper_bound=10, step=0.5):
    best_parameters = (0,0,0,0)
    text = ''
    # search the best parameters in the fusion process
    for i in xfrange(0, upper_bound, step):
        for j in xfrange(0,upper_bound, step):
            for z in xfrange(0,upper_bound, step):
                new_data = []                
                for k in range(0,3):
                    new_data.append(i*data_rgb[k] + j*data_rhythm[k] + z*data_flow[k])

                text,acurracy1 = obtain_acurracy(new_data[0], ground_truth[0],'fusion')
                text,acurracy2 = obtain_acurracy(new_data[1], ground_truth[1],'fusion')
                text,acurracy3 = obtain_acurracy(new_data[2], ground_truth[2],'fusion')
                acurracy = (acurracy1+acurracy2+acurracy3)/3
                if acurracy > best_parameters[3]:
                    best_parameters = (i,j,z,acurracy)
     
    # print the acurracy of the fusion data with the best parameters configuracion
    text = "Parameters : (%0.1f, %0.1f, %0.1f), " + text
    print(text % best_parameters)
    
def final_result():
    print("Obtaining final result.....")
    # read result of each model
    data_rgb=[]
    data_flow=[]
    data_rhythm=[]
    ground_truth=[]
    for i in range(1,4):
        path_file = 'splits/val_split'+str(i)+'.txt' 
        path_file_rgb = 'results/ucf101_rgb2_resnet152_s'+str(i)+'.npy'
        path_file_flow = 'results/ucf101_flow_resnet152_s'+str(i)+'.npy'
        path_file_rhythm = 'results/ucf101_rhythm2_resnet152_s'+str(i)+'.npy'
        data_rgb.append(np.load(path_file_rgb))
        data_flow.append(np.load(path_file_flow))
        data_rhythm.append(np.load(path_file_rhythm))
        ground_truth.append(obtain_ground_truth(path_file)[1])
    
    final_fusion_two_network(data_rgb, data_flow, ground_truth, 'rgb','flow')
    final_fusion_two_network(data_rgb, data_rhythm, ground_truth, 'rgb','rhythm')
    final_fusion_two_network(data_rhythm, data_flow, ground_truth, 'rhythm','flow')
    final_fusion_three_network(data_rgb, data_flow, data_rhythm, ground_truth)
    

def partial_result(split):
    # read result of each model
    data_rgb = np.load('results/ucf101_rgb2_resnet152_s'+str(split)+'.npy')
    data_rhythm = np.load('results/ucf101_rhythm_resnet152_s'+str(split)+'.npy')   
    data_flow = np.load('results/ucf101_flow_resnet152_s'+str(split)+'.npy')    

    # read the ground truth date
    path_file = 'splits/val_split'+str(split)+'.txt' 
    
    class_name, ground_truth = obtain_ground_truth(path_file)
 
    # print the acurracy of the three simple data
    print_acurracy(data_rgb, ground_truth, 'rgb')
    print_acurracy(data_rhythm, ground_truth, 'rhythm')
    print_acurracy(data_flow, ground_truth, 'flow')
    
    #create_confusion_matrix(data_rhythm, ground_truth, class_name)
    fusion_two_net(data_rgb, data_rhythm, ground_truth, class_name,'rgb','rhythm')    
    fusion_two_net(data_rgb, data_flow, ground_truth, class_name,'rgb','flow')
    fusion_two_net(data_flow, data_rhythm, ground_truth, class_name,'flow','rhythm')    
    
    fusion_three_net(data_rgb,data_rhythm,data_flow,ground_truth,class_name)   
    
def main():
    args = parser.parse_args()
    if args.split==-1: 
        final_result()
    else:
        partial_result(args.split)   

if __name__ == '__main__':
    main()
