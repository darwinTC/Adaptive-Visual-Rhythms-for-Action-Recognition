import os
import time
import argparse
import shutil
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

import video_transforms
import models
import datasets

model_path = './checkpoints'
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

dataset_names = sorted(name for name in datasets.__all__)

parser = argparse.ArgumentParser(description='PyTorch Two-Stream Action Recognition')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--settings', metavar='DIR', default='./datasets/settings',
                    help='path to datset setting files')
parser.add_argument('--modality', '-m', metavar='MODALITY', default='rgb',
                    choices=["rgb", "flow", "rhythm"],
                    help='modality: rgb | flow | rhythm')
parser.add_argument('--dataset', '-d', default='ucf101',
                    choices=["ucf101", "hmdb51"],
                    help='dataset: ucf101 | hmdb51')
parser.add_argument('--arch', '-a', metavar='ARCH', default='rgb_resnet152',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: rgb_vgg16)')
parser.add_argument('-s', '--split', default=2, type=int, metavar='S',
                    help='which split of data to work on (default: 1)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=250, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=25, type=int,
                    metavar='N', help='mini-batch size (default: 50)')
parser.add_argument('--iter-size', default=5, type=int,
                    metavar='I', help='iter size as in Caffe to reduce memory usage (default: 5)')
parser.add_argument('--new_length', default=1, type=int,
                    metavar='N', help='length of sampled video frames (default: 1)')
parser.add_argument('--new_width', default=340, type=int,
                    metavar='N', help='resize width (default: 340)')
parser.add_argument('--new_height', default=299, type=int,
                    metavar='N', help='resize height (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_steps', default=[100, 200], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('-pf','--print-freq', default=50,  type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('-sf','--save-freq', default=25, type=int,
                    metavar='N', help='save frequency (default: 25)')
parser.add_argument('--resume', default='./checkpoints', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

best_prec1 = 0
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def createNewDataset(fileNameRead, fileNameWrite):
    '''
        Generate new train file to train the network with two RGB images,
        taking the first and second images randomly in the first and second 
        half respectively
    '''
    newPathFile = os.path.join(args.settings, args.dataset, fileNameWrite)
    train_setting_file = fileNameRead % (args.modality, args.split)
    pathFile = os.path.join(args.settings, args.dataset, train_setting_file)    

    file_ = open(pathFile,'r')
    linesFile = file_.readlines()
    detallLines = list()
    for line in linesFile:
        lineInfo = line.split(' ')
        first_frame = random.randint(0, int(lineInfor[1])//2-4)
        second_frame = int(lineInfor[1])//2 + random.randint(0, int(lineInfor[1])//2-4)
        detallLines.append('{} {} {}'.format(lineInfo[0], first_frame, lineInfo[2]))
        detallLines.append('{} {} {}'.format(lineInfo[0], second_frame, lineInfo[2]))
    open(newPathFile,'w').writelines(detallLines)

def main():
    global args, best_prec1
    args = parser.parse_args()

    print(args.modality+" network trained whith the split "+str(args.split)+".")

    # create model
    print("Building model ... ")
    exits_model, model = build_model(int(args.start_epoch))
    if not exits_model:
        return 
    else:
        print("Model %s is loaded. " % (args.arch))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # create file where we allocate the models by each args.save_freq epochs
    if not os.path.exists(args.resume):
        os.makedirs(args.resume)
    print("Saving everything to directory %s." % (args.resume))

    cudnn.benchmark = True

    # Data transforming
    if args.modality == "rgb" or args.modality == "rhythm" or args.modality == "history":
        is_color = True
        scale_ratios = [1.0, 0.875, 0.75, 0.66]
        clip_mean = [0.485, 0.456, 0.406] * args.new_length
        clip_std = [0.299, 0.224, 0.225] * args.new_length
    elif args.modality == "flow":
        is_color = False
        scale_ratios = [1.0, 0.875, 0.75]
        clip_mean = [0.5, 0.5] * args.new_length
        clip_std = [0.226, 0.226] * args.new_length
    else:
        print("No such modality. Only rgb and flow supported.")

    new_size= 299 if args.arch =='rgb_inception_v3' else 224

    normalize = video_transforms.Normalize(mean=clip_mean,
                                           std=clip_std)
    train_transform = video_transforms.Compose([
            #video_transforms.Scale((256)),
            video_transforms.MultiScaleCrop((new_size, new_size), scale_ratios),
            video_transforms.RandomHorizontalFlip(),
            video_transforms.ToTensor(),
            normalize,
        ])

    val_transform = video_transforms.Compose([
            # video_transforms.Scale((256)),
            video_transforms.CenterCrop((new_size)),
            video_transforms.ToTensor(),
            normalize,
        ])
    
    #createNewDataset("train_%s_split%d.txt" , "new_train.txt")
    #createNewDataset("val_%s_split%d.txt", "new_test.txt")
    
    # data loading
    #train_setting_file = 'new_train.txt'
    modality_ = "rgb" if (args.modality == "rhythm" or args.modality == "history") else args.modality    
    train_setting_file = "train_%s_split%d.txt" % (modality_, args.split)
    train_split_file = os.path.join(args.settings, args.dataset, train_setting_file)
    #val_setting_file = 'new_test.txt'
    val_setting_file = "val_%s_split%d.txt" % (modality_, args.split)
    val_split_file = os.path.join(args.settings, args.dataset, val_setting_file)
    if not os.path.exists(train_split_file) or not os.path.exists(val_split_file):
        print("No split file exists in %s directory. Preprocess the dataset first" % (args.settings))

    train_dataset = datasets.__dict__['dataset'](root=args.data,
                                                    source=train_split_file,
                                                    phase="train",
                                                    modality=args.modality,
                                                    is_color=is_color,
                                                    new_length=args.new_length,
                                                    new_width=args.new_width,
                                                    new_height=args.new_height,
                                                    video_transform=train_transform)
    val_dataset = datasets.__dict__['dataset'](root=args.data,
                                                  source=val_split_file,
                                                  phase="val",
                                                  modality=args.modality,
                                                  is_color=is_color,
                                                  new_length=args.new_length,
                                                  new_width=args.new_width,
                                                  new_height=args.new_height,
                                                  video_transform=val_transform)

    print('{} samples found, {} train samples and {} test samples.'.format(len(val_dataset)+len(train_dataset),
                                                                           len(train_dataset),
                                                                           len(val_dataset)))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    
    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = 0.0
        if (epoch + 1) % args.save_freq == 0:
            prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if (epoch + 1) % args.save_freq == 0:
            checkpoint_name = "%03d_%s" % (epoch + 1, "checkpoint_"+args.modality+"_split_"+str(args.split)+".pth.tar")
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint_name, args.resume)
    
def build_model(resume_epoch):
    is_new = (resume_epoch==0)
    found = True
    num_classes = 51 if args.dataset =='hmdb51' else 101
    print("====================")
    model = models.__dict__[args.arch](pretrained=is_new, num_classes=num_classes)
    if not is_new:
        path = os.path.join(model_path,'{0:03d}_checkpoint.pth.tar'.format(resume_epoch))
        if os.path.isfile(path):    
            print('loading checkpoint {0:03d} ...'.format(resume_epoch))    
            params = torch.load(path)
            model.load_state_dict(params['state_dict'])
            print('loaded checkpoint {0:03d}'.format(resume_epoch))
        else:
            print('ERROR: No checkpoint found')
            found = False
    model.cuda()
    return found, model

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    optimizer.zero_grad()
    loss_mini_batch = 0.0
    acc_mini_batch = 0.0

    for i, (input, target) in enumerate(train_loader):

        input = input.float().cuda(async=True)
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        if args.arch =='rgb_inception_v3':
            output, __= model(input_var)
        else:   
            output= model(input_var)
        # measure accuracy and record loss
        prec1, prec3 = accuracy(output.data, target, topk=(1, 3))
        acc_mini_batch += prec1[0]
        loss = criterion(output, target_var)
        loss = loss / args.iter_size
        loss_mini_batch += loss.data[0]
        loss.backward()

        if (i+1) % args.iter_size == 0:
            # compute gradient and do SGD step
            optimizer.step()
            optimizer.zero_grad()

            # losses.update(loss_mini_batch/args.iter_size, input.size(0))
            # top1.update(acc_mini_batch/args.iter_size, input.size(0))
            losses.update(loss_mini_batch, input.size(0))
            top1.update(acc_mini_batch/args.iter_size, input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            loss_mini_batch = 0
            acc_mini_batch = 0

            if (i+1) % args.print_freq == 0:

                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       epoch, i+1, len(train_loader)+1, batch_time=batch_time, loss=losses, top1=top1))

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input = input.float().cuda(async=True)
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec3 = accuracy(output.data, target, topk=(1, 3))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top3.update(prec3[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top3=top3))

    print(' * Prec@1 {top1.avg:.3f} Prec@3 {top3.avg:.3f}'
          .format(top1=top1, top3=top3))

    return top1.avg

def save_checkpoint(state, is_best, filename, resume_path):
    cur_path = os.path.join(resume_path, filename)
    best_path = os.path.join(resume_path, 'model_best_'+args.modality+'_split_'+str(args.split)+'.pth.tar')
    torch.save(state, cur_path)
    if is_best:
        shutil.copyfile(cur_path, best_path)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 150 epochs"""

    decay = 0.1 ** (sum(epoch >= np.array(args.lr_steps)))
    lr = args.lr * decay
    print("Current learning rate is %4.6f:" % lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()

