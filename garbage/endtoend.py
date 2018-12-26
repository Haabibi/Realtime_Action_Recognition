import argparse
import time
import os
import numpy as np
import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix

from dataset2 import TSNDataSet
from models import TSN
from transforms import *
from ops import ConsensusModule
from streaming2 import streaming
import threading
def make_ucf():
    index_dir = '/cmsdata/hdd2/cmslab/haabibi/UCF101CLASSIND.txt'
    index_dict = {}
    with open(index_dir) as f:
        for line in f.readlines():
            s = line.split(' ')
            index_dict.update({int(s[0])-1: s[1]})
    return index_dict 

def make_hmdb():
    index_dir = '/cmsdata/hdd2/cmslab/haabibi/HMDB51CLASSIND.txt'
    index_dict = {}
    with open(index_dir) as f:
        for line in f.readlines():
            s = line.split(' ')
            index_dict.update({int(s[0]): s[1]})
    return index_dict 

def eval_video(video_data, length, net, modality):
    data = video_data
    num_crops = args.test_crops 
       
    input_var = torch.autograd.Variable(data.view(-1, length, data.size(2), data.size(3)),
					volatile=True)
    rst = net(input_var).data.cpu().numpy().copy()
    output =  rst.reshape((args.test_crops, 25, num_class)).mean(axis=0).reshape((25, 1, num_class)).shape
    return rst.reshape((num_crops, args.test_segments, num_class)).mean(axis=0).reshape(
	(args.test_segments, 1, num_class))



def infer_rgb():
    tic = time.time()
    net = TSN(num_class, 1, 'RGB',
	      base_model=args.arch,
	      consensus_type=args.crop_fusion_type,
	      dropout=args.dropout)
    checkpoint = torch.load(args.rgb_weights)
    print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))
    base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
    net.load_state_dict(base_dict)
    if args.test_crops == 1:
        cropping = torchvision.transforms.Compose([
	    GroupScale(net.scale_size),
	    GroupCenterCrop(net.input_size),
	])
    elif args.test_crops == 10:
        cropping = torchvision.transforms.Compose([
	    GroupOverSample(net.input_size, net.scale_size)
	])
    else:
        raise ValueError("Only 1 and 10 crops are supported while we got {}".format(args.test_crops))
    rgb_data_loader = torch.utils.data.DataLoader(
	    TSNDataSet(os.path.join(args.root_path), num_segments=args.test_segments,
		       new_length=1,
		       modality='RGB',
		       image_tmpl="img_{:05d}.jpg",
		       test_mode=True,
		       transform=torchvision.transforms.Compose([
			   cropping,
			   Stack(roll=args.arch == 'BNInception'),
			   ToTorchFormatTensor(div=args.arch != 'BNInception'),
			   GroupNormalize(net.input_mean, net.input_std),
		       ])),
	    batch_size=80, shuffle=False,
	    num_workers=args.workers * 2, pin_memory=True)

    if args.gpus is not None:
        devices = [args.gpus[i] for i in range(args.workers)]
    else:
        devices = list(range(args.workers))

    net = torch.nn.DataParallel(net.cuda(devices[0]), device_ids=devices)
    net.eval()

    max_num = args.max_num if args.max_num > 0 else len(rgb_data_loader.dataset)
    data_gen = enumerate(rgb_data_loader)
    for i, (data) in data_gen:
        if i >= max_num:
            break
        rst = eval_video(data, 3, net, 'RGB')
        video_pred = np.argmax(np.mean(rst[0], axis=0))
    print("this is from rgb and got ", make_hmdb()[video_pred])
    return video_pred


def infer_of(OF_DIR):
    print("this is test_segments from infer_of()", args.test_segments)
    net = TSN(num_class, 1, 'Flow',
	      base_model=args.arch,
	      consensus_type=args.crop_fusion_type,
	      dropout=args.dropout)
    checkpoint = torch.load(args.of_weights)
    print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))
    base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
    net.load_state_dict(base_dict)
    if args.test_crops == 1:
        cropping = torchvision.transforms.Compose([
            GroupScale(net.scale_size),
            GroupCenterCrop(net.input_size),
        ])
    elif args.test_crops == 10:
        cropping = torchvision.transforms.Compose([
            GroupOverSample(net.input_size, net.scale_size)
        ])
    else:
        raise ValueError("Only 1 and 10 crops are supported while we got {}".format(args.test_crops))

    #loading data in 'torch'-readable format
    of_data_loader = torch.utils.data.DataLoader(
        TSNDataSet(OF_DIR, num_segments=args.test_segments,
		       new_length= 5,
		       modality='Flow',
		       image_tmpl=  args.flow_prefix+"{}_{:05d}.jpg",
		       test_mode=True,
		       transform=torchvision.transforms.Compose([
			   cropping,
			   Stack(roll=args.arch == 'BNInception'),
			   ToTorchFormatTensor(div=args.arch != 'BNInception'),
			   GroupNormalize(net.input_mean, net.input_std),
		       ])),
	    batch_size=80, shuffle=False,
	    num_workers=args.workers * 2, pin_memory=True)

    if args.gpus is not None:
        devices = [args.gpus[i] for i in range(args.workers)]
    else:
        devices = list(range(args.workers))

    net = torch.nn.DataParallel(net.cuda(devices[0]), device_ids=devices)
    net.eval()

    max_num = args.max_num if args.max_num > 0 else len(of_data_loader.dataset)
    data_gen = enumerate(of_data_loader)
    for i, (data) in data_gen:
        if i >= max_num:
            break
        rst = eval_video(data, 10, net, 'Flow')
        video_pred = np.argmax(np.mean(rst[0], axis=0))
    print("this is from optical flow! And got ", make_hmdb()[video_pred])
    return video_pred

if __name__=="__main__":
    # options
    parser = argparse.ArgumentParser(
	description="Standard video-level testing")
    parser.add_argument('dataset', type=str, choices=['ucf101', 'hmdb51'])
    parser.add_argument('root_path', type=str)
    parser.add_argument('rgb_weights', type=str)
    parser.add_argument('of_weights', type=str)
    parser.add_argument('--arch', type=str, default="resnet101")
    parser.add_argument('--test_segments', type=int, default=25)
    parser.add_argument('--max_num', type=int, default=-1)
    parser.add_argument('--test_crops', type=int, default=10)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--crop_fusion_type', type=str, default='avg',
			choices=['avg', 'max', 'topk'])
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.7)
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
			help='number of data loading workers (default: 4)')
    parser.add_argument('--gpus', nargs='+', type=int, default=None)
    parser.add_argument('--flow_prefix', type=str, default='flow_')

    args = parser.parse_args()

    if args.dataset == 'ucf101':
        num_class = 101
        index_dict = make_ucf() 
    elif args.dataset == 'hmdb51':
        num_class = 51
        index_dict = make_hmdb()
    else:
        raise ValueError('Unknown dataset '+args.dataset)
  
    avg_time_rgb = 0 
    for _ in range(1000):
        tic1= time.time()
        infer_rgb()
        toc1 = time.time() 
        print("how long it took", toc1 -tic1)
        avg_time_rgb += toc1-tic1 
    #infer_rgb() 
    #print("inferencing rgb and in " +  str(time.time()-tic1))
    print("inferencing rgb in ", avg_time_rgb) 
    #extract optical flow 
    OF_DIR = '/home/haabibi/fall_detection/tsn-pytorch_tmp/sample_img_frames2' 
    #tic2 = time.time()
    #streaming(args.root_path, OF_DIR )
    #toc2 = time.time() 
    #print("streaming " + str(toc2 - tic2))
    #print("avg streaming time", (toc2-tic2)/len(os.listdir(args.root_path))) 
    #tic3 = time.time()
    #infer_of(OF_DIR)
    #print("inferencing of in  "  +  str(time.time() - tic3))  



