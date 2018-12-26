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



def infer_rgb(net):
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
    tic_loading_rgb = time.time()
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
    toc_loading_rgb = time.time() 
    if args.gpus is not None:
        devices = [args.gpus[i] for i in range(args.workers)]
    else:
        devices = list(range(args.workers))
    net_tic = time.time() 
    net = torch.nn.DataParallel(net.cuda(devices[0]), device_ids=devices)
    net.eval()
    net_toc = time.time()  
    max_num = args.max_num if args.max_num > 0 else len(rgb_data_loader.dataset)
    data_gen = enumerate(rgb_data_loader)
    video_pred_tic = time.time() 
    for i, (data) in data_gen:
        if i >= max_num:
            break
        rst = eval_video(data, 3, net, 'RGB')
        video_pred = np.argmax(np.mean(rst[0], axis=0))
    video_pred_toc = time.time() 
    print("loading RGB DATA Loader in {}, Net.eval in {}, VIDEO PRED in {} ".format(toc_loading_rgb - tic_loading_rgb, net_toc - net_tic, video_pred_toc-video_pred_tic)) 
    print("this is from rgb and got ", make_hmdb()[video_pred])
    return video_pred


def infer_of(OF_DIR, net):
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
    tic_loading_of = time.time()
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
    toc_loading_of  = time.time() 
    if args.gpus is not None:
        devices = [args.gpus[i] for i in range(args.workers)]
    else:
        devices = list(range(args.workers))
    net_tic = time.time() 
    net = torch.nn.DataParallel(net.cuda(devices[0]), device_ids=devices)
    net.eval()
    net_toc = time.time() 
    max_num = args.max_num if args.max_num > 0 else len(of_data_loader.dataset)
    data_gen = enumerate(of_data_loader)
    video_pred_tic = time.time() 
    for i, (data) in data_gen:
        if i >= max_num:
            break
        rst = eval_video(data, 10, net, 'Flow')
        video_pred = np.argmax(np.mean(rst[0], axis=0))
    video_pred_toc = time.time() 
    print("loading OF DATA Loader in {}, Net.eval in {}, VIDEO PRED in {} ".format(toc_loading_of - tic_loading_of, net_toc - net_tic, video_pred_toc-video_pred_tic)) 
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
    '''
    #######LOADING RGB NET####### 
    #before = time.time()
    rgb_net = TSN(num_class, 1, 'RGB',
	      base_model=args.arch,
	      consensus_type=args.crop_fusion_type,
	      dropout=args.dropout)
    rgb_checkpoint = torch.load(args.rgb_weights)
    print("model epoch {} best prec@1: {}".format(rgb_checkpoint['epoch'], rgb_checkpoint['best_prec1']))
    base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(rgb_checkpoint['state_dict'].items())}
    rgb_net.load_state_dict(base_dict)
    #after = time.time()
    #print("loading rgb_net: ", after-before)
    ########INFERENCING RGB#######
    avg_time_rgb = 0
    N = 10
    for i in range(N):
        tic1= time.time()
        infer_rgb(rgb_net)
        toc1 = time.time() 
        print("how long it took", toc1 -tic1)
        if i == 0:
            pass
        else:
            avg_time_rgb += toc1-tic1 
    print("inferencing rgb in ", avg_time_rgb/(N-1)) 
    
    #######EXTRACT OPTICAL FLOW#######
    OF_DIR = '/home/haabibi/fall_detection/tsn-pytorch_tmp/sample_img_frames2' 
    tic2 = time.time()
    streaming(args.root_path, OF_DIR )
    toc2 = time.time() 
    print("streaming " + str(toc2 - tic2))
    '''
    #######LOADING OF NET####### 
    OF_DIR = '/home/haabibi/fall_detection/tsn-pytorch_tmp/sample_img_frames2' 
    tic_load_of = time.time()
    of_net = TSN(num_class, 1, 'Flow',
	      base_model=args.arch,
	      consensus_type=args.crop_fusion_type,
	      dropout=args.dropout)
    of_checkpoint = torch.load(args.of_weights)
    print("model epoch {} best prec@1: {}".format(of_checkpoint['epoch'], of_checkpoint['best_prec1']))
    of_base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(of_checkpoint['state_dict'].items())}
    of_net.load_state_dict(of_base_dict)
    toc_load_of = time.time() 
    print("loading of_net: ", toc_load_of-tic_load_of)
    ########INFERENCING OF#######
    avg_time_of = 0
    N = 10
    for i in range(N):
        tic1= time.time()
        infer_of(OF_DIR, of_net)
        toc1 = time.time() 
        print("how long it took", toc1 -tic1)
        if i == 0:
            print(
            pass
        else:
            avg_time_of += toc1-tic1 
    print("inferencing rgb in ", avg_time_of/(N-1)) 

