import time 
import argparse
import numpy as np 
import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix
import cv2
from dataset4 import TSNDataSet 
from models import TSN
from transforms import * 
from ops import ConsensusModule
from streaming4 import streaming
from threading import Thread

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

def eval_video(data, length, net, style):
    input_var = torch.autograd.Variable(data.view(-1, length, data.size(2), data.size(3)), volatile=True)
    rst = net(input_var).data.cpu().numpy().copy()
    output =  rst.reshape((args.test_crops, 25, num_class)).mean(axis=0).reshape((25, 1, num_class)).shape
    return rst.reshape((args.test_crops, 25, num_class)).mean(axis=0).reshape((25, 1, num_class))

def make_infer(style, weights, fifty_data, net): 
    '''
    net = TSN(num_class, 1, style, 
              base_model=args.arch, 
              consensus_type=args.crop_fusion_type, 
              dropout=args.dropout)
    checkpoint = torch.load(weights)
    print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))
    base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
    net.load_state_dict(base_dict)
    '''
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
    if style == 'RGB':
        flow_prefix = 'img'
    else:
        flow_prefix = 'flow_{}' 
    data_loader = torch.utils.data.DataLoader(
           TSNDataSet(fifty_data,
                      modality=style,
                      image_tmpl= flow_prefix + "_{:05d}.jpg", num_segments=25, 
                      new_length=1 if style == 'RGB' else 5, 
                      transform=torchvision.transforms.Compose([
                          cropping,
                          Stack(roll=args.arch == 'BNInception'),
                          ToTorchFormatTensor(div=args.arch != 'BNInception'),
                          GroupNormalize(net.input_mean, net.input_std),
                      ]),test_mode=True),
               batch_size=1, shuffle=False,
               num_workers=args.workers * 2, pin_memory=True)

    if args.gpus is not None:
        devices = [args.gpus[i] for i in range(args.workers)]
    else:
        devices = list(range(args.workers))

    net = torch.nn.DataParallel(net.cuda(devices[0]), device_ids=devices)
    net.eval()

    max_num = args.max_num if args.max_num > 0 else len(data_loader.dataset)
    data_gen = enumerate(data_loader)
    for i, (data) in data_gen:
        if i >= max_num:
            break
        if style == 'RGB':  
            rst = eval_video(data, 3, net, style)
        else:
            rst = eval_video(data, 10, net, style)
        video_pred = np.argmax(np.mean(rst[0], axis=0))
    #return make_hmdb()[video_pred]
    return rst 


if __name__=="__main__":
    parser = argparse.ArgumentParser( 
        description = "Standard video-level testing")
    parser.add_argument('dataset', type=str, choices=['ucf101', 'hmdb51'])
    parser.add_argument('rgb_weights', type=str)
    parser.add_argument('of_weights', type=str)
    parser.add_argument('--arch', type=str, default="BNInception")
    parser.add_argument('--max_num', type=int, default=-1)
    parser.add_argument('--test_crops', type=int, default=10)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--crop_fusion_type', type=str, default='avg', choices=['avg', 'max', 'topk'])
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.7)
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
			help='number of data loading workers (default: 2)')
    parser.add_argument('--gpus', nargs='+', type=int, default=None)
    parser.add_argument('--flow_prefix', type=str, default='')

    args = parser.parse_args()

    if args.dataset == 'ucf101':
        num_class = 101
        index_dict = make_ucf() 
    elif args.dataset == 'hmdb51':
        num_class = 51
        index_dict = make_hmdb()
    else:
        raise ValueError('Unknown dataset '+args.dataset)
    #vid_dir = '/cmsdata/hdd2/cmslab/haabibi/HMBD51/fall_floor/APOCALYPTO_fall_floor_f_nm_np1_ba_med_5.avi'
    vid_dir = '/cmsdata/hdd2/cmslab/haabibi/HMBD51/pullup/100_pullups_pullup_f_nm_np1_fr_med_1.avi'

    #######LOADING RGB_NET#######
    rgb_net = TSN(num_class, 1, 'RGB',
                  base_model=args.arch,
                  consensus_type=args.crop_fusion_type,
                  dropout=args.dropout)
    rgb_checkpoint = torch.load(args.rgb_weights)
    print("model epoch {} best prec@1: {}".format(rgb_checkpoint['epoch'], rgb_checkpoint['best_prec1']))
    base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(rgb_checkpoint['state_dict'].items())}
    rgb_net.load_state_dict(base_dict)
    ########LOADING OF_NET#######
    of_net = TSN(num_class, 1, 'Flow',
                 base_model=args.arch,
                 consensus_type=args.crop_fusion_type,
                 dropout=args.dropout)
    of_checkpoint = torch.load(args.of_weights)
    print("model epoch {} best prec@1: {}".format(of_checkpoint['epoch'], of_checkpoint['best_prec1']))
    of_base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(of_checkpoint['state_dict'].items())}
    of_net.load_state_dict(of_base_dict)
    
    #'/cmsdata/hdd2/cmslab/haabibi/fall_detection_dataset/high_quality/Fall1_Cam1.avi'
    cap = cv2.VideoCapture(vid_dir)
    rgb_list, of_list, tmp_of_list = list(), list(), list()
    rgb_epoch, of_epoch, avg_time, epoch_avg_time, time_for_appending = 0, 0, 0, 0, 0
    start_time = time.time()
    print("starting time: ", start_time)
    while(cap.isOpened()):
        ret, frame = cap.read()
        frame_N = 40
        if ret == True:
            time_for_appending = time.time()
            rgb_list.append(frame)
            tmp_of_list.append(frame)
            print("this is rgb_length: ", len(rgb_list), "this is of_list: ", len(of_list))
            if(len(tmp_of_list) >= 2):
                of_list.append(streaming(tmp_of_list[0], tmp_of_list[1], 'tvl1'))
                tmp_of_list.pop(0)
            if(len(rgb_list) == frame_N):
                ##SCORE FUSION##
                score_fusion = (make_infer('RGB', args.rgb_weights, rgb_list, rgb_net) + make_infer('Flow', args.of_weights, of_list, of_net))/2
                video_pred = np.argmax(np.mean(score_fusion[0], axis=0))
                rgb_only = make_infer('RGB', args.rgb_weights, rgb_list, rgb_net)
                of_only = make_infer('Flow', args.of_weights, of_list, of_net)
                pred1= np.argmax(np.mean(rgb_only[0], axis=0))
                pred2= np.argmax(np.mean(of_only[0], axis=0))
                print("this is rgb", make_hmdb()[pred1])
                print("this is of", make_hmdb()[pred2])
                print(make_hmdb()[video_pred])
                rgb_list.clear()
                of_list.clear()
        
        else:
            break
    print("total duration time: ", time.time()-start_time)
