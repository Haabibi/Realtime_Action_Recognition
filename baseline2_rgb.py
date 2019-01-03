import time 
import argparse
import numpy as np 
import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix
import cv2
from dataset_memory import TSNDataSet 
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
    return rst.reshape((args.test_crops, args.test_segments, num_class)).mean(axis=0).reshape((args.test_segments, 1, num_class))

def make_infer(style, weights, fifty_data, net): 
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
    data_tic = time.time() 
    data_loader = torch.utils.data.DataLoader(
           TSNDataSet(fifty_data,
                      modality=style,
                      image_tmpl= flow_prefix + "_{:05d}.jpg", num_segments=args.test_segments, 
                      new_length=1 if style == 'RGB' else 5, 
                      transform=torchvision.transforms.Compose([
                          cropping,
                          Stack(roll=args.arch == 'BNInception'),
                          ToTorchFormatTensor(div=args.arch != 'BNInception'),
                          GroupNormalize(net.input_mean, net.input_std),
                      ]),test_mode=True),
               batch_size=args.q_size, shuffle=False,
               num_workers=args.workers * 2, pin_memory=True)
    data_toc = time.time() 
    if args.gpus is not None:
        devices = [args.gpus[i] for i in range(args.workers)]
    else:
        devices = list(range(args.workers))
    net_tic = time.time()
    net = torch.nn.DataParallel(net.cuda(devices[0]), device_ids=devices) #only net  ?? ?
    net.eval()
    net_toc = time.time()
    
    max_num = args.max_num if args.max_num > 0 else len(data_loader.dataset)
    data_gen = enumerate(data_loader)
    video_pred_tic = time.time() 
    for i, (data) in data_gen:
        if i >= max_num:
            break
        if style == 'RGB':  
            rgb_eval_vid_tic = time.time()
            rst = eval_video(data, 3, net, style)
            rgb_eval_vid_toc = time.time()
        else:
            of_eval_vid_tic = time.time()
            rst = eval_video(data, 10, net, style)
            of_eval_vid_toc = time.time()
        video_pred = np.argmax(np.mean(rst[0], axis=0))
    video_pred_toc = time.time() 
    if style == 'RGB':
        print("evaluating rgb(NET EVAL) in {:.4f}, {} data_loading in {:.4f}, video_pred in {:.4f}, net_eval and parallelism {:.4f}".format(rgb_eval_vid_toc-rgb_eval_vid_tic, style, data_toc-data_tic, video_pred_toc-video_pred_tic, net_toc-net_tic))
    return rst 


if __name__=="__main__":
    parser = argparse.ArgumentParser( 
        description = "Standard video-level testing")
    parser.add_argument('dataset', type=str, choices=['ucf101', 'hmdb51'])
    parser.add_argument('rgb_weights', type=str)
    parser.add_argument('of_weights', type=str)
    parser.add_argument('--arch', type=str, default="BNInception")
    parser.add_argument('--test_segments', type=int, default=25)
    parser.add_argument('--max_num', type=int, default=-1)
    parser.add_argument('--test_crops', type=int, default=10)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--crop_fusion_type', type=str, default='avg', choices=['avg', 'max', 'topk'])
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.7)
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
			help='number of data loading workers (default: 2)')
    parser.add_argument('--gpus', nargs='+', type=int, default=None)
    parser.add_argument('--flow_prefix', type=str, default='')
    parser.add_argument('--q_size', type=int, default=40)
    parser.add_argument('--num_repeat', type=int, default=10)

    args = parser.parse_args()

    if args.dataset == 'ucf101':
        num_class = 101
        index_dict = make_ucf() 
    elif args.dataset == 'hmdb51':
        num_class = 51
        index_dict = make_hmdb()
    else:
        raise ValueError('Unknown dataset '+args.dataset)
    vid_dir = '/cmsdata/hdd2/cmslab/haabibi/HMBD51/pullup/100_pullups_pullup_f_nm_np1_fr_med_1.avi'

    #######LOADING RGB_NET#######
    before = time.time()
    rgb_net = TSN(num_class, 1, 'RGB',
                  base_model=args.arch,
                  consensus_type=args.crop_fusion_type,
                  dropout=args.dropout)
    rgb_checkpoint = torch.load(args.rgb_weights)
    print("model epoch {} best prec@1: {}".format(rgb_checkpoint['epoch'], rgb_checkpoint['best_prec1']))
    base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(rgb_checkpoint['state_dict'].items())}
    rgb_net.load_state_dict(base_dict)
    after = time.time() 
    print("loading rgb_net: ", after-before)
    
    cap = cv2.VideoCapture(vid_dir)
    rgb_list = list()
    accumulated_time_for_inf = 0 
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            rgb_list.append(frame)
            if len(rgb_list) == 1:
                first_time_rgb = time.time() 
            #print("this is rgb_length: ", len(rgb_list))
            if len(rgb_list) == args.q_size :
                ##SCORE FUSION##
                got_here_rgb = time.time() 
                ##INFERENCE RGB## 
                for i in range(args.num_repeat+1):
                    if i == 0:
                        cold_case_tic = time.time()
                        make_infer('RGB', args.rgb_weights, rgb_list, rgb_net)
                        cold_case_toc = time.time()
                        print("cold case inf time: ", cold_case_toc-cold_case_tic)
                    else: 
                        rgb_inf_tic = time.time() 
                        rgb_inference = make_infer('RGB', args.rgb_weights, rgb_list, rgb_net)
                        rgb_inf_toc = time.time() 
                        video_pred = np.argmax(np.mean(rgb_inference[0], axis=0))
                        print(make_hmdb()[video_pred])
                        accumulated_time_for_inf += (rgb_inf_toc-rgb_inf_tic)
                        print("inference rgb in {:.4f}".format(rgb_inf_toc-rgb_inf_tic))
                print("accumulated time for rgb: {:.4f}".format(accumulated_time_for_inf/(args.num_repeat)))
                accumulated_time_for_inf = 0 
                rgb_list.clear()

        else:
            print("done reading")
            break
