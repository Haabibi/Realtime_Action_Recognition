import time 
import argparse
import numpy as np 
import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix
import cv2
from dataset_memory_2 import TSNDataSet 
from models import TSN
from transforms import * 
from ops import ConsensusModule
from streaming4 import streaming
import pycuda.driver as cuda
import pycuda.gpuarray 
from PIL import Image

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
    print("[these are the parameters of eval_vid]: ", length, data.size(2), data.size(3), data.shape)
    input_var = torch.autograd.Variable(data.view(-1, length, data.size(2), data.size(3)), volatile=True)
    input_var = input_var.cuda()
    print("[eval_vid]: ", type(input_var), input_var.shape) 
    rst = net(input_var)
     
    rst_data = rst.data
    rst_data_cpu = rst_data.cpu()
    rst_data_np = rst_data_cpu.numpy().copy()
    print("[rst_data_np]: ", type(rst_data_np), rst_data_np.shape)
    
    return rst_data_np.reshape((args.test_crops, args.test_segments, num_class)).mean(axis=0).reshape((args.test_segments, 1, num_class))

def make_infer(style, weights, batched_array, net): 
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
           TSNDataSet(batched_array,
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
               num_workers=0,#args.workers * 2,
               pin_memory=True)
    answer = make_hmdb() if args.dataset == 'hmdb51' else make_ucf()
    data_toc = time.time()
    net.float() 
    net.eval() 
    net = net.cuda() 
    video_pred_tic = time.time()
    data = next(iter(data_loader))
    print("[type of data]: ", type(data))
    eval_vid_tic = time.time()
    rst = eval_video(data, 3 if style =="RGB" else 5, net, style) 
    eval_vid_toc = time.time()
    video_pred = np.argmax(np.mean(rst[0], axis=0))
    print("THIS IS THE RESULT: ", answer[video_pred])
    video_pred_toc = time.time() 
    if style == 'RGB':
        print("evaluating rgb(NET EVAL) in {:.4f}, {} data_loading in {:.4f}, video_pred in {:.4f}, net_eval and parallelism {:.4f}".format(eval_vid_toc-eval_vid_tic, style, data_toc-data_tic, video_pred_toc-video_pred_tic, video_pred_tic-data_toc))
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
    rgb_net = rgb_net.cuda()
    print("[rgb_net_cuda]: ", next(rgb_net.parameters()).is_cuda, type(rgb_net))  # RETURNS TRUE
    after = time.time() 
    print("loading rgb_net: ", after-before)
    cap = cv2.VideoCapture(vid_dir)
    counter = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            counter += 1 
            if counter == 1:
                 inp = np.zeros((240, 320, 3))
                 inp += frame
                 print("[c1] inp.shape: ", inp.shape)
            elif counter < args.q_size: 
                 inp = np.concatenate((inp, frame))
                 print("[c{}] inp.shape: ".format(counter), inp.shape)
            elif counter == args.q_size:
                 inp = np.concatenate((inp, frame))
                 inp = inp.reshape((args.q_size, 240, 320, 3))
                 print("[c{}] inp.shape: ".format(counter), inp.shape)
                 make_infer('RGB', args.rgb_weights, inp, rgb_net)
                 counter = 0 

        else:
            print("done reading")
            break
