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
    input_var = torch.autograd.Variable(data.view(-1, length, data.size(2), data.size(3)), volatile=True)
    cuda_tic = time.time()
    input_var = input_var.cuda()
    cuda_toc = time.time()
    rst = net(input_var)
    time_run_net = time.time()
    rst_data = rst.data
    rst_data_cpu = rst_data.cpu()
    rst_data_np = rst_data_cpu.numpy().copy()
    time_cpu_copy_time = time.time() 
    print("[eval_video_time]: [cuda]: {}, [run_net]: {}, [cpu_cpy_time]: {}".format(cuda_toc-cuda_tic, time_run_net-cuda_toc, time_cpu_copy_time-time_run_net))
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
               num_workers=args.workers * 2,
               pin_memory=True)
    #answer = make_hmdb() if args.dataset == 'hmdb51' else make_ucf()
    net.float() 
    net.eval() 
    net_cuda_tic = time.time()
    net = net.cuda() 
    net_cuda_toc = time.time() 
    print("[net_cuda_time]: ", net_cuda_toc-net_cuda_tic)
    
    max_num = args.max_num if args.max_num > 0 else len(data_loader.dataset)
    enumrate_time = time.time() 
    data_gen = enumerate(data_loader)
    video_pred_tic = time.time() 
    print("[time for enumerating data_loader]: ", video_pred_tic-enumrate_time, "[max_num]: ", max_num)
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
    video_pred_toc = time.time() 
    print("[time actually running for loop]: ", video_pred_toc-video_pred_tic)
    return rst
    '''
    data_load_tic = time.time()
    iter_data_load = iter(data_loader)
    data_load_tic_2 = time.time()
    data = next(iter_data_load)
    #data = next(iter(data_loader))
    data_load_toc = time.time()  
    print("[data_iter time], ",data_load_tic_2 - data_load_tic, "[net_daata], ", data_load_toc-data_load_tic_2 )
    eval_vid_tic = time.time()
    rst = eval_video(data, 3 if style =="RGB" else 5, net, style) 
    eval_vid_toc = time.time()
    print("[rst, eval_vid]: ", eval_vid_toc-eval_vid_tic)
    return rst 
    '''

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
            frame = frame.reshape(1, 240, 320, 3)
            counter += 1 
            if counter == 1:
                 first_time_rgb = time.time() 
                 inp = np.zeros((1, 240, 320, 3))
                 inp += frame
                 print("[c1] inp.shape: ", inp.shape)
            elif counter < args.q_size: 
                 inp = np.concatenate((inp, frame))
                 print("[c{}] inp.shape: ".format(counter), inp.shape)
            elif counter == args.q_size:
                 ready_for_inf = time.time()
                 print("[Time when frames are full]: ", ready_for_inf-first_time_rgb)
                 inp = np.concatenate((inp, frame))
                 #inp = inp.reshape((args.q_size, 240, 320, 3))
                 for i in range(args.num_repeat+1):
                     if i == 0: 
                         cold_case_tic = time.time()
                         rgb_inf_score = make_infer('RGB', args.rgb_weights, inp, rgb_net)
                         cold_case_toc = time.time()
                         print("[cold case time]: ", cold_case_toc-cold_case_tic)
                         video_pred = np.argmax(np.mean(rgb_inf_score[0], axis=0))
                         print("THIS IS THE RESULT: ", make_hmdb()[video_pred])
                     else:                         
                         rgb_inf_tic = time.time()
                         rgb_inf_score = make_infer('RGB', args.rgb_weights, inp, rgb_net)
                         rgb_inf_toc = time.time()
                         video_pred = np.argmax(np.mean(rgb_inf_score[0], axis=0))
                         print("[rgb_inf time]: ", rgb_inf_toc-rgb_inf_tic)
                         print("THIS IS THE RESULT: ", make_hmdb()[video_pred])
                  
                 counter = 0 

        else:
            print("done reading")
            break
