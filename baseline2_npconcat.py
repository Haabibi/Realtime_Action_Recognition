import time 
import argparse
import numpy as np 
from sklearn.metrics import confusion_matrix
import cv2
from models import TSN
from transforms import * 
from ops import ConsensusModule
import pycuda.driver as cuda
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
    data = data.cuda() 
    input_var = torch.autograd.Variable(data.view( length * args.test_segments, data.size(1), data.size(2)), volatile=True)
    rst = net(input_var)
    time_run_net = time.time()
    rst_data = rst.data.cpu().numpy().copy()
    #print("[eval_video_time]: [cuda]: {}, [run_net]: {}, [cpu_cpy_time]: {}".format(cuda_toc-cuda_tic, time_run_net-cuda_toc, time_cpu_copy_time-time_run_net))

    return rst_data.reshape((1, args.test_segments, num_class)).mean(axis=0).reshape((args.test_segments, 1, num_class))

def _get_indices(data, style):
    new_length = 1 if style == 'RGB' else 5 
    tick =  (len(data) - new_length + 1) / float(args.test_segments)
    offsets = np.array([int(tick / 2.0 + tick * x) for x in range(args.test_segments)])
    return offsets

def _get_item(data, net, style):
    list_imgs = []
    offsets = _get_indices(data, style)
    transform = torchvision.transforms.Compose([ Stack(roll=args.arch == 'BNInception'),
                                                 ToTorchFormatTensor(div=args.arch != 'BNInception'),
                                                 GroupNormalize(net.input_mean, net.input_std),
                                               ])

    for seg_ind in offsets:
        seg_img = data[seg_ind]
        im = Image.fromarray(seg_img, mode='RGB' if style == 'RGB' else 'L') 
        im = im.resize((224, 224)) 
        list_imgs.append(im) 
    process_data = transform(list_imgs) 
    print("[type of process_data]: ", type(process_data), process_data.shape)
    return process_data

def make_infer(weights, batched_array, net, style): 
    if style == 'RGB':
        flow_prefix = 'img'
    else:
        flow_prefix = 'flow_{}'
    net.float() 
    net.eval() 
    net_cuda_tic = time.time()
    net = net.cuda() 
    net_cuda_toc = time.time() 
    print("[net_cuda_time]: ", net_cuda_toc-net_cuda_tic, next(net.parameters()).is_cuda)
    eval_vid_tic = time.time()
    time_data_tic = time.time()
    data = _get_item(batched_array, net, style) 
    time_data_toc = time.time()
    rst = eval_video(data, 3 if style =="RGB" else 5, net, style) 
    eval_vid_toc = time.time()
    print("[getitem]: ", time_data_toc-time_data_tic, "[rst, eval_vid]: ", eval_vid_toc-eval_vid_tic)
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
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
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
                 for i in range(args.num_repeat+1):
                     if i == 0: 
                         cold_case_tic = time.time()
                         rgb_inf_score = make_infer(args.rgb_weights, inp, rgb_net, 'RGB')
                         cold_case_toc = time.time()
                         print("[cold case time]: ", cold_case_toc-cold_case_tic)
                         video_pred = np.argmax(np.mean(rgb_inf_score[0], axis=0))
                         print("THIS IS THE RESULT: ", make_hmdb()[video_pred])
                     else:                         
                         rgb_inf_tic = time.time()
                         rgb_inf_score = make_infer(args.rgb_weights, inp, rgb_net, 'RGB')
                         rgb_inf_toc = time.time()
                         video_pred = np.argmax(np.mean(rgb_inf_score[0], axis=0))
                         print("[rgb_inf time]: ", rgb_inf_toc-rgb_inf_tic)
                         print("THIS IS THE RESULT: ", make_hmdb()[video_pred])
                  
                 counter = 0 

        else:
            print("done reading")
            break
