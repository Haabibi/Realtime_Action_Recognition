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
    print("[FROM EVAL_VID]: ", type(data))
    data = data.cuda() 
    input_var = torch.autograd.Variable(data.view(-1, length, data.size(1), data.size(2)), volatile=True)
    rst = net(input_var)
    time_run_net = time.time()
    rst_data = rst.data.cpu().numpy().copy()
    #print("[eval_video_time]: [cuda]: {}, [run_net]: {}, [cpu_cpy_time]: {}".format(cuda_toc-cuda_tic, time_run_net-cuda_toc, time_cpu_copy_time-time_run_net))

    return rst_data.reshape((10, args.test_segments, num_class)).mean(axis=0).reshape((args.test_segments, 1, num_class))

def _get_indices(data, style):
    new_length = 1 if style == 'RGB' else 5 
    tick =  (len(data) - new_length + 1) / float(args.test_segments)
    offsets = np.array([int(tick / 2.0 + tick * x) for x in range(args.test_segments)])
    return offsets

def _get_item(data, net, style):
    list_imgs = []
    offsets = _get_indices(data, style) 
    cropping = torchvision.transforms.Compose([
        GroupOverSample(net.input_size, net.scale_size)
    ]) 

    transform = torchvision.transforms.Compose([ 
        cropping,
        Stack(roll=args.arch == 'BNInception'),
        ToTorchFormatTensor(div=args.arch != 'BNInception'),
        GroupNormalize(net.input_mean, net.input_std)
    ])

    for seg_ind in offsets:
        seg_img = data[seg_ind]
        im = Image.fromarray(seg_img, mode='RGB' if style == 'RGB' else 'L') 
        im = im.resize((224, 224)) 
        list_imgs.append(im) 
    process_data = transform(list_imgs) 
    #return list_imgs
    return process_data

def make_infer(weights, batched_array, net, style): 
    if style == 'RGB':
        flow_prefix = 'img'
    else:
        flow_prefix = 'flow_{}'
    net.float() 
    net.eval() 
    net = net.cuda() 
    eval_vid_tic = time.time()
    data = _get_item(batched_array, net, style) 
    time_data_toc = time.time()
    rst = eval_video(data, 3 if style =="RGB" else 5, net, style) 
    eval_vid_toc = time.time()
    print("[getitem]: ", time_data_toc-eval_vid_tic, "[rst, eval_vid]: ", eval_vid_toc-eval_vid_tic)
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
    parser.add_argument('--vid_dir', type=str) 
    parser.add_argument('--sliding_window', type=int, default=40)
    args = parser.parse_args()

    if args.dataset == 'ucf101':
        num_class = 101
        index_dict = make_ucf() 
    elif args.dataset == 'hmdb51':
        num_class = 51
        index_dict = make_hmdb()
    else:
        raise ValueError('Unknown dataset '+args.dataset)
    #vid_dir = '/cmsdata/hdd2/cmslab/haabibi/HMBD51/pullup/100_pullups_pullup_f_nm_np1_fr_med_1.avi'

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
    
    cap = cv2.VideoCapture(args.vid_dir)
    rgb_list, _tmp_of, of_list = list(), list(), list()
    counter = 0
    accumulated_time = 0
    loading_frames = 0
    output_rst = {} 
    before = time.time()
    while(cap.isOpened()):
        ret, frame = cap.read()
        accumulated_time_for_inf = 0 
        if ret == True:
            rgb_list.append(frame)
            print("[{}] length of rgb_list: ".format(counter), len(rgb_list))
            if len(rgb_list) == 1:
                first_time_rgb = time.time() 
            if len(rgb_list) == args.q_size :
                loading_frames += (time.time() - first_time_rgb)
                counter += 1
                got_here_rgb = time.time() 
                print("[[{}] time when all frames are ready to be run]: ".format(counter), got_here_rgb-first_time_rgb)
                for i in range(args.num_repeat+1):
                    if i == 0:
                        cold_case_tic = time.time()
                        make_infer(args.rgb_weights, rgb_list, rgb_net, 'RGB')
                        cold_case_toc = time.time()
                        print("cold case inf time: ", cold_case_toc-cold_case_tic)
                    else: 
                        rgb_inf_tic = time.time() 
                        rgb_inference = make_infer(args.rgb_weights, rgb_list, rgb_net, 'RGB')
                        rgb_inf_toc = time.time() 
                        video_pred = np.argmax(np.mean(rgb_inference[0], axis=0))
                        arg_max_time = time.time() 
                        output_rst[counter] = make_hmdb()[video_pred]
                        print(make_hmdb()[video_pred])
                        answer_time = time.time() 
                        accumulated_time_for_inf += (rgb_inf_toc-rgb_inf_tic)
                        print("inference rgb in {:.5f}, video pred arg_max in {:.5f}, getting the result in {:.5f} ".format(rgb_inf_toc-rgb_inf_tic, arg_max_time-rgb_inf_toc, answer_time-arg_max_time))
                print("accumulated time for rgb: {:.5f}".format(accumulated_time_for_inf/(args.num_repeat)))
                accumulated_time += accumulated_time_for_inf/args.num_repeat

                accumulated_time_for_inf = 0 
                rgb_list = rgb_list[args.sliding_window:]
                print("how much there is left for rgb_list: ", len(rgb_list)) 
            
        else:
            print("[TOTAL END_TO_END LATENCY]: ", time.time()-before)
            print("[LOADING FRAMES AVG]: ", loading_frames/counter)
            print("[AVG INF TIME]: ", accumulated_time/counter)
            print("[OUTPUT RESULT]: ", output_rst)
            break
