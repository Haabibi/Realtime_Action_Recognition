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
from streaming import streaming 
from threading import Thread, Event 
from queue import Queue
def setting():
    import os 
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    cuda.init()
    #######CHECK GPU STATUS#######
    print("NUM OF DEVICES: ", cuda.Device.count())
    gpu_list = [ i for i in range(cuda.Device.count())]
    print("THIS IS GPU LIST: ", gpu_list)

    #######LOADING RGB_NET#######
    torch.cuda.nvtx.range_push('RGB NET')
    before = time.time()
    rgb_net = TSN(num_class, 1, 'RGB',
              base_model=args.arch,
              consensus_type=args.crop_fusion_type,
              dropout=args.dropout)
    rgb_checkpoint = torch.load(args.rgb_weights)
    print("model epoch {} best prec@1: {}".format(rgb_checkpoint['epoch'], rgb_checkpoint['best_prec1']))
    base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(rgb_checkpoint['state_dict'].items())}
    rgb_net.load_state_dict(base_dict)
    torch.cuda.set_device(0)
    rgb_net = rgb_net.cuda()
    print("[rgb_net_cuda]: ", next(rgb_net.parameters()).is_cuda, type(rgb_net))  # RETURNS TRUE
    after = time.time() 
    print("loading rgb_net: ", after-before)
    torch.cuda.nvtx.range_pop()

    #######LOADING OF_NET#######
    torch.cuda.nvtx.range_push('OF NET')
    before = time.time()
    of_net = TSN(num_class, 1, 'Flow',
              base_model=args.arch,
              consensus_type=args.crop_fusion_type,
              dropout=args.dropout)
    of_checkpoint = torch.load(args.of_weights)
    print("model epoch {} best prec@1: {}".format(of_checkpoint['epoch'], of_checkpoint['best_prec1']))
    base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(of_checkpoint['state_dict'].items())}
    of_net.load_state_dict(base_dict)
    #torch.cuda.set_device(1) if len(gpu_list) != 0 else torch.cuda.set_device(0)
    of_net = of_net.cuda()
    print("[of_net_cuda]: ", next(rgb_net.parameters()).is_cuda, type(of_net))  # RETURNS TRUE
    after = time.time() 
    print("loading of_net: ", after-before)
    torch.cuda.nvtx.range_pop() 
    
    return rgb_net, of_net

def make_ucf():
    index_dir = '/cmsdata/hdd2/cmslab/haabibi/UCF101CLASSIND.txt'
    index_dict = {}
    with open(index_dir) as f:
        for line in f.readlines():
            s = line.split(' ')
            index_dict.update({int(s[0])-1: s[1]})
    return index_dict 

def make_hmdb():
    index_dir = './HMDB51CLASSIND.txt'
    index_dict = {}
    with open(index_dir) as f:
        for line in f.readlines():
            s = line.split(' ')
            index_dict.update({int(s[0]): s[1]})
    return index_dict

def eval_video(data, length, net, style):
    data = data.cuda() 
    input_var = torch.autograd.Variable(data.view(-1, length , data.size(1), data.size(2)), volatile=True)
    rst = net(input_var)
    rst_data = rst.data.cpu().numpy().copy()
    output = rst_data.reshape((-1 , args.test_segments, num_class)).mean(axis=0).reshape((args.test_segments, 1, num_class))
    return output

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
    sample_img_time_1 = time.time()
    if style == 'RGB':
        for seg_ind in offsets:
          im = Image.fromarray(data[seg_ind], mode='RGB')
          im = im.resize((224, 224))
          list_imgs.append(im) 
    
    if style == 'Flow':
        for seg_ind in offsets:
            for i in range(5):
                time_streaming_1 = time.time()
                x_img, y_img = streaming(data[seg_ind+i], data[seg_ind+(i+1)])
                time_streaming_2 = time.time()
                time_streaming_of.append(time_streaming_2 - time_streaming_1)
                x_img = Image.fromarray(x_img)
                y_img = Image.fromarray(y_img)
                x_img = x_img.resize((224, 224))
                y_img = y_img.resize((224, 224))
                list_imgs.extend([x_img.convert('L'), y_img.convert('L')]) 
    sample_img_time_2 = time.time() 
    process_data = transform(list_imgs) 
    preprocess_img_time_3 = time.time()
    if style == 'RGB':
        time_img_transformation_rgb.append(preprocess_img_time_3 - sample_img_time_2)
        time_preprocessing_rgb.append(sample_img_time_2 - sample_img_time_1)
    else:
        time_img_transformation_of.append(preprocess_img_time_3 - sample_img_time_2)
        time_preprocessing_of.append(sample_img_time_2 - sample_img_time_1)
    #print("[{}] Process Image: {}, Image Transformation: {}".format(style, sample_img_time_2-sample_img_time_1, preprocess_img_time_3-sample_img_time_2)) 
    return process_data

def make_infer(batched_array, net, style): 
    net.float() 
    net.eval() 
    data = _get_item(batched_array, net, style) 
    eval_vid_time_1 = time.time()
    rst = eval_video(data, 3 if style =="RGB" else 10, net, style) 
    eval_vid_time_2 = time.time()
    if style == 'RGB':
        time_run_nn_rgb.append(eval_vid_time_2 - eval_vid_time_1)
    else: 
        time_run_nn_of.append(eval_vid_time_2 - eval_vid_time_1)
    #print("[{}] Run NN: {}".format(style, eval_vid_time_2-eval_vid_time_1))
    
    return rst 

def run_rgb_queue(rgb_queue, rgb_net, score_queue, in_progress):
    counter = 0 
    while True:
        rgb_score = make_infer(rgb_queue.get(), rgb_net, 'RGB')  
        score_queue.put(('RGB', rgb_score)) 
        counter += 1
        if counter == args.num_repeat: 
            break

def run_of_queue(of_queue, of_net, score_queue, in_progress):
    counter = 0
    while True:
        of_score = make_infer(of_queue.get(), of_net, 'Flow')  
        rgb_score = score_queue.get()[1]
        score_fusion_time_1 = time.time()
        avg = (of_score + rgb_score) / 2
        video_pred = np.argmax(np.mean(avg[0], axis=0))
        final_result=make_ucf()[video_pred]
        score_fusion_time_2 = time.time()
        print("RESULT: ", final_result)
        time_fuse_score.append(score_fusion_time_2 - score_fusion_time_1)
        counter += 1
        in_progress.set()
        if counter == args.num_repeat:
            break

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
    parser.add_argument('--gpus', type=str, default=None)
    parser.add_argument('--flow_prefix', type=str, default='')
    parser.add_argument('--sliding_window', type=int, default=40)
    parser.add_argument('--num_repeat', type=int, default=13)
    parser.add_argument('--vid_dir', type=str, default='/cmsdata/hdd2/cmslab/haabibi/UCF101/v_Archery_g10_c05.avi')
    parser.add_argument('--interval', type=int, default = 40)
    args = parser.parse_args()

    if args.dataset == 'ucf101':
        num_class = 101
        index_dict = make_ucf() 
    elif args.dataset == 'hmdb51':
        num_class = 51
        index_dict = make_hmdb()
    else:
        raise ValueError('Unknown dataset '+args.dataset)
    
    rgb_net, of_net = setting()

    #######TIME LIST ########
    time_streaming = []
    time_preprocessing_rgb = []
    time_preprocessing_of = []
    time_img_transformation_rgb = []
    time_img_transformation_of = []
    time_run_nn_rgb = []
    time_run_nn_of = [] 
    time_fuse_score = []
    time_streaming_of = []

    #######MAKING QUEUE#######
    rgb_queue = Queue()
    of_queue = Queue()
    score_queue = Queue()
    in_progress = Event()
    
    #######MAKING THREADS#######
    jobs = [ Thread(target=run_rgb_queue, args=(rgb_queue, rgb_net, score_queue, in_progress)), 
             Thread(target=run_of_queue, args=(of_queue, of_net, score_queue, in_progress))]

    [ job.start() for job in jobs ] 
    in_progress.set() 
    counter = 0 
    for i in range(args.num_repeat):
        print("[Iteration {}]".format(i))
        while in_progress.wait():
            in_progress.clear()
            video_streaming_1 = time.time()
            cap = cv2.VideoCapture(args.vid_dir)
            frame_list = list()
            while(cap.isOpened()):
                ret, frame = cap.read()
                accumulated_time_for_rgb = 0 
                accumulated_time_for_of = 0 
                accumulated_RGB, accumulated_OF = 0, 0
                if ret == True:
                    frame_list.append(frame)
                else:
                    video_streaming_2 = time.time()
                    time_streaming.append(video_streaming_2 - video_streaming_1)
                    of_queue.put(frame_list)
                    rgb_queue.put(frame_list)
                    counter += 1
                    break
            break
        if counter == args.num_repeat:
            break

    [ job.join() for job in jobs ]
    
    time_list = [ time_streaming,
                  time_preprocessing_rgb,
                  time_preprocessing_of,
                  time_img_transformation_rgb,
                  time_img_transformation_of,
                  time_run_nn_rgb,
                  time_run_nn_of, 
                  time_fuse_score,
                  time_streaming_of]

    for item in time_list:
        zero = 0 
        for time in item:
            zero += time 
        print(item)
        print(zero/len(item))
