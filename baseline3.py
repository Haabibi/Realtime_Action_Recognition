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
import threading
from threading import *
from queue import Queue 

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

def make_infer(style, fifty_data, net): 
    print(" and this is the length of fifty data list (supposed to be fifty?", len(fifty_data))
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

    data_toc1= time.time()
    
    if args.gpus is not None:
        devices = [args.gpus[i] for i in range(args.workers)]
    else:
        devices = list(range(args.workers))
    
    if style == 'Flow' or style =='RGB':
        net = torch.nn.DataParallel(net.cuda(devices[0]), device_ids=devices)
        net.eval()
    max_num = args.max_num if args.max_num > 0 else len(data_loader.dataset)
   
   #data_gen = enumerate(data_loader)
    for (data) in data_loader:
        #if i >= max_num:
        #    break
        if style == 'RGB':  
            rst = eval_video(data, 3, net, style)
        else:
            rst = eval_video(data, 10, net, style)
        video_pred = np.argmax(np.mean(rst[0], axis=0))
    data_toc2 = time.time() 
    print("how long it took to infer a video of style: {} in {}seconds. Loading DATA took {} sec.".format(style, data_toc2-data_toc1, data_toc1-data_tic))
    print("I GOT THIS: ", make_hmdb()[video_pred])
    return make_hmdb()[video_pred]

def get_from_queue(input_queue, output_queue, style):
    tmp_list = []
    avg_extraction_time = 0 
    while True:
        tic = time.time()
        #print(input_queue.qsize(), style)
        if input_queue.qsize()> 0 and style == 'Flow':
            s_tic = time.time()
            streamed_output = streaming(input_queue.get(), input_queue.get(), 'tvl1')
            tmp_list.append(streamed_output)
            s_toc = time.time()
            avg_extraction_time += (s_toc - s_tic) 
            print("this is the length of optical_flow list from get_from_queue", len(tmp_list), input_queue.qsize())

        if input_queue.qsize() > 0 and style == 'RGB':
          #  print("I am here for rgb", input_queue.qsize()) 
            tmp_list.append(input_queue.get())
           # print("and this is how long the tmp_list before stacking 50 frames", len(tmp_list)) 
        if len(tmp_list) == 50:
            output_queue.put(tmp_list)
            toc = time.time()
            avg_extraction_time = avg_extraction_time / 50 
            print("putting fifty frames in queue for {} frames for about {:f} and extracting optical flow too {}".format(style, toc-tic, avg_extraction_time))
            tmp_list = []
            avg_extraction_time = 0 
        #if 1 < len(tmp_list) < 50 and input_queue.qsize() == 0:
        #    break
            
def get_ready_for_inf(input_queue, style, net):
    epoch = 0 
    while True:
        if input_queue.qsize() > 0:
            tic = time.time()
            epoch += 1 
            list_from_queue = input_queue.get()
            print("this is how long the input_queue is after taking one elem out", input_queue.qsize(), len(list_from_queue))
            toc1 = time.time() 
            print("{} MANY TIMES.THE RESULT OF INFERENCE OF {} COMING OUT!".format(epoch, style))
            make_infer(style, list_from_queue, net)
            toc2 = time.time()
            print("Getting the fifty_list from queue took {}. Running inference took {} for style {}".format(toc1-tic, toc2-toc1, style))

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
    parser.add_argument('--q_size', type=int, default=40)

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
    #'/cmsdata/hdd2/cmslab/haabibi/fall_detection_dataset/high_quality/Fall1_Cam1.avi'
    cap = cv2.VideoCapture(vid_dir)
    rgb_epoch, of_epoch, avg_time, epoch_avg_time = 0, 0, 0, 0 
    frame_to_rgb_q, list_of_rgb_frames_q = Queue(maxsize=0), Queue(maxsize=0)
    frame_to_of_q, list_of_OF_frames_q = Queue(maxsize=0), Queue(maxsize=0) 
    
    
    rgb_net = TSN(num_class, 1, 'RGB', 
                  base_model=args.arch, 
                  consensus_type=args.crop_fusion_type,
                  dropout=args.dropout)
    rgb_checkpoint = torch.load(args.rgb_weights)
    rgb_base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(rgb_checkpoint['state_dict'].items())}
    rgb_net.load_state_dict(rgb_base_dict)
    
    jobs = [Thread(name='streaming rgb input', target=get_from_queue, args=(frame_to_rgb_q, list_of_rgb_frames_q, 'RGB')),
            Thread(name='getting ready for inference', target=get_ready_for_inf, args=(list_of_rgb_frames_q, 'RGB', rgb_net))]
            #Thread(name='streaming of input', target=get_from_queue, args=(frame_to_of_q, list_of_OF_frames_q, 'Flow')),    
            #Thread(name='getting ready for inference for optical flow', target=get_ready_for_inf, args=(list_of_OF_frames_q, 'Flow', of_net))]

    for job in jobs:
        job.start()
    
    if not cap.isOpened():
        print("Error Opening a video stream/ file")

    num_read_frames = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            num_read_frames += 1
            tic = time.time() 
            frame_to_rgb_q.put(frame)
            toc1 = time.time()
            print("[F{}] checking q size".format(num_read_frames), frame_to_rgb_q.qsize(), frame_to_of_q.qsize())
        else:
            print("finished reading {} frames".format(num_read_frames))
            break

    for job in jobs:
        job.join()
    print("terminating..")
