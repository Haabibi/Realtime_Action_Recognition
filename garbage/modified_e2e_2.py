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

def make_infer(style, weights, fifty_data): 
    net = TSN(num_class, 1, style, 
              base_model=args.arch, 
              consensus_type=args.crop_fusion_type, 
              dropout=args.dropout)
    checkpoint = torch.load(weights)
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
    return make_hmdb()[video_pred]

#def list_append(name_list, item, style, queue):
def list_append(name_list, incoming_queue, style, queue):
    print(threading.currentThread().getName(), "Starting! And this is how long the list is for style {}: ".format(style), len(name_list))
    item = incoming_queue.get()
    name_list.append(item)
    print("this is the len of name_list", len(name_list))
    of_list = list()
    if style == 'RGB' and len(name_list) == 50:
        print("I AM HEREEREREURE IN RGB DONENENE") 
        queue.put(name_list)
        name_list.clear()

    if style == 'tmp_flow' and len(name_list)>=2:
        print(" TEMP FLOW DONEOINEW???")
        extract_of = streaming(name_list[0], name_list[1], 'tvl1')
        of_list.append(extract_of)
        name_list.pop()
        if len(of_list) == 50:
            print("got 50 items in of_list")
            queue.put(of_list)
            of_list.clear()

def receive_and_run(style, weights, queue):
    what_to_run = queue.get() 
    output = make_infer(style, weights, what_to_run)
    print(output)

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
    vid_dir = '/cmsdata/hdd2/cmslab/haabibi/fall_detection_dataset/high_quality/Fall1_Cam1.avi'
    cap = cv2.VideoCapture(vid_dir)
    rgb_list,  tmp_of_list =  list(), list()
    rgb_epoch, of_epoch, avg_time, epoch_avg_time = 0, 0, 0, 0
    read_frame_queue_rgb, read_frame_queue_of, rgb_queue, of_queue = Queue(), Queue(), Queue(), Queue() 
   
    t1 = Thread(name = 'append frames to rgb_list', target = list_append, args=(rgb_list, read_frame_queue_rgb, 'RGB', rgb_queue))
    t2 = Thread(name = 'append frames to tmp_of_list', target = list_append, args = (tmp_of_list, read_frame_queue_of, 'tmp_flow', of_queue)) 
    t3 = Thread(name = 'run rgb inference', target = receive_and_run, args=('RGB', args.rgb_weights, rgb_queue))
    t4 = Thread(name = 'run of inference', target = receive_and_run, args=('Flow', args.of_weights, of_queue))
    t1.start()

    while(cap.isOpened()):
        ret, frame = cap.read()
        local_thread = []
        if ret == True:
            read_frame_queue_rgb.put(frame)
            read_frame_queue_of.put(frame)
            print("queue_size", read_frame_queue_rgb.qsize(), read_frame_queue_of.qsize())
#            t2.start()
#        if rgb_queue.qsize() > 0:
#            print("starting to start threading number 3") 
#            t3.start()
#        if of_queue.qsize() > 0:
#            print("starting to start threading number 4")
 #           t4.start() 
        else:
            break




'''
            list_append(rgb_list, frame, 'RGB')
            list_append(tmp_of_list, frame, 'Flow')
            print("this is rgb_length: ", len(rgb_list), "this is of_list: ", len(of_list))
            
            if(len(rgb_list) == 10):
                rgb_epoch += 1 
                print("result of inference of rgb of epoch {}: ".format(rgb_epoch), make_infer('RGB', args.rgb_weights, rgb_list))
                rgb_list.clear()
            if(len(tmp_of_list) >= 2):
                tic = time.time()
                of_list.append(streaming(tmp_of_list[0], tmp_of_list[1], 'tvl1')) 
                toc = time.time()
                avg_time += toc - tic
                # extected streaming output: [ <PIL IMG>, <PIL IMG> ] 
                tmp_of_list.pop(0) 
                if (len(of_list) == 8): 
                    of_epoch += 1 
                    print("result of inference of o_f of epoch {}: ".format(of_epoch), make_infer('Flow', args.of_weights, of_list))
                    epoch_avg_time = avg_time / 10
                    print("avg time for extracting optical flow", epoch_avg_time)
                    avg_time = 0
                    of_list.clear()
        else:
            print("total avg_time for extracting optical flow??: ", epoch_avg_time / of_epoch)
            break
'''
