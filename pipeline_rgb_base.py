import time 
import argparse
import numpy as np 
from sklearn.metrics import confusion_matrix
import cv2
from models import TSN
from transforms import * 
import pycuda.driver as cuda
from PIL import Image
#from streaming import streaming 
import os 
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
    index_dir = './HMDB51CLASSIND.txt'
    index_dict = {}
    with open(index_dir) as f:
        for line in f.readlines():
            s = line.split(' ')
            index_dict.update({int(s[0]): s[1]})
    return index_dict

def make_infer(incoming_frames, net, style):
    net.float() 
    net.eval()
    transformed_frames = transform(incoming_frames, net, style)
    rst = eval_video(transformed_frames, 3 if style=='RGB' else 10, net, style)
    return rst

def transform(frames, net, style):
    list_imgs = []
    offsets = get_indices(frames, style)
    
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
        if style == 'RGB':
            seg_img = frames[seg_ind]
            im = Image.fromarray(seg_img, mode='RGB')
            list_imgs.extend([im]) 
        if style == 'Flow':
            for i in range(5):
                seg_img = frames[seg_ind + i] 
                x_img = Image.fromarray(seg_img[0])
                y_img = Image.fromarray(seg_img[1])
                list_imgs.extend([x_img.convert('L'), y_img.convert('L')])
    process_data = transform(list_imgs)
    return process_data

def eval_video(data, length, net, style): 
    input_var = torch.autograd.Variable(data.view(-1, length , data.size(1), data.size(2)), volatile=True)
    rst = net(input_var)
    rst_data = rst.data.cpu().numpy().copy()

    output = rst_data.reshape((-1 , args.test_segments, num_class)).mean(axis=0).reshape((args.test_segments, 1, num_class))
    return output

def get_indices(frames, style):
    new_length = 1 if style == 'RGB' else 5 
    tick =  (len(frames) - new_length +1) / float(args.test_segments)
    offsets = np.array([int(tick / 2.0 + tick * x) for x in range(args.test_segments)])
    return offsets

def get_frames_and_run(frame_queue, net, style, output_list):
    frame_list = [] 
    i = 0
    while True:
        (frame, ret, num_frames) = frame_queue.get()
        if ret == True:
            frame_list.append(frame)
            i += 1
            print("[f{}] number of frames in frame_list: {}".format(i, len(frame_list)), (frame.shape, ret))
        if ret == False:
            print(" I & LEN : ", len(frame_list), i)
            rst = make_infer(frame_list[:i], net, style)
            final_rst = np.argmax(np.mean(rst, axis=0))
            frame_list = []
            i = 0
            output_list.append((final_rst, ret))
            print("[OUTPUT LIST]: " , len(output_list), final_rst, make_ucf()[final_rst])
            #output_queue.put((final_rst, ret))
            #print("[OUTPUT QUEUE]: ", output_queue.qsize())
            
        '''
        if ret == True and len(frame_list) % 40 == 0: 
            i += 1 
            rst = make_infer(frame_list[40*(i-1):40*i], net, style)
            output_queue.put((rst, ret))
            print("[OUTPUT QUEUE]: ", output_queue.qsize())
        if ret == False:
            rst = make_infer(frame_list[40*i:], net, style)
            output_queue.put((rst, ret))
            print("SIZE OF OUTPUT_QUEUE: ", output_queue.qsize())
            frame_list = []
        '''

'''
def fuse_all_scores(output_queue, final_output_list):
    while True:
         (rst, ret) = output_queue.get()
         fuse_score_list = [] 
         seg_num = 0 
         fused_rst = 0 
         while ret == True: 
             fuse_score_list.append(rst) 
             seg_num += 1
         if ret == False: 
             fuse_score_list.append(rst)
             seg_num += 1
             for item in fuse_score_list:
                 fused_rst += item 
                 temp_fused_rst = np.argmax(np.mean(fused_rst/seg_num, axis=0))
                 final_output_list.append(temp_fused_rst)
                 print("LEN OF FINAL OUPUT LIST: " , len(final_output_list))
             seg_num = 0 
             fuse_score_list = [] 
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
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
			help='number of data loading workers (default: 2)')
    parser.add_argument('--gpus', type=str, default='5')
    parser.add_argument('--flow_prefix', type=str, default='')
    parser.add_argument('--sliding_window', type=int, default=40)
    parser.add_argument('--num_repeat', type=int, default=1)
    #parser.add_argument('--vid_dir', type=str, default=None)
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
   
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    rgb_net = TSN(num_class, 1, 'RGB',
                  base_model=args.arch,
                  consensus_type=args.crop_fusion_type,
                  dropout=args.dropout)
    rgb_checkpoint = torch.load(args.rgb_weights)
    print("model epoch {} best prec@1: {}".format(rgb_checkpoint['epoch'], rgb_checkpoint['best_prec1']))
    base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(rgb_checkpoint['state_dict'].items())}
    rgb_net.load_state_dict(base_dict)

    of_net = TSN(num_class, 1, 'Flow',
                  base_model=args.arch,
                  consensus_type=args.crop_fusion_type,
                  dropout=args.dropout)
    of_checkpoint = torch.load(args.of_weights)
    print("model epoch {} best prec@1: {}".format(of_checkpoint['epoch'], of_checkpoint['best_prec1']))
    base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(of_checkpoint['state_dict'].items())}
    of_net.load_state_dict(base_dict)
    
    output_rst, video_labels = [], [] 
    counter = 0 
    output_results = {} 
    accumulated_time = 0 
    
    video_data = open('../tsn-pytorch/ucf101_file_lists/video_101classes.txt', 'r')
    data_loader = video_data.readlines()

    frame_queue, output_queue = Queue(maxsize=0), Queue(maxsize=0)
    final_output_list = []
    
    jobs = [ Thread(name='Inference certain number of frames and output rst to Queue', target=get_frames_and_run, args=(frame_queue, rgb_net, 'RGB', final_output_list))]
            # Thread(name='Fuse scores of all segments and output the final score', target=fuse_all_scores, args=(output_queue, final_output_list))]

    for job in jobs: 
        job.start() 

    for video in data_loader:
        line_list = video.split(' ')
        video_full_link = line_list[0]
        video_labels.append(int(line_list[1].strip()))
        num_frames = 0  
        cap = cv2.VideoCapture(video_full_link)
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True: 
                num_frames += 1 
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_queue.put((frame, ret, num_frames))
                shape = frame.shape
            else:
                frame_queue.put((np.zeros(shape), ret, num_frames))
                print("Finished Reading Video {} with {} number of frames.".format(video_full_link.split('/')[-1], num_frames))
                break
        
    for job in jobs:
        job.join()
    print("Terminating.. ")

    cap.release()
    cv2.destroyAllWindows()
    cf = confusion_matrix(video_labels, final_output_list).astype(float)
    print("This is cf: ", cf)
    cls_cnt = cf.sum(axis=1)
    print("This is cls_cnt: ", cls_cnt)
    cls_hit = np.diag(cf)
    print("This is cls_hit: ", cls_hit)
    cls_acc = cls_hit / cls_cnt 
    print('Accuracy {:.02f}%'.format(np.mean(cls_acc) * 100))








