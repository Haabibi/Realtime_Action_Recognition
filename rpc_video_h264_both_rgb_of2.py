import time
from queue import Queue
from data_sender import send
import argparse
from threading import Thread, Event
import cv2
import numpy as np
from redis import Redis
import pickle
import torch
from models import TSN
from baseline_rpc_rgb2 import make_ucf, make_infer, make_hmdb, eval_video
from sklearn.metrics import confusion_matrix 
import os 
from streaming_multi import streaming 
from multiprocessing import Process, Manager
import torchvision
from transforms import * 

#### DEVIVCE ####
def streaming_device():
    while True:
        with picamera.PiCamera() as camera:
            camera.resolution = (224, 224)
            camera.framerate = 40 
            camera.start_recording('./device-video/1.h264')
            camera.wait_recording(2)
            counter = 1
            while True:
                counter += 1
                camera.split_recording('./device-video/%d.h264' % counter)
                camera.wait_recording(2)
            camera.stop_recording()

#### DEVICE ####
def data_send():
    counter = 0 
    while True:
        LIST_DIR = os.listdir('./device-video')
        SORTED_DIR = [ int(x.split('.')[0]) for x in LIST_DIR]
        SORTED_DIR.sort()
        SORTED_DIR = [ str(x) + '.h264' for x in SORTED_DIR] 
        LENGTH = len(LIST_DIR)

        if LENGTH > 1 and LENGTH > counter:
            item = SORTED_DIR[counter]
            PATH = os.path.join('/home/pi/venv/video', item)
            read_file = open(PATH, 'rb')
            encoding = read_file.read()
            if encoding != b'':
                send('Frame', encoding)
                counter += 1
#### HUB ####                
def receive_and_save(rgb_net, redis):
    counter = 1 
    frames_to_run = []
    while True:
        initial_time = time.time()
        if redis.llen('Frame') > 0:
            incoming_video = redis.lpop('Frame')
            video = pickle.loads(incoming_video)
            f = open('./video/%d.h264'%counter, 'wb+')
            f.write(video)
            counter += 1 

#### HUB ####             
def read_file_and_run(rgb_queue, of_queue):
    counter = 0
    while True:
        LIST_DIR = sorted(os.listdir('./video'))
        SORTED_DIR = [ int(x.split('.')[0]) for x in LIST_DIR]
        SORTED_DIR.sort()
        SORTED_DIR = [ str(x) + '.h264' for x in SORTED_DIR ]
        LENGTH = len(LIST_DIR)
        if LENGTH > 1 and LENGTH > counter:
            item = SORTED_DIR[counter]
            PATH = os.path.join('./video', item)
            cap = cv2.VideoCapture(PATH)
            counter += 1
            fake_counter = 0 
            opt_flow_list = []
            while cap.isOpened():
                ret, frame = cap.read()
                if ret == True:
                    fake_counter += 1
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb_queue.put((frame, ret))
                    #of_queue.put((frame, ret))
                    opt_flow_list.append(frame)
                else:
                    print("FAKE COUNTER", fake_counter, ret)
                    rgb_queue.put((frame, ret))
                    #of_queue.put((frame, ret))
                    of_queue.put(opt_flow_list)
                    break


def run_rgb_queue(rgb_queue):
   rgb_frame = []
   counter = 0
   while True:
       item = rgb_queue.get()
       if item[1] == True:    
           rgb_frame.append(item[0])
       else: #item[1] == False
           counter += 1
           print("RGB_QUEUE LENGTH: ", len(rgb_frame))
           rst = make_infer(args.rgb_weights, rgb_frame, rgb_net, 'RGB', args.test_segments, num_class)
           tmp_rst = np.argmax(np.mean(rst, axis=0))
           rgb_frame = []
           if args.dataset == 'ucf101':
                print("[RGB] THIS IS THE RESULT FROM OF of SECTION {}".format(counter), make_ucf()[tmp_rst])
           else: # args.dataset == 'hmdb51'
                print("[RGB] THIS IS THE RESULT FROM OF of SECTION {}".format(counter), make_hmdb()[tmp_rst])

def get_indices(length, num_seg, new_length):
    tick = (length - new_length + 1) / float(num_seg)
    offsets = np.array([int(tick / 2.0 + tick * x) for x in range(num_seg)])
    return offsets

def run_of_queue(of_queue, net):
    i = 1
    counter = 0

    cropping = torchvision.transforms.Compose([
        GroupOverSample(net.input_size, net.scale_size)
    ]) 
    transform = torchvision.transforms.Compose([ 
        cropping,
        Stack(True),
        ToTorchFormatTensor(False),
        GroupNormalize(net.input_mean, net.input_std)
    ])
    
    while True:
        beginning = time.time()
        of_frame = []
        item = of_queue.get()  # item == a stack of rgb frames yet to be optical-flow-extracted 
        new_length = 5 #how many consecutive optical frames to read 
        print("OF: " , len(item))
        offsets = get_indices(len(item), args.test_segments, new_length)
        print("OF: " , offsets)
        extracted_optical_flow = []
        ####NAIVE WAY TO EXTRACT FLOWS####
        if offsets[1] - offsets[0] < new_length:
            for i in range(len(item)-1):
                [flow_x, flow_y] = streaming(item[i], item[i+1])
                extracted_optical_flow.append(streaming(item[i], item[i+1]))
                rst = make_infer(args.of_weights, extracted_optical_flow, of_net, 'Flow', args.test_segments, num_class)
                tmp_rst = np.argmax(np.mean(rst, axis=0))
                print("RESULT OF Optical Flow: 1  = ", make_ucf()[tmp_rst])
        ####SPECIAL PART####
        else: #when we need a clever way to extract optical flow frames
            for index in offsets:
                for i in range(new_length):
                    start_streaming = time.time()
                    [flow_x , flow_y] = streaming(item[index+i], item[index+(i+1)])
                    end_streaming = time.time()
                    x_img, y_img = Image.fromarray(flow_x).convert('L'), Image.fromarray(flow_y).convert('L')
                    time_array = time.time()
                    of_frame.extend([x_img, y_img])
                    print("LEN OF OF", len(of_frame), "streaming time:", end_streaming-start_streaming, "Array to IMG: ", time_array-end_streaming)
            transform_time_1 = time.time()

            process_data = transform(of_frame)
            transform_time_2 = time.time()
            rst = eval_video(process_data, 2 * new_length, net, 'Flow', args.test_segments, num_class)
            eval_time_1 = time.time()
            tmp_rst = np.argmax(np.mean(rst, axis=0))
            print("TRANSFORMING TIME: ", transform_time_2-transform_time_1, eval_time_1-transform_time_2)
        counter += 1
        if args.dataset == 'ucf101':
            print("=====[OF]===== \n THIS IS THE RESULT FROM OF of SECTION {}".format(counter), make_ucf()[tmp_rst])
        else: # args.dataset == 'hmdb51'
            print("[OF] THIS IS THE RESULT FROM OF of SECTION {}".format(counter), make_hmdb()[tmp_rst])
        ending_time = time.time()
        print("HOW LONG IT TOOK to make ONE INFERENCE IN TOTAL OF SECTION {}".format(counter), ending_time-beginning)



'''
        if item[1] == False:
            print("IS ITEM TRUE OR FALSE???", item[1])
        tmp_of_frame.append(item)
        ###RUN OF INFERENCE###
        if len(tmp_of_frame) >= 2 and tmp_of_frame[i][1] == False:
            counter += 1
            start_infer = time.time()
            rst = make_infer(args.of_weights, of_frame, of_net, 'Flow', args.test_segments, num_class)
            end_infer = time.time()
            print("TIME FOR INFERENCE: ", end_infer-start_infer)
            tmp_rst = np.argmax(np.mean(rst, axis=0))
            print("BEFORE THE INF: ", len(of_frame), len(tmp_of_frame))
            of_frame, tmp_of_frame = of_frame[i:], tmp_of_frame[i+1:] 
            i = 1
            print("AFTER THE INFERENCE :", len(of_frame), len(tmp_of_frame), i)
            if args.dataset == 'ucf101':
                print("=====[OF]===== \n THIS IS THE RESULT FROM OF of SECTION {}".format(counter), make_ucf()[tmp_rst])
            else: # args.dataset == 'hmdb51'
                print("[OF] THIS IS THE RESULT FROM OF of SECTION {}".format(counter), make_hmdb()[tmp_rst])
        
        ###APPEND OF FRAMES TO LIST###
        if len(tmp_of_frame) >= 2 and tmp_of_frame[i-1][1] == True and tmp_of_frame[i][1] == True:
            start_streaming = time.time()
            of_frame.append(streaming(tmp_of_frame[i-1][0], tmp_of_frame[i][0]))
            end_streaming = time.time()
            i += 1
            print("OF_FRAME length: ", len(of_frame), end_streaming-start_streaming)
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sending streaming images from pi to hub')
    parser.add_argument('--video_path', type=str) 
    parser.add_argument('--hub_device', type=str, help='Specify where this will be run')  
    parser.add_argument('--rgb_weights', type=str)
    parser.add_argument('--of_weights', type=str)
    parser.add_argument('--dataset', type=str, default='ucf101')
    parser.add_argument('--arch',  type=str, default='BNInception')
    parser.add_argument('--crop_fusion_type', type=str, default='avg', choices=['avg', 'max', 'topk'])
    parser.add_argument('--dropout', type=float, default=0.7)
    parser.add_argument('--test_segments', type=int, default=5)
    args = parser.parse_args()

    if args.dataset == 'ucf101':
        num_class = 101
    elif args.dataset == 'hmdb51':
        num_class = 51 
    else:
        raise ValueError('Unknown dataset' + args.dataset)

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

    output = []
    label = []
     
    need_stop = Event()
    rgb_queue, of_queue = Queue(), Queue()
    
    host_address = '147.46.219.146'
    redis = Redis(host_address)
   
    redis.flushall()
    
    if args.hub_device == 'Hub':
        import shutil
        shutil.rmtree('./video')
        os.mkdir('./video')
        jobs = [ Thread(target=receive_and_save, args=(rgb_net, redis)),
                 Thread(target=read_file_and_run, args=(rgb_queue, of_queue)),
                 Thread(target=run_rgb_queue, args=(rgb_queue, )),
                 Thread(target=run_of_queue, args=(of_queue, of_net ))]
    else:
        jobs = [ Thread(target=streaming, args=(args.video_path, need_stop))]

    [job.start() for job in jobs]
    [job.join() for job in jobs]
    print("Terminating..")
    if args.hub_device == 'Hub':
        cf = confusion_matrix(label, output).astype(float)
        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)
        print("CLS CNT, HIT", cls_cnt, cls_hit)
        cls_acc = cls_hit / cls_cnt 
        print('Accuracy {:.02f}%'.format(np.mean(cls_acc) * 100))
'''
def run_of_queue(of_queue):
    of_extracted_list = Manager().list()
    while True:
        stack_of_frames = of_queue.get()
        LEN_STACK = len(stack_of_frames)
        proc = []
        print("HERE IN OF QUEUE FUNC: ", LEN_STACK)

        for i in range(LEN_STACK-1):
            print("THIS IS i", i)
            print("Validify i", stack_of_frames[i].shape)
            p = Process(target=streaming, args= (stack_of_frames[i], stack_of_frames[i+1], of_extracted_list, i))
            proc.append(p)
            p.start()
        #[ p.start() for p in proc]
        [ p.join() for p in proc]
        
        print("THIS IS THE LENGTH OF OF_EXTRACTED_LIST: ", len(of_extracted_list))
        rst = make_infer(args.of_weights, of_extracted_list, of_net, 'Flow', args.test_segments, num_class)
        temp_rst = np.argmax(np.mean(rst, axis=0))
        if args.dataset == 'ucf101':
            print("[OF] THIS IS THE RESULT FROM OF of SECTION {}".format(counter), make_ucf()[tmp_rst])
        else: # args.dataset == 'hmdb51'
            print("[OF] THIS IS THE RESULT FROM OF of SECTION {}".format(counter), make_hmdb()[tmp_rst])
        of_extracted_list = Manager().list()
'''

   
