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
            popping_time = time.time()
            video = pickle.loads(incoming_video)
            loading_time = time.time()
            f = open('./video/%d.h264'%counter, 'wb+')
            opening_time = time.time()
            f.write(video)
            writing_time = time.time()
            counter += 1 
            print("POPPING TIME: {}, LOADING TIME: {}, OPENING TIME: {}, WRITING TIME: {}".format(popping_time-initial_time, loading_time-popping_time, opening_time-loading_time, writing_time-opening_time))

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
            video_time1 = time.time()
            cap = cv2.VideoCapture(PATH)
            video_time2 = time.time()
            counter += 1
            fake_counter = 0 
            opt_flow_list = []
            initial_time =time.time()
            while cap.isOpened():
                ret, frame = cap.read()
                if ret == True:
                    fake_counter += 1
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    before = time.time()
                    rgb_queue.put((frame, ret))
                    after1 = time.time()
                    #of_queue.put((frame, ret))
                    opt_flow_list.append(frame)
                    after2 = time.time()
                else:
                    rgb_queue.put((frame, ret))
                    rgb_done = time.time()
                    of_queue.put(opt_flow_list)
                    of_done = time.time()
                    print("[RGB READING FRAMES DONE IN {}] [OF READING DONE IN {}]".format(rgb_done-initial_time, of_done-initial_time))
                    break

def run_rgb_queue(rgb_queue, rgb_result_queue):
   rgb_frame = []
   counter = 0
   while True:
       item = rgb_queue.get()
       if item[1] == True:    
           rgb_frame.append(item[0])
       else: #item[1] == False
           counter += 1
           #print("RGB_QUEUE LENGTH: ", len(rgb_frame))
           time1 = time.time()
           rst = make_infer(args.rgb_weights, rgb_frame, rgb_net, 'RGB', args.test_segments, num_class)
           time2 = time.time()
           rgb_result_queue.put(rst)
           time3 = time.time()
           tmp_rst = np.argmax(np.mean(rst, axis=0))
           print("INFERENCE TIME:", time2-time1, time3-time2)
           rgb_frame = []
           if args.dataset == 'ucf101':
                print("[R{}]".format(counter), make_ucf()[tmp_rst])
           else: # args.dataset == 'hmdb51'
                print("[R{}]".format(counter), make_hmdb()[tmp_rst])

def get_indices(num_seg, tick):
    offsets = np.array([int(tick / 2.0 + tick * x) for x in range(num_seg)])
    return offsets

def run_of_queue(of_queue, net, rgb_result_queue):
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
        #print("OF: " , len(item))
        tick = (len(item) - new_length + 1) / float(args.test_segments)
        if tick < 2.0:
            continue
        offsets = get_indices(args.test_segments, tick)
        extracted_optical_flow = []
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
        of_rst = eval_video(process_data, 2 * new_length, net, 'Flow', args.test_segments, num_class)
        eval_time_1 = time.time()
        tmp_rst = np.argmax(np.mean(of_rst, axis=0))
        counter += 1
        print("\t\t\t[OF{}]".format(counter), make_ucf()[tmp_rst])
        rgb_rst = rgb_result_queue.get()
        rgb_tmp_rst = np.argmax(np.mean(rgb_rst, axis=0))
        fused = ( of_rst + rgb_rst ) / 2
        final_rst = np.argmax(np.mean(fused, axis=0))
        print("\t\t\t\t\t\t[{}] {} ({} / {})".format(counter,make_ucf()[final_rst], make_ucf()[rgb_tmp_rst] ,make_ucf()[tmp_rst]) )

        print("TRANSFORMING TIME: ", transform_time_2-transform_time_1, "SOON SOO INF", eval_time_1-transform_time_2)
        #if args.dataset == 'ucf101':
        #    print("\t\t\t\t\t[OF{}]: ".format(counter), make_ucf()[tmp_rst])
           # print("=====[OF]===== \n THIS IS THE RESULT FROM OF of SECTION {}".format(counter), make_ucf()[tmp_rst])
        #else: # args.dataset == 'hmdb51'
        #    print("\t\t\t\t\t[OF]: ".format(counter), make_hmdb()[tmp_rst])
            #print("[OF] THIS IS THE RESULT FROM OF of SECTION {}".format(counter), make_hmdb()[tmp_rst])
        ending_time = time.time()
        #print("HOW LONG IT TOOK to make ONE INFERENCE IN TOTAL OF SECTION {}".format(counter), ending_time-beginning)

def fuse_score(rgb_result, of_result):
    rgb_list, of_list = [], []
    counter = 0
    while True:
        if of_result.qsize() > 0 and rgb_result.qsize()>0:
            counter += 1
           # fused_score = (of_result.get() + rgb_result.get()) / 2
           # result = np.argmax(np.mean(fused_score, axis=0))
          #  print("\t\t\t\t\t\t\t\t[FUSED{}]: ".format(counter), make_ucf()[result])

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
    #print("model epoch {} best prec@1: {}".format(rgb_checkpoint['epoch'], rgb_checkpoint['best_prec1']))
    base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(rgb_checkpoint['state_dict'].items())}
    rgb_net.load_state_dict(base_dict)
   
    of_net = TSN(num_class, 1, 'Flow',
                  base_model=args.arch,
                  consensus_type=args.crop_fusion_type,
                  dropout=args.dropout)
    of_checkpoint = torch.load(args.of_weights)
    #print("model epoch {} best prec@1: {}".format(of_checkpoint['epoch'], of_checkpoint['best_prec1']))
    base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(of_checkpoint['state_dict'].items())}
    of_net.load_state_dict(base_dict)

    output = []
    label = []
     
    need_stop = Event()
    rgb_queue, of_queue, rgb_result = Queue(), Queue(), Queue()
    
    host_address = '147.46.219.146'
    redis = Redis(host_address)
   
    redis.flushall()
    
    if args.hub_device == 'Hub':
        import shutil
        shutil.rmtree('./video')
        os.mkdir('./video')
        jobs = [ Thread(target=receive_and_save, args=(rgb_net, redis)),
                 Thread(target=read_file_and_run, args=(rgb_queue, of_queue)),
                 Thread(target=run_rgb_queue, args=(rgb_queue, rgb_result)),
                 Thread(target=run_of_queue, args=(of_queue, of_net, rgb_result))]
    else:
        jobs = [ Thread(target=streaming, args=(args.video_path, need_stop))]

    [job.start() for job in jobs]
    [job.join() for job in jobs]
    #print("Terminating..")
    if args.hub_device == 'Hub':
        cf = confusion_matrix(label, output).astype(float)
        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)
        #print("CLS CNT, HIT", cls_cnt, cls_hit)
        cls_acc = cls_hit / cls_cnt 
        #print('Accuracy {:.02f}%'.format(np.mean(cls_acc) * 100))
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

   
