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
from baseline_rpc_rgb import make_ucf, make_infer, make_hmdb
from sklearn.metrics import confusion_matrix 
import os 

#### DEVIVCE ####
def streaming():
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
            start_redis = time.time()
            incoming_video = redis.lpop('Frame')
            stop_redis = time.time()
            print("TYPE OF FRAME: ", type(incoming_video), "Popping Time: ", stop_redis-start_redis)
            start_load = time.time()
            video = pickle.loads(incoming_video)
            end_load = time.time()
            f = open('./video/%d.h264'%counter, 'wb+')
            opened = time.time()
            f.write(video)
            write_file = time.time()
            #f.write(video.encode('utf-8'))
            print("[Pickle Loading Time: ]", end_load-start_load)
            print("[Video Open Time: ]", opened-end_load)
            print("[Video Write Time: ]", write_file - opened)
            counter += 1 

#### HUB ####             
def read_file_and_run():
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
            start_capture = time.time()
            cap = cv2.VideoCapture(PATH)
            end_capture = time.time()
            print("[VIDEO CAPTURE OF ", item, end_capture-start_capture) 
            counter += 1
            tmp_stack_frames = []
            before_opening_cap = time.time()
            while cap.isOpened():
                ret, frame = cap.read()
                new_frame = time.time()
                if ret == True:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    tmp_stack_frames.append(frame)
                    append_frame = time.time()
                    print("[APPENDING ONE FRAME EACH]: ", append_frame-new_frame)
                else: 
                    print("DONE READING ", item)
                    ready_to_infer = time.time()
                    rst = make_infer(args.rgb_weights, tmp_stack_frames[1:], rgb_net, 'RGB', args.test_segments, num_class)
                    inferred = time.time()
                    print("[TIME WHEN ALL FRAMES ARE READY]:{}, [INF TIME]: {}".format(ready_to_infer-before_opening_cap, inferred-ready_to_infer))
                    tmp_rst = np.argmax(np.mean(rst, axis=0))
                    print(make_hmdb()[tmp_rst])
                    break
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sending streaming images from pi to hub')
    parser.add_argument('--video_path', type=str) 
    parser.add_argument('--hub_device', type=str, help='Specify where this will be run')  
    parser.add_argument('--rgb_weights', type=str)
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
    
    output = []
    label = []
     
    redis_queue = Queue()
    need_stop = Event()
    
    host_address = '147.46.219.146'
    redis = Redis(host_address)
   
    redis.flushall()
    
    if args.hub_device == 'Hub':
        import shutil
        shutil.rmtree('./video')
        os.mkdir('./video')
        jobs = [ Thread(target=receive_and_save, args=(rgb_net, redis)),
                 Thread(target=read_file_and_run)]
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
   
