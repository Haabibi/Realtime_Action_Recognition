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
from baseline_rpc_rgb import make_ucf, make_infer
from sklearn.metrics import confusion_matrix 

def streaming(video_path, event):
    while True:
        cap = cv2.VideoCapture(video_path)
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                send('Frame', frame)
            else:
                print("End of the streaming") 
                send('End', ret)
                break
        event.is_set()

def receive_frames(queue, redis, event):
    while True:
        if redis.llen('Frame') > 0:
            frame = pickle.loads(redis.lpop('Frame'))
            queue.put(frame)
        if redis.llen('Frame') == 0 and event.is_set():
            break 

def receive_and_run_inference(rgb_net, queue):
    counter = 0 
    frames_to_run = []
    while True:
        if queue.qsize()>0:
            frames_to_run.append(queue.get())
            print("LEN OF FRAMES TO RUN: ", len(frames_to_run))
            if len(frames_to_run) == 40: 
                print("MADE IT TON INF: ")
                rst = make_infer(args.rgb_weights, frames_to_run, rgb_net, 'RGB', args.test_segments, num_class)
                
                frames_to_run = frames_to_run[40:]
                temp_rst = np.argmax(np.mean(rst, axis=0))
                print(make_ucf()[temp_rst])
        if redis.llen('End') > 0 and queue.qsize() ==0:
           print("Hub Terminating.. ") 
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
    parser.add_argument('--test_segments', type=int, default=25)
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
        jobs = [ Thread(target=receive_and_run_inference, args=(rgb_net, redis_queue)),
                 Thread(target=receive_frames, args=(redis_queue, redis, need_stop))]
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
   
