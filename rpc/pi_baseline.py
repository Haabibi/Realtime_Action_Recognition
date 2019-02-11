from queue import Queue
from data_sender import send
import argparse
from threading import Thread
import cv2
import numpy as np
from redis import Redis
import pickle
import torch
from models import TSN
from baseline1_test_rgb import make_ucf, make_infer

def load_image(video_path, queue):
    video_data = open(video_path, 'r')
    data_loader = video_data.readlines()
    frame_list = []
    for video in data_loader:
        video_load = video.split(' ')
        cap = cv2.VideoCapture(video_load[0])
        video_id = video_load[0].split('/')[-1]
        video_label = video_load[1].strip()
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_list.append(frame)
                #frame = frame.tostring()
            else:
                queue.put((video_id, frame_list, video_label))
                frame_list = []
                break

def network_sender(queue):
    while True:
        (video_id, frame_list, video_label) = queue.get()
        #(video_id, frame_list, len(frame_list)) = queue.get()
        #print("WHAT I GOT: ", queue.get())
        print("BEFORE SENDING IT TO REDIS: ", redis.llen('Key'))
        send('Key', (video_id, frame_list, video_label)) 
        print("WHAT I AM SENDING: ",(video_id, len(frame_list))) 

def receive_and_run_inference(rgb_net):
    counter = 0 
    while True:
        if redis.llen('Key') > 0:
            (video_id, frame_list, video_label) = pickle.loads(redis.lpop('Key')) 
            rst = make_infer(args.rgb_weights, frame_list, rgb_net, 'RGB')
            temp_rst = np.argmax(np.mean(rst, axis=0))
            output.append(rst) 
            label.append(video_label)
            #print("RECEIVE & RUN: " ,video_id, num_frames, len(frame_list), frame_list[0].shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sending streaming images from pi to hub')
    parser.add_argument('--video_path', type=str) 
    parser.add_argument('--hub_device', type=str, help='Specify where this will be run')  
    parser.add_argument('--rgb_weights', type=str)
    parser.add_argument('--dataset', type=str, default='ucf101')
    parser.add_argument('--arch',  type=str, default='BNInception')
    parser.add_argument('--crop_fusion_type', type=str, default='avg', choices=['avg', 'max', 'topk'])
    parser.add_argument('--dropout', type=float, default=0.7)
    args = parser.parse_args()

    redis_queue, finished_queue = Queue(), Queue()
    
    host_address = '147.46.219.146'
    redis = Redis(host_address)
   
    redis.flushall()
   
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
    

    if args.hub_device == 'Hub':
        jobs = [ Thread(target=receive_and_run_inference, args=(rgb_net, ))]
    else:
        jobs = [ Thread(target=load_image, args=(args.video_path, redis_queue)),
                 Thread(target=network_sender, args=(redis_queue,))] 
 
    [job.start() for job in jobs]
    [job.join() for job in jobs]
