import time
from queue import Queue
from data_sender import send
import argparse
from threading import Thread, Event
import cv2
import numpy as np
from redis import Redis
import pickle
from baseline_rpc_rgb import make_ucf, make_infer
from sklearn.metrics import confusion_matrix 

def load_image(video_path, queue, event):
    video_data = open(video_path, 'r')
    data_loader = video_data.readlines()
    frame_list = []
    for video in data_loader:
        video_load = video.split(' ')
        cap = cv2.VideoCapture(video_load[0])
        video_id = video_load[0].split('/')[-1]
        video_label = video_load[1].strip()
        counter = 0 
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                counter += 1
                if counter % 40 != 0:
                    pass
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_list.append(frame)
            else:
                print("LEN OF FRAME_LIST", len(frame_list)) 
                queue.put((video_id, frame_list, int(video_label)))
                print("LEN OF QUEUE: ", queue.qsize())
                frame_list = []
                break
    event.set()

def network_sender(queue, event):
    while True:
        (video_id, frame_list, video_label) = queue.get()
        print("BEFORE SENDING IT TO REDIS: ", redis.llen('Key'))
        send('Key', (video_id, frame_list, video_label)) 
        print("Item that was just sent: ",(video_id, len(frame_list)), "HOW MANY LEFT IN QUEUE", queue.qsize()) 
        if queue.qsize() == 0 and event.is_set():
            print("Terminating the sender.. " )
            send('some_key', True)
            break

def receive_and_run_inference(rgb_net):
    counter = 0 
    while True:
        if redis.llen('Key') > 0:
            (video_id, frame_list, video_label) = pickle.loads(redis.lpop('Key')) 
            when_gotten_an_item = time.time() 
            print(video_id, video_label) 
            rst = make_infer(args.rgb_weights, frame_list, rgb_net, 'RGB', args.test_segments, num_class)
            right_after_inf = time.time()
            temp_rst = np.argmax(np.mean(rst, axis=0))
            output.append(temp_rst) 
            label.append(video_label)
            print("WHOLE INF TIME: ", right_after_inf -when_gotten_an_item)
            
            print("RECEIVE & RUN: ", video_id,video_label, "RESULT", temp_rst, make_ucf()[temp_rst])
        if redis.llen('Key') == 0 and redis.llen('some_key') > 0:
           print("Hub Terminating.. ") 
           break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sending streaming images from pi to hub')
    parser.add_argument('--video_path', type=str, default='') 
    parser.add_argument('--hub_device', type=str, help='Specify where this will be run')  
    parser.add_argument('--rgb_weights', type=str, default ='')
    parser.add_argument('--dataset', type=str, default='ucf101')
    parser.add_argument('--arch',  type=str, default='BNInception')
    parser.add_argument('--crop_fusion_type', type=str, default='avg', choices=['avg', 'max', 'topk'])
    parser.add_argument('--dropout', type=float, default=0.7)
    parser.add_argument('--test_segments', type=int, default=25)
    args = parser.parse_args()

    redis_queue = Queue()
    need_stop = Event()
    
    host_address = '147.46.219.146'
    redis = Redis(host_address)
   
    redis.flushall()
   
    if args.dataset == 'ucf101':
        num_class = 101
    elif args.dataset == 'hmdb51':
        num_class = 51 
    else:
        raise ValueError('Unknown dataset' + args.dataset)

    if args.hub_device == 'Hub':
        import torch
        from models import TSN
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
    
    if args.hub_device == 'Hub':
        jobs = [ Thread(target=receive_and_run_inference, args=(rgb_net, ))]
    else:
        jobs = [ Thread(target=load_image, args=(args.video_path, redis_queue, need_stop)),
                 Thread(target=network_sender, args=(redis_queue, need_stop))] 
 
    [job.start() for job in jobs]
    [job.join() for job in jobs]
    print("Terminating..")
    if args.hub_device == 'Hub':
        print("LEN OF OUTPUT / LABEL: ", len(output), len(label))
        cf = confusion_matrix(label, output).astype(float)
        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)
        print("CLS CNT, HIT", cls_cnt, cls_hit)
        cls_acc = cls_hit / cls_cnt 
        print('Accuracy {:.02f}%'.format(np.mean(cls_acc) * 100))
   
