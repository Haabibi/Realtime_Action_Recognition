import argparse
import os
import glob
import cv2
from PIL import Image
from multiprocessing import Pool, current_process
import time
import threading 

def run_optical_flow(in_path, out_path, bound=15):
    ###vid_name = in_path.split('/')[-1]
    #print(vid_name)
    #out_full_path = os.path.join(out_path, vid_name)
    #print("this is out_full_path: ", out_full_path)
    #print("this is out_path: ", out_path)
    if len(os.listdir(in_path)) == 0: 
        pass
    else:
        frame1 = sorted(os.listdir(in_path))[0]
        frame1 = os.path.join(in_path, frame1)
        prevgray = cv2.cvtColor(cv2.imread(frame1), cv2.COLOR_BGR2GRAY)
        latency = 0
        for i in range(len(os.listdir(in_path))-1):
            tic = time.time()
            frame2 = sorted(os.listdir(in_path))[i+1]
            frame2 = os.path.join(in_path, frame2)
            gray = cv2.cvtColor(cv2.imread(os.path.join(frame2)), cv2.COLOR_BGR2GRAY)
            frame_0 = prevgray
            frame_1 = gray
            dtvl1 = cv2.createOptFlow_DualTVL1()
            flowDTVL1 = dtvl1.calc(frame_0, frame_1, None)
            save_flows(flowDTVL1, cv2.imread(frame1), out_path, i) 
            prevgray = gray
            frame1 = frame2
            toc = time.time()
            print("how much it took to process one optical flow", frame1,  toc-tic)
            latency += (toc - tic) 
     	
def save_flows(flows, image, out_path, num, bound=15):
   # if not os.path.exists(os.path.join(in_path, out_path)):
   #     os.makedirs(os.path.join(in_path, out_path))
    flow_x=ToImg(flows[...,0],bound)
    flow_y=ToImg(flows[...,1],bound)

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    save_img = os.path.join(out_path, 'img_{:05d}.jpg'.format(num))
    print("saving image here", save_img)
    cv2.imwrite(save_img, image)

    save_x=os.path.join(out_path, 'flow_x_{:05d}.jpg'.format(num))
    save_y=os.path.join(out_path, 'flow_y_{:05d}.jpg'.format(num))
    
    flow_x_img=Image.fromarray(flow_x)
    flow_y_img=Image.fromarray(flow_y)
    return 0

def ToImg(raw_flow, bound=15):
    flow=raw_flow
    flow[flow>bound]=bound
    flow[flow<-bound]=-bound
    flow-=-bound
    flow*=(255/float(2*bound))
    return flow

def streaming(in_path, out_path, flow_type='tvl1'):
    #print(threading.currentThread().getName(), 'Starting')
    if not os.path.isdir(out_path):
        print("Creating a new folder: " + out_path)
        os.makedirs(out_path)
    if flow_type == 'tvl1':
         print("starting to make optical_flows!")
         tic = time.time()
         run_optical_flow(in_path, out_path)
         toc = time.time()
         print("successfully done in {}!".format(toc-tic))
    print(threading.currentThread().getName(), 'Exiting')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="extracting optical flows")
    parser.add_argument("src_dir")
    parser.add_argument("out_dir")
    parser.add_argument("--flow_type", type=str, default = 'tvl1', choices=['tvl1', 'warped_tvl1'])
    parser.add_argument("--num_worker", type=int, default=8)
    parser.add_argument("--new_width", type=int, default=0, help='resize image width')
    parser.add_argument("--new_height", type=int, default=0, help='resize image height')
    parser.add_argument("--bound", type=int, default=15, help="set the max num of extracted optical flow")
    parser.add_argument("--num_gpu", type=int, default=8, help='number of GPU')
  
    args = parser.parse_args()
    
    in_path = args.src_dir
    out_path = args.out_dir
    new_size = (args.new_width, args.new_height)
    flow_type = args.flow_type
    NUM_GPU = args.num_gpu
    bound = args.bound
    if not os.path.isdir(out_path):
        print("Creating a new folder: " + out_path)
        os.makedirs(out_path)
    if flow_type == 'tvl1':
         print("starting to make optical_flows!")
         tic = time.time()
         run_optical_flow(in_path, out_path)
         toc = time.time()
         print("successfully done in {}!".format(toc-tic))



