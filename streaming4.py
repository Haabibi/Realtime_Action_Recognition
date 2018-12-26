import argparse
import cv2
from PIL import Image
from multiprocessing import Pool, current_process
import time
import threading 
import numpy as np

def run_optical_flow(img1, img2, bound=15):
    gray_1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    dtvl1 = cv2.createOptFlow_DualTVL1()
    flowDTVL1 = dtvl1.calc(gray_1, gray_2, None) 
    
    flow_x = ToImg(flowDTVL1[...,0],bound) #type: np array
    flow_y = ToImg(flowDTVL1[...,1],bound)
    #flow_x_img = Image.fromarray(flow_x) #type: PIL 
    #flow_y_img = Image.fromarray(flow_y) 
    return [flow_x, flow_y]
        
def ToImg(raw_flow, bound=15):
    flow=raw_flow
    flow[flow>bound]=bound
    flow[flow<-bound]=-bound
    flow-=-bound
    flow*=(255/float(2*bound))
    return flow

def streaming(img1, img2, flow_type='tvl1'):
    if flow_type == 'tvl1':
#        print("starting to make optical_flows!")
    #    tic = time.time()
        output = run_optical_flow(img1, img2)
     #   toc = time.time()
      #  print("successfully extracted optical flow in {}!".format(toc-tic))
        return output
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="extracting optical flows")
    parser.add_argument("--flow_type", type=str, default = 'tvl1', choices=['tvl1', 'warped_tvl1'])
    parser.add_argument("--num_worker", type=int, default=8)
    parser.add_argument("--new_width", type=int, default=0, help='resize image width')
    parser.add_argument("--new_height", type=int, default=0, help='resize image height')
    parser.add_argument("--bound", type=int, default=15, help="set the max num of extracted optical flow")
    parser.add_argument("--num_gpu", type=int, default=8, help='number of GPU')
  
    args = parser.parse_args()
    
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
'''


