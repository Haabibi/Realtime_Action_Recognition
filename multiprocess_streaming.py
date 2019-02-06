import argparse
import cv2
from PIL import Image
from multiprocessing import Pool, current_process, Process
import time
import threading 
import numpy as np
import os 

def ToImg(raw_flow, bound=15):
    flow=raw_flow
    flow[flow>bound]=bound
    flow[flow<-bound]=-bound
    flow-=-bound
    flow*=(255/float(2*bound))
    return flow

def streaming(img1, img2, of_list, indx, flow_type='tvl1', bound=15):
    if flow_type == 'tvl1':
        gray_1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray_2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        dtvl1 = cv2.createOptFlow_DualTVL1()
        flowDTVL1 = dtvl1.calc(gray_1, gray_2, None) 
        
        flow_x = ToImg(flowDTVL1[...,0],bound) #type: np array
        flow_y = ToImg(flowDTVL1[...,1],bound)
    
    #of_list.append([flow_x, flow_y])
    of_list.insert(indx, [flow_x, flow_y])
    #return of_list



'''
if __name__ == '__main__':
    PATH ='/home/haabibi/fall_detection/fall_detection_TSN/Eye0706/' 
    img_list =  sorted(os.listdir(PATH))
    #num_list = [i for i in range(len(img_list))]
    #zipped = zip(num_list, img_list)
    of_list = []
    for i in range(int(len(img_list)/5)):
        i = i*5 
        
        proc = [ Process(target=streaming(os.path.join(PATH, img_list[i+0]), os.path.join(PATH, img_list[i+1]))),
                 Process(target=streaming(img_list[i+1], img_list[i+2])),
                 Process(target=streaming(img_list[i+2], img_list[i+3])),
                 Process(target=streaming(img_list[i+3], img_list[i+4])),
                 Process(target=streaming(img_list[i+4], img_list[i+5])) ]
    
        for p in proc:
            of_list.append(p)
            p.start()
         
        
        for p in proc: 
            p.join()
''' 
