import argparse
import cv2
from PIL import Image
from multiprocessing import Pool, current_process, Process, Manager
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

#def streaming(img1, img2, of_extracted_list, num, flow_type='tvl1', bound=15):
def streaming(img1, img2, flow_type='tvl1', bound=15):
    if flow_type == 'tvl1':
        gray_1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        gray_2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        
        start = time.time()
        dtvl1 = cv2.createOptFlow_DualTVL1()
        middle = time.time()
        flowDTVL1 = dtvl1.calc(gray_1, gray_2, None) 
        middle2 = time.time()

        flow_x = ToImg(flowDTVL1[...,0],bound) #type: np array
        flow_y = ToImg(flowDTVL1[...,1],bound)
        end = time.time()
        
        print("CREATE OP FLOW {0:8.4f}, DTVL1 CALC {1:8.4f}, TO IMG {2:8.4f}".format(middle-start, middle2-middle, end-middle2))
        return [flow_x, flow_y]
