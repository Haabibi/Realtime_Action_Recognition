import argparse
import cv2
from PIL import Image
from multiprocessing import Pool, current_process
import time
import threading 
import numpy as np

def ToImg(raw_flow, bound=15):
    flow=raw_flow
    flow[flow>bound]=bound
    flow[flow<-bound]=-bound
    flow-=-bound
    flow*=(255/float(2*bound))
    return flow

def streaming(img1, img2, flow_type='tvl1', bound=15):
    if flow_type == 'tvl1':
        gray_1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray_2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        dtvl1 = cv2.createOptFlow_DualTVL1()
        flowDTVL1 = dtvl1.calc(gray_1, gray_2, None) 
        
        flow_x = ToImg(flowDTVL1[...,0],bound) #type: np array
        flow_y = ToImg(flowDTVL1[...,1],bound)

    return [flow_x, flow_y]

