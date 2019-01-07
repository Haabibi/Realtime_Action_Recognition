# Fall Detection using Temporal Spatial Network 

**Note**: always use `git clone --recursive https://github.com/Haabibi/fall_detection_TSN.git` to clone this project. 
Otherwise you will not be able to use the inception series CNN archs. 

This is a reimplementation of temporal segment networks (TSN) in PyTorch. All settings are kept identical to the original caffe implementation.
This was adapted from  `https://github.com/yjxiong/tsn-pytorch`. 

## Version 
PyTorch: 0.3.1
TorchVision: 0.2.0

## Pretrained Model
The Temporal State Network was already pretrained on two datasets by splits of three on each dataset respectively: UCF101 and HMDB51.
Make a new directory named "**./ckpt/**" and download the checkpoints from this link: [GoogleDrive](https://drive.google.com/open?id=1lMRBsBLQlkKUSry0TqEeFBIxFVFc8Jrm)
Or run the following commands for your convenience. (It will only download one checkpoint file for each modality: RGB and Optical FFlow.) 
```bash
mkdir ckpt && cd ckpt
wget --no-check-certificate 'https://drive.google.com/open?id=13DLSYd_2jfwaSFo7Kud-BhoQWqQm2kOj' -O hmdb_rgb_1_ckpt.pth.tar
wget --no-check-certificate 'https://drive.google.com/open?id=1HSetROoXuBMw_xcMIujseG25Hsr-Vwcm' -O hmdb_flow_1_ckpt.pth.tar
```

## End-to-End Testing 
So far there exist three baselines for this project. 

#### Baseline1
Just a basic pipeline with no threads. Frames are already extracted in **pullup_img_frames** directory.  This baseline reads *all the already-extracted-frames* (no matter how many frames there are) from a directory.
To run Baseline1, run the following: 
```bash
python baseline1.py hmdb51 pullup_img_frames/ ./ckpt/hmdb51_rgb_1_ckpt.pth.tar ./ckpt/hmdb51_flow_1_ckpt.pth.tar --arch BNInception
```

#### Baseline2
Another basic pipeline with no threads. This baseline aims to reduce the time for loading data by not reading each frame from the disk but from _memory_. Execution of models happens when every 40 frames are appended to the RGB list, leading 39 frames to be left in the OF list. 
In order to get the average running time, two networks are run simultaneouly for ten times, and the scores from each network will be fused every iteration is over. 
To run Baseline2, run the following on your command line: 
```bash
python baseline2.py hmdb51 ./ckpt/hmdb51_bninception_1_rgb_checkpoint.pth.tar ./ckpt/hmdb51_bnInception_1_flow_checkpoint.pth.tar --q_size 40 --test_segments 10
```

#### Baseline2_1 (RGB)
In order to seek for optimization and shortening the latency of the current Baseline2 code, Baseline2_1 is explored. 
**(THIS IS WHERE I AM CURRENTLY AT)**
```bash
python baseline2_rgb.py hmdb51 ./ckpt/hmdb51_bninception_1_rgb_checkpoint.pth.tar ./ckpt/hmdb51_bnInception_1_flow_checkpoint.pth.tar --q_size 40 --test_segments 10
```

#### Baseline3
This would be the most similar pipeline to what we are pursuing using threads.
This stage will be explored after the successful investigation of former baselines. 

