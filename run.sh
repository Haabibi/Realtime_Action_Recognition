ls -d /cmsdata/hdd2/cmslab/haabibi/OPTICAL_FLOW_HMBD/* | while read d 
do
    python3 run_inf.py ucf101 RGB $d ucf101_bninception_1_rgb_checkpoint.pth.tar --arch BNInception | tee ~/UCF_outcome.txt
done
