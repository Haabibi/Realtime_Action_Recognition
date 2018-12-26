for i in {1.. 3}; 
do python main.py hmdb51 RGB ./hmdb51_file_lists/hmdb51_rgb_train_split_$i.txt ./hmdb51_file_lists/hmdb51_rgb_val_split_$i.txt --arch BNInception --num_segments 3 --gd 20 --lr_steps 30 60 --epochs 80 -b 128 -j 8 --dropout 0.8 --snapshot_pref hmdb51_bninception_$i;
done
