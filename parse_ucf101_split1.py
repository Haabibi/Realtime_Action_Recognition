import os

ucf_dir = '/home/haabibi/fall_detection/tsn-pytorch/ucf101_file_lists/modified_ucf101_rgb_val_split_1.txt'

read_file = open(ucf_dir, 'r')
content = read_file.readlines()
new_dir = '/home/haabibi/fall_detection/fall_detection_TSN/SPLIT1'
write_file = open('/home/haabibi/fall_detection/tsn-pytorch/ucf101_file_lists/ESCtoDisk.txt', 'a+') 

for line in content:
    items = line.split(' ')
    vid_category = items[0].split('/')[-1]
    num_frames = int(items[1])
    vid_dir = os.path.join(new_dir, vid_category)
    write_file.write(str(vid_dir) + ' ' + str(num_frames+1) + ' ' + items[2])


