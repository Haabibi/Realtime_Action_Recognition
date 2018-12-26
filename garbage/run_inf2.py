import argparse
import time
import os
import numpy as np
import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix

from dataset2 import TSNDataSet
from models import TSN
from transforms import *
from ops import ConsensusModule

# options
parser = argparse.ArgumentParser(
    description="Standard video-level testing")
parser.add_argument('dataset', type=str, choices=['ucf101', 'hmdb51', 'kinetics'])
parser.add_argument('modality', type=str, choices=['RGB', 'Flow', 'RGBDiff'])
parser.add_argument('root_path', type=str)
parser.add_argument('weights', type=str)
parser.add_argument('--arch', type=str, default="resnet101")
parser.add_argument('--save_scores', type=str, default=None)
parser.add_argument('--test_segments', type=int, default=25)
parser.add_argument('--max_num', type=int, default=-1)
parser.add_argument('--test_crops', type=int, default=10)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--crop_fusion_type', type=str, default='avg',
                    choices=['avg', 'max', 'topk'])
parser.add_argument('--k', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.7)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--flow_prefix', type=str, default='')

args = parser.parse_args()


if args.dataset == 'ucf101':
    num_class = 101
elif args.dataset == 'hmdb51':
    num_class = 51
elif args.dataset == 'kinetics':
    num_class = 400
else:
    raise ValueError('Unknown dataset '+args.dataset)

index_dir = '/cmsdata/hdd2/cmslab/haabibi/UCF101CLASSIND.txt'
index_dict = {}
with open(index_dir) as f:
     for line in f.readlines():
         s = line.split(' ')
         index_dict.update({int(s[0])-1: s[1]})

source = '/cmsdata/hdd2/cmslab/haabibi/HMBD51'
classes = os.listdir(source)
hmdb_datadict = {}
for one_class in classes:
    new_dir = os.path.join(source, one_class)
    video_list = os.listdir(new_dir)
    for one_video in video_list:
        hmdb_datadict.update({one_video[:-4]:one_class})

def eval_video(video_data):
    data = video_data
    num_crops = args.test_crops 
    if args.modality == 'RGB':
        length = 3
    elif args.modality == 'Flow':
        length = 10 
    elif args.modality == 'RGBDiff': 
        length = 18
    else:
        raise ValueError("Unknown modality "+args.modality)
   
    input_var = torch.autograd.Variable(data.view(-1, length, data.size(2), data.size(3)),
					volatile=True)
    rst = net(input_var).data.cpu().numpy().copy()
    return rst.reshape((num_crops, args.test_segments, num_class)).mean(axis=0).reshape(
	(args.test_segments, 1, num_class))

output_txt = open("result_ucftrained_multicam_1.txt", "w+") 

for video in sorted(os.listdir(args.root_path)):
    net = TSN(num_class, 1, args.modality,
              base_model=args.arch,
              consensus_type=args.crop_fusion_type,
              dropout=args.dropout)
    checkpoint = torch.load(args.weights)
    print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))
    base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
    net.load_state_dict(base_dict)
    if args.test_crops == 1:
        cropping = torchvision.transforms.Compose([
            GroupScale(net.scale_size),
            GroupCenterCrop(net.input_size),
        ])
    elif args.test_crops == 10:
        cropping = torchvision.transforms.Compose([
            GroupOverSample(net.input_size, net.scale_size)
        ])
    else:
        raise ValueError("Only 1 and 10 crops are supported while we got {}".format(args.test_crops))


    #loading data in 'torch'-readable format
    data_loader = torch.utils.data.DataLoader(
	    TSNDataSet(os.path.join(args.root_path, video), num_segments=args.test_segments,
		       new_length=1 if args.modality == "RGB" else 5,
		       modality=args.modality,
		       image_tmpl="img_{:05d}.jpg" if args.modality in ['RGB', 'RGBDiff'] else args.flow_prefix+"{}_{:05d}.jpg",
		       test_mode=True,
		       transform=torchvision.transforms.Compose([
			   cropping,
			   Stack(roll=args.arch == 'BNInception'),
			   ToTorchFormatTensor(div=args.arch != 'BNInception'),
			   GroupNormalize(net.input_mean, net.input_std),
		       ])),
	    batch_size=1, shuffle=False,
	    num_workers=args.workers * 2, pin_memory=True)

    if args.gpus is not None:
        devices = [args.gpus[i] for i in range(args.workers)]
    else:
        devices = list(range(args.workers))

    net = torch.nn.DataParallel(net.cuda(devices[0]), device_ids=devices)
    net.eval()

    max_num = args.max_num if args.max_num > 0 else len(data_loader.dataset)
    data_gen = enumerate(data_loader)
    for i, (data) in data_gen:
        if i >= max_num:
            break
        tic = time.time()
        rst = eval_video(data)
        video_pred = np.argmax(np.mean(rst[0], axis=0))
        toc = time.time()
        #print(video, hmdb_datadict[video], index_dict[video_pred])
        #output_txt.write(video + '\t' + hmdb_datadict[video] + '\t' + index_dict[video_pred] + '\t' + str(toc-tic) + '\n') 
        #print(video, label, index_dict[video_pred][:-1]) 
        #output_txt.write(video + '\t' + label + '\t' + index_dict[video_pred][:-1] + '\t' + str(toc-tic) + '\n')
        print(video, index_dict[video_pred][:-1])
        output_txt.write(video + '\t' +index_dict[video_pred][:-1] + '\t' + str(toc-tic) + '\n')  
if args.save_scores is not None:

    # reorder before saving
    name_list = [x.strip().split()[0] for x in open(args.test_list)]

    order_dict = {e:i for i, e in enumerate(sorted(name_list))}

    reorder_output = [None] * len(output)
    reorder_label = [None] * len(output)

    for i in range(len(output)):
        idx = order_dict[name_list[i]]
        reorder_output[idx] = output[i]
        reorder_label[idx] = video_labels[i]

    np.savez(args.save_scores, scores=reorder_output, labels=reorder_label)


