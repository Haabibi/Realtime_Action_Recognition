import time 
import argparse
import numpy as np 
from sklearn.metrics import confusion_matrix
import cv2
from models import TSN
from transforms import * 
from ops import ConsensusModule
import pycuda.driver as cuda
from PIL import Image
from streaming import streaming 

def make_ucf():
    index_dir = '/cmsdata/hdd2/cmslab/haabibi/UCF101CLASSIND.txt'
    index_dict = {}
    with open(index_dir) as f:
        for line in f.readlines():
            s = line.split(' ')
            index_dict.update({int(s[0])-1: s[1]})
    return index_dict 

def make_hmdb():
    index_dir = './HMDB51CLASSIND.txt'
    index_dict = {}
    with open(index_dir) as f:
        for line in f.readlines():
            s = line.split(' ')
            index_dict.update({int(s[0]): s[1]})
    return index_dict

def eval_video(data, length, net, style):
    #torch.cuda.set_device(0) if style == 'RGB' else torch.cuda.set_device(1)
    data = data.cuda()
    input_var = torch.autograd.Variable(data.view(-1, length , data.size(1), data.size(2)), volatile=True)
    #torch.cuda.nvtx.range_push(style)
    net_tic = time.time()
    rst = net(input_var)
    net_toc = time.time()
    print("INF TIME: ", net_toc-net_tic)
    #torch.cuda.nvtx.range_pop()
    time_run_net = time.time()
    rst_data = rst.data.cpu().numpy().copy()

    output = rst_data.reshape((-1 , args.test_segments, num_class)).mean(axis=0).reshape((args.test_segments, 1, num_class))
    print("Output shape: " , output.shape)
    return output

def _get_indices(data, style):
    new_length = 1 if style == 'RGB' else 5 
    tick =  (len(data)  - new_length + 1) / float(args.test_segments)
    offsets = np.array([int(tick / 2.0 + tick * x) for x in range(args.test_segments)])
    
    return offsets + 1

def _get_item(data, net, style):
    list_imgs = []
    offsets = _get_indices(data, style)
    cropping = torchvision.transforms.Compose([
        GroupOverSample(net.input_size, net.scale_size)
    ]) 
    
    transform = torchvision.transforms.Compose([ 
        cropping,
        Stack(roll=args.arch == 'BNInception'),
        ToTorchFormatTensor(div=args.arch != 'BNInception'),
        GroupNormalize(net.input_mean, net.input_std)
    ])

    for seg_ind in offsets:
        if style == 'RGB':
            seg_img = data[seg_ind]
            im = Image.fromarray(seg_img, mode='RGB')
            #im = im.resize((224, 224))
            list_imgs.extend([im]) 
        if style == 'Flow':
            for i in range(5):
                seg_img = data[seg_ind + i] 
                x_img = Image.fromarray(seg_img[0])
                y_img = Image.fromarray(seg_img[1])
            #x_img = x_img.resize((224, 224))
            #y_img = y_img.resize((224, 224))
                list_imgs.extend([x_img.convert('L'), y_img.convert('L')])
    process_data = transform(list_imgs)
    return process_data

def make_infer(weights, batched_array, net, style): 
    #torch.cuda.set_device(0) if style == 'RGB' else torch.cuda.set_device(1)
    #print("Current GPU for style {}: ".format(style), torch.cuda.current_device())
    net.float() 
    net.eval() 
    net = net.cuda() 
    eval_vid_tic = time.time()
    time_data_tic = time.time()
    data = _get_item(batched_array, net, style) 
    time_data_toc = time.time()
    rst = eval_video(data, 3 if style =="RGB" else 10, net, style) 
    eval_vid_toc = time.time()
    return rst 

if __name__=="__main__":
    parser = argparse.ArgumentParser( 
        description = "Standard video-level testing")
    parser.add_argument('dataset', type=str, choices=['ucf101', 'hmdb51'])
    parser.add_argument('rgb_weights', type=str)
    parser.add_argument('of_weights', type=str)
    parser.add_argument('--arch', type=str, default="BNInception")
    parser.add_argument('--test_segments', type=int, default=25)
    parser.add_argument('--max_num', type=int, default=-1)
    parser.add_argument('--test_crops', type=int, default=10)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--crop_fusion_type', type=str, default='avg', choices=['avg', 'max', 'topk'])
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.7)
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
			help='number of data loading workers (default: 2)')
    parser.add_argument('--gpus', type=str, default=None)
    parser.add_argument('--flow_prefix', type=str, default='')
    parser.add_argument('--sliding_window', type=int, default=40)
    parser.add_argument('--num_repeat', type=int, default=1)
    #parser.add_argument('--vid_dir', type=str, default=None)
    parser.add_argument('--interval', type=int, default = 40)
    args = parser.parse_args()

    if args.dataset == 'ucf101':
        num_class = 101
        index_dict = make_ucf() 
    elif args.dataset == 'hmdb51':
        num_class = 51
        index_dict = make_hmdb()
    else:
        raise ValueError('Unknown dataset '+args.dataset)
    
    import os 
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    #cuda.init()
    #######CHECK GPU STATUS#######
    #print("NUM OF DEVICES: ", cuda.Device.count())
    #gpu_list = [ i for i in range(cuda.Device.count())]
    #print("THIS IS GPU LIST: ", gpu_list)

    #######LOADING RGB_NET#######
    #torch.cuda.nvtx.range_push('RGB NET')
    before = time.time()
    rgb_net = TSN(num_class, 1, 'RGB',
                  base_model=args.arch,
                  consensus_type=args.crop_fusion_type,
                  dropout=args.dropout)
    rgb_checkpoint = torch.load(args.rgb_weights)
    print("model epoch {} best prec@1: {}".format(rgb_checkpoint['epoch'], rgb_checkpoint['best_prec1']))
    base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(rgb_checkpoint['state_dict'].items())}
    rgb_net.load_state_dict(base_dict)
    #torch.cuda.set_device(0)
    #rgb_net = rgb_net.cuda()
    #print("[rgb_net_cuda]: ", next(rgb_net.parameters()).is_cuda, type(rgb_net))  # RETURNS TRUE
    #after = time.time() 
    #print("loading rgb_net: ", after-before)
    #torch.cuda.nvtx.range_pop()

    #######LOADING OF_NET#######
    #torch.cuda.nvtx.range_push('OF NET')
    before = time.time()
    of_net = TSN(num_class, 1, 'Flow',
                  base_model=args.arch,
                  consensus_type=args.crop_fusion_type,
                  dropout=args.dropout)
    of_checkpoint = torch.load(args.of_weights)
    print("model epoch {} best prec@1: {}".format(of_checkpoint['epoch'], of_checkpoint['best_prec1']))
    base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(of_checkpoint['state_dict'].items())}
    of_net.load_state_dict(base_dict)
    #torch.cuda.set_device(1) if len(gpu_list) > 0 else torch.cuda.set_device(0)
    #of_net = of_net.cuda()
    #print("[of_net_cuda]: ", next(rgb_net.parameters()).is_cuda, type(of_net))  # RETURNS TRUE
    after = time.time() 
    #print("loading of_net: ", after-before)
    #torch.cuda.nvtx.range_pop() 
    
    output, video_labels = [], []
    video_data = open('../tsn-pytorch/ucf101_file_lists/single_video.txt', 'r')
    #video_data = open('../tsn-pytorch/ucf101_file_lists/video_ucf101_rgb_val_split_1.txt', 'r')
    #video_data = open('../tsn-pytorch/ucf101_file_lists/short_video_ucf.txt', 'r')
    
    data_loader = video_data.readlines()
    counter = 0 
    output_results = {} 
    for video in data_loader: 
        video_load = video.split(' ') #video_load: video full link 
        video_labels.append(int(video_load[1].strip()))
        cap = cv2.VideoCapture(video_load[0])
        rgb_list, _tmp_of, of_list = list(), list(), list()
        loading_frames =0
        before11 = time.time()
        rst = 0
        num_frames = 0 
        #directory_write = open(os.path.join())
        while(cap.isOpened()):
            ret, frame = cap.read()
            accumulated_time_for_rgb = 0 
            accumulated_time_for_of = 0 
            accumulated_RGB, accumulated_OF = 0, 0 
            if ret == True:
                num_frames += 1 
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_id = num_frames
                rgb_list.append(frame)
                first_time_rgb = time.time()
            else:
                inf_tic = time.time()
                rst = make_infer(args.rgb_weights, rgb_list, rgb_net, 'RGB')
                offsets = _get_indices(rgb_list, 'RGB')
                inf_toc = time.time()
                print("video {} done, total {}/{}".format(counter, counter+1, len(data_loader) ), rst.shape) 
                counter+= 1
                output.append(rst)
                temp_rst = np.argmax(np.mean(rst[0], axis=0))
                output_results[video_load[0][37:]] = temp_rst
                print("[output length]: ", len(output), "how long it took to infer one video: ", inf_toc-inf_tic)
                break 
    print(output_results)
    video_pred = [np.argmax(np.mean(x[0], axis=0)) for x in output]
    print(video_labels, video_pred)
    cf = confusion_matrix(video_labels, video_pred).astype(float)
    print("this is cf:" ,cf)
    cls_cnt = cf.sum(axis=1)
    print("cls_cnt: ", cls_cnt)
    cls_hit = np.diag(cf) 
    print("cls_hit: ", cls_hit)
    cls_acc = cls_hit / cls_cnt
    print(cls_acc) 
    print('Accuracy {:.02f}%'.format(np.mean(cls_acc) * 100))
