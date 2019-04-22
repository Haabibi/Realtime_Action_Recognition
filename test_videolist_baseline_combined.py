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
from optical_flow_streaming import streaming 

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
    torch.cuda.set_device(0) if style == 'RGB' else torch.cuda.set_device(1)
    data = data.cuda() 
    input_var = torch.autograd.Variable(data.view(-1, length , data.size(1), data.size(2)), volatile=True)
    #torch.cuda.nvtx.range_push(style)
    rst = net(input_var)
    #torch.cuda.nvtx.range_pop()
    time_run_net = time.time()
    rst_data = rst.data.cpu().numpy().copy()

    output = rst_data.reshape((-1 , args.test_segments, num_class)).mean(axis=0).reshape((args.test_segments, 1, num_class))
    return output

def _get_indices(data, style):
    new_length = 1 if style == 'RGB' else 5 
    tick =  (len(data) - new_length + 1) / float(args.test_segments)
    offsets = np.array([int(tick / 2.0 + tick * x) for x in range(args.test_segments)])
    return offsets

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
            list_imgs.append(im) 
        if style == 'Flow':
            for i in range(5):
                seg_img = data[seg_ind + i] 
                x_img = Image.fromarray(seg_img[0])
                y_img = Image.fromarray(seg_img[1])
                list_imgs.extend([x_img.convert('L'), y_img.convert('L')])
    process_data = transform(list_imgs) 
    return process_data

def make_infer(weights, batched_array, net, style): 
    torch.cuda.set_device(0) if style == 'RGB' else torch.cuda.set_device(1)
    print("Current GPU for style {}: ".format(style), torch.cuda.current_device())
    net.float() 
    net.eval() 
    net = net.cuda() 
    data = _get_item(batched_array, net, style) 
    rst = eval_video(data, 3 if style =="RGB" else 10, net, style) 
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
    cuda.init()
    #######CHECK GPU STATUS#######
    print("NUM OF DEVICES: ", cuda.Device.count())
    gpu_list = [ i for i in range(cuda.Device.count())]
    print("THIS IS GPU LIST: ", gpu_list)

    #######LOADING RGB_NET#######
    #torch.cuda.nvtx.range_push('RGB NET')
    rgb_net = TSN(num_class, 1, 'RGB',
                  base_model=args.arch,
                  consensus_type=args.crop_fusion_type,
                  dropout=args.dropout)
    rgb_checkpoint = torch.load(args.rgb_weights)
    print("model epoch {} best prec@1: {}".format(rgb_checkpoint['epoch'], rgb_checkpoint['best_prec1']))
    base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(rgb_checkpoint['state_dict'].items())}
    rgb_net.load_state_dict(base_dict)
    torch.cuda.set_device(0)
    rgb_net = rgb_net.cuda()
    #print("[rgb_net_cuda]: ", next(rgb_net.parameters()).is_cuda, type(rgb_net))  # RETURNS TRUE
    #torch.cuda.nvtx.range_pop()

    #######LOADING OF_NET#######
    #torch.cuda.nvtx.range_push('OF NET')
    of_net = TSN(num_class, 1, 'Flow',
                  base_model=args.arch,
                  consensus_type=args.crop_fusion_type,
                  dropout=args.dropout)
    of_checkpoint = torch.load(args.of_weights)
    print("model epoch {} best prec@1: {}".format(of_checkpoint['epoch'], of_checkpoint['best_prec1']))
    base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(of_checkpoint['state_dict'].items())}
    of_net.load_state_dict(base_dict)
    torch.cuda.set_device(1) if len(gpu_list) > 0 else torch.cuda.set_device(0)
    of_net = of_net.cuda()
    #print("[of_net_cuda]: ", next(rgb_net.parameters()).is_cuda, type(of_net))  # RETURNS TRUE
    #torch.cuda.nvtx.range_pop() 
    
    rgb_output, of_output, video_labels = [], [], []
    video_data = open('../tsn-pytorch/ucf101_file_lists/video_ucf101_rgb_val_split_1.txt', 'r')
    #video_data = open('/home/haabibi/fall_detection/tsn-pytorch/ucf101_file_lists/short_video_ucf.txt', 'r')
    data_loader = video_data.readlines()
    counter = 0 
    accumulated_RGB_time, accumulated_OF_time = 0, 0 
    
    for video in data_loader: 
        video_load = video.split(' ') #video_load: video full link 
        cap = cv2.VideoCapture(video_load[0])
        video_labels.append(int(video_load[1][:-1]))#video_label
        rgb_list, _tmp_of, of_list, real_output = list(), list(), list(), list()
        output_results = {} 
        loading_frames =0
        rst = 0 
        accu_flow_toc = 0
        num_frames = 0 

        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                rgb_list.append(frame)
                _tmp_of.append(frame)
                first_time_rgb = time.time()
                if len(_tmp_of) >= 2: 
                    flow_tic= time.time()
                    of_list.append(streaming(_tmp_of[0], _tmp_of[1], 'tvl1'))
                    flow_toc = time.time()
                    _tmp_of.pop(0)
                    num_frames += 1 
                    accu_flow_toc += (flow_toc-flow_tic)
            else:
                counter+= 1
                rgb_rst = make_infer(args.rgb_weights, rgb_list, rgb_net, 'RGB')
                of_rst = make_infer(args.of_weights, of_list, of_net, 'Flow')
                print("video {} done, total {}/{}".format(counter-1, counter, len(data_loader) )) 
                rgb_output.append(rgb_rst)
                of_output.append(of_rst)
                
                break 
    
    rgb_video_pred = [np.argmax(np.mean(x, axis=0)) for x in rgb_output]
    rgb_cf = confusion_matrix(video_labels, rgb_video_pred).astype(float)
    rgb_cls_cnt = rgb_cf.sum(axis=1)
    rgb_cls_hit = np.diag(rgb_cf) 
    rgb_cls_acc = rgb_cls_hit / rgb_cls_cnt
    print(rgb_cls_acc) 
    print('Accuracy {:.02f}%'.format(np.mean(rgb_cls_acc) * 100))

    of_video_pred = [np.argmax(np.mean(x, axis=0)) for x in of_output]
    of_cf = confusion_matrix(video_labels, of_video_pred).astype(float)
    of_cls_cnt = of_cf.sum(axis=1) 
    of_cls_hit = np.diag(of_cf) 
    of_cls_acc = of_cls_hit / of_cls_cnt
    print(of_cls_acc) 
    print('Accuracy {:.02f}%'.format(np.mean(of_cls_acc) * 100))


    for i in range(len(rgb_output)):
        real_output.append((np.add(rgb_output[i], of_output[i])))
    video_pred = [np.argmax(np.mean(x, axis=0)) for x in real_output]
    print(video_pred)
    cf = confusion_matrix(video_labels, video_pred).astype(float) 
    cls_cnt = cf.sum(axis=1)

    cls_hit = np.diag(cf) 
    cls_acc = cls_hit / cls_cnt
    print(cls_acc) 
    print('Accuracy {:.02f}%'.format(np.mean(cls_acc) * 100))
    print("AVG TIME FOR LATENCY: RGB:{} OF:{}".format(accumulated_RGB_time/counter, accumulated_OF_time/counter), "just in case.. counter:{}, number of videos: {}".format(counter, len(rgb_output)))


