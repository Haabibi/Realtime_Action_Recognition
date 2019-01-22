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
    torch.cuda.set_device(0) if style == 'RGB' else torch.cuda.set_device(1)
    data = data.cuda() 
    print("EVAL_VIDEO DATASIZE", data.shape, type(data), style, torch.cuda.current_device())
    input_var = torch.autograd.Variable(data.view(-1, length , data.size(1), data.size(2)), volatile=True)
    print("INPUT VAR", input_var.shape)
    torch.cuda.nvtx.range_push(style)
    rst = net(input_var)
    print("RST SHAPE", rst.shape, style)
    torch.cuda.nvtx.range_pop()
    time_run_net = time.time()
    rst_data = rst.data.cpu().numpy().copy()

    output = rst_data.reshape((-1 , args.test_segments, num_class)).mean(axis=0).reshape((args.test_segments, 1, num_class))
    print("THIS IS OUTPUT SHAPE FOR STYLE {}".format(style), output.shape)
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
        seg_img = data[seg_ind]
        if style == 'RGB':
            im = Image.fromarray(seg_img, mode='RGB')
            im = im.resize((224, 224))
            list_imgs.append(im) 
        if style == 'Flow':
            x_img = Image.fromarray(seg_img[0])
            y_img = Image.fromarray(seg_img[1])
            x_img = x_img.resize((224, 224))
            y_img = y_img.resize((224, 224))
            list_imgs.extend([x_img.convert('L'), y_img.convert('L')])
            print("FROM PROCESS_DATA: ", len(list_imgs), seg_img[0].shape, x_img.size)
    process_data = transform(list_imgs) 
    print("PROCESS DATA ", process_data.shape)
    return process_data

def make_infer(weights, batched_array, net, style): 
    torch.cuda.set_device(0) if style == 'RGB' else torch.cuda.set_device(1)
    print("Current GPU for style {}: ".format(style), torch.cuda.current_device())
    net.float() 
    net.eval() 
    net_cuda_tic = time.time()
    net = net.cuda() 
    net_cuda_toc = time.time() 
    print("[net_cuda_time]: ", net_cuda_toc-net_cuda_tic, next(net.parameters()).is_cuda)
    eval_vid_tic = time.time()
    time_data_tic = time.time()
    data = _get_item(batched_array, net, style) 
    print("THIS IS FROM MAKE_INFER Data: ", data.shape) 
    time_data_toc = time.time()
    rst = eval_video(data, 3 if style =="RGB" else 10, net, style) 
    eval_vid_toc = time.time()
    print("[getitem]: ", time_data_toc-time_data_tic, "[rst, eval_vid]: ", eval_vid_toc-eval_vid_tic)
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
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
			help='number of data loading workers (default: 2)')
    parser.add_argument('--gpus', type=str, default=None)
    parser.add_argument('--flow_prefix', type=str, default='')
    parser.add_argument('--sliding_window', type=int, default=40)
    parser.add_argument('--num_repeat', type=int, default=1)
    parser.add_argument('--vid_dir', type=str)
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
    torch.cuda.nvtx.range_push('RGB NET')
    before = time.time()
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
    print("[rgb_net_cuda]: ", next(rgb_net.parameters()).is_cuda, type(rgb_net))  # RETURNS TRUE
    after = time.time() 
    print("loading rgb_net: ", after-before)
    torch.cuda.nvtx.range_pop()

    #######LOADING OF_NET#######
    torch.cuda.nvtx.range_push('OF NET')
    before = time.time()
    of_net = TSN(num_class, 1, 'Flow',
                  base_model=args.arch,
                  consensus_type=args.crop_fusion_type,
                  dropout=args.dropout)
    of_checkpoint = torch.load(args.of_weights)
    print("model epoch {} best prec@1: {}".format(of_checkpoint['epoch'], of_checkpoint['best_prec1']))
    base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(of_checkpoint['state_dict'].items())}
    of_net.load_state_dict(base_dict)
    torch.cuda.set_device(1) if len(gpu_list) != 0 else torch.cuda.set_device(0)
    of_net = of_net.cuda()
    print("[of_net_cuda]: ", next(rgb_net.parameters()).is_cuda, type(of_net))  # RETURNS TRUE
    after = time.time() 
    print("loading of_net: ", after-before)
    torch.cuda.nvtx.range_pop() 

    cap = cv2.VideoCapture(args.vid_dir)
    rgb_list, _tmp_of, of_list = list(), list(), list()
    output_results = {} 
    counter = 0 
    loading_frames =0
    before11 = time.time()
    while(cap.isOpened()):
        ret, frame = cap.read()
        accumulated_time_for_rgb = 0 
        accumulated_time_for_of = 0 
        accumulated_RGB, accumulated_OF = 0, 0
        if ret == True:
            rgb_list.append(frame)
            _tmp_of.append(frame)
            first_time_rgb = time.time()
            if len(_tmp_of) >= 2: 
                of_list.append(streaming(_tmp_of[0], _tmp_of[1], 'tvl1'))
                _tmp_of.pop(0)
            if len(rgb_list) == args.sliding_window :
                loading_frames += (time.time() - first_time_rgb)
                counter += 1
                got_here_rgb = time.time() 
                print("[{}] How many seconds were needed to get full list ".format(counter), got_here_rgb-first_time_rgb)
                for i in range(args.num_repeat+1):
                    if i == 0:
                        cold_case_tic = time.time()
                        make_infer(args.rgb_weights, rgb_list, rgb_net, 'RGB')
                        cold_case_toc = time.time()
                        make_infer(args.of_weights, of_list, of_net, 'Flow') 
                        cold_case_toc2 = time.time() 
                        print("cold case inf time for {RGB}: ", cold_case_toc-cold_case_tic)
                        print("cold case inf time for {OF}: ", cold_case_toc2-cold_case_toc, "\n")
                    else: 
                        rgb_inf_tic = time.time()
                        torch.cuda.nvtx.range_push('RGB'+ str(counter))
                        rgb_inference = make_infer(args.rgb_weights, rgb_list, rgb_net, 'RGB')
                        torch.cuda.nvtx.range_pop()
                        rgb_inf_toc = time.time() 
                        
                        of_inf_tic = time.time()
                        torch.cuda.nvtx.range_push('OF' + str(counter))
                        of_inference = make_infer(args.of_weights, of_list, of_net, 'Flow') 
                        torch.cuda.nvtx.range_pop()
                        of_inf_toc = time.time() 
                        print("[EACH RUN TIME {}] RGB: {}, OF: {}".format(i, rgb_inf_toc-rgb_inf_tic, of_inf_toc-of_inf_tic)) 
                        #rgb_pred = np.argmax(np.mean(rgb_inference[0], axis=0))
                        #of_pred = np.argmax(np.mean(of_inference[0], axis=0))
                        #print("this is rgb_pred: ", make_hmdb()[rgb_pred], "this is of_pred: ", make_hmdb()[of_pred])
                         
                        torch.cuda.nvtx.range_push('COMPUTATION')
                        torch.cuda.set_device(2)
                        
                        score_fusion = (rgb_inference + of_inference)/2
                        video_pred = np.argmax(np.mean(score_fusion[0], axis=0))
                        output_results[counter] = make_hmdb()[video_pred]

                        print(make_hmdb()[video_pred])
                        torch.cuda.nvtx.range_pop()
                        accumulated_time_for_rgb += (rgb_inf_toc-rgb_inf_tic)
                       
                        accumulated_time_for_of += (of_inf_toc - of_inf_tic) 
                print("accumulated time for rgb: {:.7f}".format(accumulated_time_for_rgb/(args.num_repeat)))
                print("accumulated timeefor of: {:.7f}".format(accumulated_time_for_of/(args.num_repeat)))
                accumulated_RGB += accumulated_time_for_rgb / args.num_repeat
                accumulated_OF += accumulated_time_for_of / args.num_repeat

                accumulated_time_for_rgb = 0 
                accumulated_time_for_of = 0 
                rgb_list = rgb_list[args.interval:] 
                of_list = of_list[args.interval:]
             
        else:
            print("END_TO_END LATENCY] ", time.time() -before11)
            print("LOADING FRAMES AVG", loading_frames/counter)
            print("avg inf time", accumulated_RGB/counter)
            print("avg inf OF time", accumulated_OF/counter)
            print("OUTPUT RESULTS: ", output_results)
            #torch.cuda.nvtx.range_pop()
            break
