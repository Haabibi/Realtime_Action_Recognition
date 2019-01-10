import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import numpy as np
from numpy.random import randint
import time 
class TSNDataSet(data.Dataset):
    def __init__(self, 
                 data,
                 modality,
                 image_tmpl, num_segments=3, new_length=1, transform=None,
                 force_grayscale=False, random_shift=True, test_mode=False):

        self.datalist = data
        print("[self.datalist]: ", len(self.datalist), data.shape, type(data))
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.num_frames = data.shape[0] 
        if self.modality == 'RGBDiff':
            self.new_length += 1# Diff needs one more image to calculate diff
    '''
    def _load_image(self, fifty_data, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            im = Image.fromarray(fifty_data[idx-1])
            return [im.convert('RGB')]
        elif self.modality == 'Flow':
            x_img = Image.fromarray(fifty_data[idx-1][0])
            y_img = Image.fromarray(fifty_data[idx-1][1]) 
            return [x_img.convert('L'), y_img.convert('L')] 
    ''' 
    def _get_test_indices(self):
        tick = (self.num_frames - self.new_length + 1) / float(self.num_segments)
        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        return offsets 
    

    def __getitem__(self, idx):
            
        if not self.test_mode:
            segment_indices = self._sample_indices() if self.random_shift else self._get_val_indices()
        else:
            segment_indices = self._get_test_indices()
            list_imgs = []
            for seg_ind in segment_indices:
                p = int(seg_ind)
                seg_img = self.datalist[seg_ind]
                time_array_yet = time.time()  
                im = Image.fromarray(seg_img, mode='RGB')
                print("[IM size Original]: ", im.size)
                im = im.resize((224, 224)) 
                print("[IM size after resize]: ", im.size)
                #im = transforms.functional.resize(im, (224, 224), interpolation=2),
                #print("[IM SIZE AFTER TRANSFORM]: ", type(im))
                time_image_now = time.time() 
                #print("[ARRAY --> IMG]: ", time_image_now-time_array_yet)
                #print("[time_array_yet]: ", time_array_yet, "[time_image_now]: ", time_image_now)
                list_imgs.append(im)
                if p < self.num_frames:
                    p += 1
                
                process_data = self.transform(list_imgs)
                #print("this is what i've got for process_data ", type(process_data), process_data.shape, "len of imgs", len(list_imgs))
        return process_data
    '''
    def get(self, indices):
        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(self.datalist, p)
                images.extend(seg_imgs)
                if p < self.num_frames:
                    p += 1

        process_data = self.transform(images)
        return process_data
    '''
    def __len__(self):
        return 1
