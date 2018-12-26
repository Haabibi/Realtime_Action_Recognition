import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint
import glob

class TSNDataSet(data.Dataset):
    def __init__(self, 
                 fifty_data,
                 modality,
                 image_tmpl, num_segments=3, new_length=1, transform=None,
                 force_grayscale=False, random_shift=True, test_mode=False):

        self.datalist = fifty_data
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.num_frames = len(fifty_data) 
        if self.modality == 'RGBDiff':
            self.new_length += 1# Diff needs one more image to calculate diff

        self._parse_list()

    def _load_image(self, fifty_data, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            im = Image.fromarray(fifty_data[idx-1])
            return [im.convert('RGB')]
        elif self.modality == 'Flow':
            x_img = Image.fromarray(fifty_data[idx-1][0])
            y_img = Image.fromarray(fifty_data[idx-1][1]) 
            return [x_img.convert('L'), y_img.convert('L')] 

    def _parse_list(self):
        self.video_list = self.datalist

    def _sample_indices(self):
        average_duration = (self.num_frames - self.new_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif self.num_frames > self.num_segments:
            offsets = np.sort(randint(self.num_frames - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_val_indices(self):
        if self.num_frames > self.num_segments + self.new_length - 1:
            tick = (self.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_test_indices(self):
        tick = (self.num_frames - self.new_length + 1) / float(self.num_segments)

        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        return offsets + 1

    def __getitem__(self, idx):

	# TODO Specify which file to load
        #record = self.video_list[index]

        if not self.test_mode:
            segment_indices = self._sample_indices() if self.random_shift else self._get_val_indices()
        else:
            segment_indices = self._get_test_indices()

	# TODO Return the result of transform
        return self.get(segment_indices)

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

    def __len__(self):
        return len(self.datalist)
