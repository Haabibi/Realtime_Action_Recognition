import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint
import glob

class TSNDataSet(data.Dataset):
    def __init__(self, root_path,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 force_grayscale=False, random_shift=True, test_mode=False):

        self.root_path = root_path
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        num_frames = 0
        for i in glob.iglob(os.path.join(self.root_path, 'img*'), recursive=True):
            num_frames += 1
        self.num_frames = num_frames 
        if self.modality == 'RGBDiff':
            self.new_length += 1# Diff needs one more image to calculate diff

        self._parse_list()

    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]
        elif self.modality == 'Flow':
            x_img = Image.open(os.path.join(directory, self.image_tmpl.format('x', idx))).convert('L')
            y_img = Image.open(os.path.join(directory, self.image_tmpl.format('y', idx))).convert('L')
            return [x_img, y_img]

    def _parse_list(self):
        self.video_list = [self.root_path]

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
  #      assert idx == 0

	# TODO Specify which file to load
        #record = self.video_list[index]

        if not self.test_mode:
            segment_indices = self._sample_indices() if self.random_shift else self._get_val_indices()
        else:
            segment_indices = self._get_test_indices()

	# TODO Return the result of transform
        print(segment_indices)
        return self.get(segment_indices)

    def get(self, indices):
        images = list() 
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(self.root_path, p)
                images.extend(seg_imgs)
                if p < self.num_frames:
                    p += 1
        
        process_data = self.transform(images)
       
        return process_data

    def __len__(self):
        return 1
     #   return len(os.listdir(self.root_path))
