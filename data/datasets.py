import torch
import os
import glob
from data.transforms import NRandomCrop
from torchvision import transforms
from PIL import Image

class DatasetEval(torch.utils.data.dataset.Dataset):
    def __init__(self, root_input='', root_target='', max_size=None, num_crops=1, crop_size=256):
        # parameters
        self.root_input = root_input
        self.root_target = root_target
        self.max_size = max_size
        self.num_crops = num_crops
        self.crop_size = crop_size

        self._init()

    def _init(self):
        # data paths
        input_paths, target_paths = [], []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
            input_paths += glob.glob(os.path.join(self.root_input, ext))
            target_paths += glob.glob(os.path.join(self.root_target, ext))
        self.paths = {'path_input': sorted(input_paths)[:self.max_size], 'path_target': sorted(target_paths)[:self.max_size]}

        # transforms
        if self.num_crops > 1:
            t_list = [NRandomCrop(crop_size=self.crop_size, num_crops=self.num_crops, seed=123),
            transforms.Lambda(lambda crops: torch.stack([(transforms.ToTensor()(crop)) for crop in crops]))]
        else:
            t_list = [transforms.ToTensor()]
        self.image_transform = transforms.Compose(t_list)

    def __getitem__(self, index):
        # images
        path_input = self.paths['path_input'][index]
        path_target = self.paths['path_target'][(index % len(self.paths['path_target']))]
        image_input = Image.open(path_input).convert('RGB')
        image_target = Image.open(path_target).convert('RGB')

        # transforms
        image_input = self.image_transform(image_input)
        image_target = self.image_transform(image_target)

        return {'input': image_input, 'target': image_target, 'path_input': path_input, 'path_target': path_target}

    def __len__(self):
        return len(self.paths['path_input'])
