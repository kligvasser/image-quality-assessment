import torch.nn.functional as F
import numbers
import random
from PIL import Image

class NRandomCrop(object):
    def __init__(self, crop_size, num_crops=1, padding=0, pad_if_needed=False, seed=None):
        if isinstance(crop_size, numbers.Number):
            self.size = (int(crop_size), int(crop_size))
        else:
            self.size = crop_size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.num_crops = num_crops
        self.seed = seed

    @staticmethod
    def get_params(img, output_size, n, seed):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        if seed:
            random.seed(seed)

        i_list = [random.randint(0, h - th) for _ in range(n)]
        j_list = [random.randint(0, w - tw) for _ in range(n)]
        return i_list, j_list, th, tw

    def __call__(self, img):
        if self.padding > 0:
            img = F.pad(img, self.padding)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (int((1 + self.size[1] - img.size[0]) / 2), 0))
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, int((1 + self.size[0] - img.size[1]) / 2)))

        i, j, h, w = self.get_params(img, self.size, self.num_crops, self.seed)

        return n_random_crops(img, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)

def _is_pil_image(img):
    return isinstance(img, Image.Image)

def n_random_crops(img, x, y, h, w):
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    crops = []
    for i in range(len(x)):
        new_crop = img.crop((y[i], x[i], y[i] + w, x[i] + h))
        crops.append(new_crop)
    return tuple(crops)