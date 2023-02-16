import os
from collections import namedtuple
from typing import Any, Callable, Optional, Tuple
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from tqdm import tqdm

def get_offset(label_y, label_x):
    coco_mean_x, coco_mean_y, coco_min_x, coco_min_y, coco_max_x, coco_max_y = np.mean(label_x).astype(np.int32), np.mean(label_y).astype(np.int32), np.min(label_x), np.min(label_y), np.max(label_x), np.max(label_y),
    return coco_mean_y, coco_mean_x, coco_mean_x - coco_min_x, coco_mean_y - coco_min_y, coco_max_x - coco_mean_x, coco_max_y - coco_mean_y

def slice_with_center(arr, center_y, center_x, actual_left_length, actual_up_length, actual_right_length, actual_down_length):
    if len(arr.shape) == 2:
        arr = arr[center_y-actual_up_length: center_y+actual_down_length].T
        arr = arr[center_x-actual_left_length: center_x+actual_right_length].T
    elif len(arr.shape) == 3:
        arr = arr[center_y-actual_up_length: center_y+actual_down_length].transpose((1, 0, 2))
        arr = arr[center_x-actual_left_length: center_x+actual_right_length].transpose((1, 0, 2))
    else:
        raise ValueError("wrong dimension number")
    return arr

class CS_COCO_Embedding(Dataset):
    def __init__(self, root: str = "./datasets/cs_coco_embedding/", mode: str = "gtFine", transform: Optional[Callable] = None, target_type: str = "semantic_train_id") -> None:
        self.root = root
        self.transform = transform
        self.images_dir = os.path.join(self.root, 'image')
        self.targets_dir = os.path.join(self.root, 'target')
        self.mode = 'gtFine' if "fine" in mode.lower() else 'gtCoarse'
        self.images = []
        self.targets = []

        for file_name in os.listdir(self.images_dir):
            self.images.append(os.path.join(self.images_dir, file_name))
            self.targets.append(os.path.join(self.targets_dir, file_name))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.targets[index])
        if self.transform is not None:
            image, target = self.transform(image, target)
        return image, target

    def __len__(self) -> int:
        return len(self.images)

    def prepare_data(self, cs_per_coco = 40, cropped_coco_root: str = "./datasets/cropped_coco/", cs_root: str = "./datasets/cityscapes/", mode: str = "gtFine", split: str = "train", target_type: str = "semantic_train_id"):
        shift = np.array([480, 480])
        high_limit = np.array([1024, 2048])
        low_limit = np.zeros(2)
        cs_images_dir = os.path.join(cs_root, 'leftImg8bit', split)
        cs_targets_dir = os.path.join(cs_root, mode, split)
        cs_images = []
        cs_targets = []
        for city in os.listdir(cs_images_dir):
            img_dir = os.path.join(cs_images_dir, city)
            target_dir = os.path.join(cs_targets_dir, city)
            for file_name in os.listdir(img_dir):
                target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                             self._get_target_suffix(mode, target_type))
                cs_images.append(os.path.join(img_dir, file_name))
                cs_targets.append(os.path.join(target_dir, target_name))
        for file_name in tqdm(os.listdir(cropped_coco_root)):
            cropped_coco_img = np.array(Image.open(os.path.join(cropped_coco_root, file_name)).convert('RGB'))
            coco_label_y, coco_label_x = np.where(np.sum(cropped_coco_img > 0, axis=2) > 0)
            coco_mean_y, coco_mean_x, coco_left_length, coco_up_length, coco_right_length, coco_down_length = get_offset(coco_label_y, coco_label_x)
            indices = np.random.choice(len(cs_images), cs_per_coco, replace=False)
            centers = np.random.randint((250, 250), (774, 1798), (cs_per_coco, 2)) # x, y
            high_bounds = np.where(centers+shift<high_limit, centers+shift, high_limit)
            low_bounds = np.where(centers-shift>low_limit, centers-shift, low_limit).astype(np.int32)
            for i, idx in enumerate(indices):
                cs_image = np.array(Image.open(cs_images[idx]).convert('RGB'))
                cs_target = np.array(Image.open(cs_targets[idx]))
                center_y, center_x = centers[i]
                high_bound, low_bound = high_bounds[i], low_bounds[i]
                actual_left_length, actual_up_length, actual_right_length, actual_down_length = min(coco_left_length, center_x-low_bound[1]), min(coco_up_length, center_y-low_bound[0]), min(coco_right_length, high_bound[1]-center_x), min(coco_down_length, high_bound[0]-center_y)
                coco_big_crop_img = slice_with_center(cropped_coco_img, coco_mean_y, coco_mean_x, actual_left_length, actual_up_length, actual_right_length, actual_down_length)
                left_padding = np.zeros((actual_down_length+actual_up_length, center_x-actual_left_length, 3))
                right_padding = np.zeros((actual_down_length+actual_up_length, 2048-center_x-actual_right_length, 3))
                coco_big_crop_img = np.concatenate([left_padding, coco_big_crop_img, right_padding], axis=1)
                cs_image[center_y-actual_up_length: center_y+actual_down_length] = np.where(np.stack([np.sum(coco_big_crop_img, axis=2)>0 for _ in range(3)], axis=2), coco_big_crop_img, cs_image[center_y-actual_up_length: center_y+actual_down_length])
                cs_target[center_y-actual_up_length: center_y+actual_down_length] = np.where(np.sum(coco_big_crop_img, axis=2)>0, 254, cs_target[center_y-actual_up_length: center_y+actual_down_length])
                embedding_img = Image.fromarray(slice_with_center(cs_image, 0, 0, -low_bound[1], -low_bound[0], high_bound[1], high_bound[0]))
                embedding_target = Image.fromarray(slice_with_center(cs_target, 0, 0, -low_bound[1], -low_bound[0], high_bound[1], high_bound[0]))
                embedding_file_name = '{}_{}.png'.format(file_name, i)
                embedding_img.save(os.path.join(self.images_dir, embedding_file_name))
                embedding_target.save(os.path.join(self.targets_dir, embedding_file_name))
    @staticmethod
    def _get_target_suffix(mode: str, target_type: str) -> str:
        if target_type == 'instance':
            return '{}_instanceIds.png'.format(mode)
        elif target_type == 'semantic_id':
            return '{}_labelIds.png'.format(mode)
        elif target_type == 'semantic_train_id':
            return '{}_labelTrainIds.png'.format(mode)
        elif target_type == 'color':
            return '{}_color.png'.format(mode)
        else:
            print("'%s' is not a valid target type, choose from:\n" % target_type +
                  "['instance', 'semantic_id', 'semantic_train_id', 'color']")
            exit()

if __name__ == '__main__':
    dataset = CS_COCO_Embedding()
    dataset.prepare_data()