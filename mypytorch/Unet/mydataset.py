### 对数据集进行处理

import os
import PIL.Image
import torch
from torch import nn
import torchvision.transforms as transforms
import torchvision
from PIL import Image
from typing import Tuple
import numpy as np

def show_type():
    img = PIL.Image.open("Drive_dataset/training/images/21_training.tif")
    print(type(img))
    
class ProcessingDataSet(nn.Module):
    def __init__(self, root : str , train : bool , transforms : torchvision.transforms = None):
        '''
        Args:
            root : 数据集的根文件夹
            train : 是否为训练集
            transforms : 数据集预处理的方式
        '''
        super(ProcessingDataSet, self).__init__()
        self.is_train = "train" if train is True else "test"
        self.data_path = os.path.join(root,"Drive_dataset","training" if self.is_train is True else "test")
        ### 如果文件不存在或者输入错误
        assert os.path.exists(self.data_path) , f"{self.data_path} does not exists\n"
        ### 如果有预处理模块的话
        self.transforms = transforms
        ### 这些图片都是以.tif结尾的
        ### 得到图片名字
        img_name = [i for i in os.listdir(os.path.join(self.data_path,"images")) if i.endswith(".tif")]
        ### 得到图片文件路径
        img_list = [os.path.join(self.data_path,"images",i) for i in img_name]
        ### 得到区分的图片
        self.manual = [os.path.join(self.data_path,"1st_manual",i.spilt("_")[0]) for i in img_name]
        ### 查看是否有遗落
        for i in self.manual:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"manual {i} does not exists\n")
        ## 读取ROI区域
        self.roi_mask = [os.path.join(self.data_path,"mask",i.split("_")[0],"training_mask.gif" if self.is_train is True else "test_mask.gif") for i in img_name]
        for i in self.roi_mask:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")
            
        def  __getitem__(self,idx : int)->Tuple[PIL.Image.Image,PIL.Image.Image]:
            img = Image.open(self.img_list[idx]).convert('RGB')
            manual = Image.open(self.manual[idx]).convert('L')
            manual = np.array(manual) / 255
            roi_mask = Image.open(self.roi_mask[idx]).convert('L')
            roi_mask = 255 - np.array(roi_mask)
            mask = np.clip(manual + roi_mask, a_min=0, a_max=255)
            mask = Image.fromarray(mask)
            if self.transforms is not None:
                img, mask = self.transforms(img, mask)
            return img, mask
        def __len__(self) -> int:
            return len(self.img_list)
        @staticmethod
        def collate_fn(batch):
            images, targets = list(zip(*batch))
            batched_imgs = cat_list(images, fill_value=0)
            batched_targets = cat_list(targets, fill_value=255)
            return batched_imgs, batched_targets
        
def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs
    
    
if __name__ == "__main__":
    show_type()