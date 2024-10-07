import cv2
import os
import torch
import torch.utils.data as data
import torch.nn as nn
import numpy as np
from typing import Tuple

class DUSTDataset(data.Dataset):
    def __init__(self,root : str , train : bool , transforms : nn.Module = None)->None:
        '''
        root : 数据集根目录
        train : 是否是训练集
        transforms : 数据增强
        '''
        abs_path = os.getcwd()
        abs_path = os.path.join(abs_path,"dataset")
        ### 判断路径是否存在
        assert os.path.exists(abs_path), f'path {abs_path} does not exist'
        if train is True:
            self.img_root = os.path.join(abs_path,"DUTS-TR","DUTS-TR-Image")
            self.mask_root = os.path.join(abs_path,"DUTS-TR","DUTS-TR-Mask")
        else:
            self.img_root = os.path.join(abs_path,"DUTS-TE","DUTS-TE-Image")
            self.mask_root = os.path.join(abs_path,"DUTS-TE","DUTS-TE-Mask")
        ##print("imag_root is {}".format(self.img_root))
        assert os.path.exists(self.img_root) , f'path {self.img_root} does not exist'
        assert os.path.exists(self.mask_root) , f'path {self.mask_root} does not exist'
        
        img_name = [p for p in os.listdir(self.img_root) if p.endswith(".jpg")]
        mask_name = [p for p in os.listdir(self.mask_root) if p.endswith(".png")]
        
        self.img_path = [os.path.join(self.img_root,p) for p in img_name]
        self.mask_path = [os.path.join(self.mask_root ,p) for p in mask_name]
        self.transforms = transforms
        
    def __getitem__(self, index : int) -> Tuple[torch.Tensor,torch.Tensor]:
        img_path = self.img_path[index]
        mask_path = self.mask_path[index]
        img : np.ndarray = cv2.imread(img_path,flags=cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) ##BGR->RGB
        h,w,_ = img.shape
        target : np.ndarray= cv2.imread(mask_path,flags=cv2.IMREAD_GRAYSCALE)
        if self.transforms is not None:
            img,target = self.transforms(img,target)
        return img,target
    def __len__(self)->int:
        return len(self.img_path)
    
    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=0)

        return batched_imgs, batched_targets
    
def cat_list(images : torch.Tensor , fill_value : int = 0) -> torch.Tensor:
   max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
   batch_shape = (len(images),) + max_size
   batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
   for img, pad_img in zip(images, batched_imgs):
    pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
   return batched_imgs


if __name__ == '__main__':
    train_set = DUSTDataset(root="",train=True)
    print(len(train_set))
        
        
            
        
        