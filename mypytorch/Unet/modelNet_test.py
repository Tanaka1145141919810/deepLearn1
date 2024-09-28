## 针对modelNet的单元测试
import unittest
import torch
import torch.nn as nn
from typing import Tuple
from mypytorch.Unet.modelNet import DoubleConv
from mypytorch.Unet.modelNet import Down
from mypytorch.Unet.modelNet import UNet

class TestDoubleConv(unittest.TestCase):
    def test_DoubleConv_random(self):
        '''
        构造一个随即大小的输入,检查输出的形状是否符合要求
        '''
        random_size : Tuple[int,int,int,int] = (64,1,480,480) ##[B,C,H,W]
        random_input : torch.Tensor = torch.randn(random_size)
        class Layer(nn.Module):
            def __init__(self,in_channels : int , out_channels : int)->None:
                super(Layer,self).__init__()
                self.layer = DoubleConv(in_channels=in_channels,out_channels=out_channels)
            def forward(self, x : torch.Tensor) -> torch.Tensor:
                out = self.layer(x)
                return out
        layer = Layer(in_channels=random_size[1],out_channels=random_size[1])
        random_output = layer(random_input)
        print("DoubleConv random_output size is {}".format(random_output.shape))
        assert(random_output.shape == random_input.shape)
    def test_Down_random(self):
        '''
        构造一个随机的输入,测试输出的形状
        '''
        random_size : Tuple[int,int,int,int] = (64,1,480,480)
        random_input : torch.Tensor = torch.randn(random_size)
        class Layer(nn.Module):
            def __init__(self,in_channels : int ,out_channels : int)->None:
                super(Layer,self).__init__()
                self.layer = Down(in_channels=in_channels,out_channels=out_channels)
            def forward(self,x : torch.Tensor)-> torch.Tensor:
                out = self.layer(x)
                return out
        layer = Layer(in_channels = random_size[1],out_channels=random_size[1])
        random_output = layer(random_input);
        print("Down random_output size is {}".format(random_output.shape))
    def test_bilinear_Upsample(self):
        '''
        测试双线性插值
        '''
        random_size : Tuple[int,int] = (1,1,4,4)
        random_input : torch.Tensor = torch.randn(random_size)
        random_output = nn.Upsample(scale_factor=2,mode="bilinear",align_corners=True)(random_input)
        print("bilinear_Upsample random_input is {}".format(random_input[0][0]))
        print("bilinear_Upsample random_output is {}".format(random_output[0][0]))
        
    def test_myUnet(self):
        '''
        测试我的Unet,用随机的数据
        '''
        random_size : Tuple[int,int,int,int] = (64,1,48,48)
        net = UNet(in_channels=1,
                   num_classes=2,
                   bilinear=True,
                   base_c=64)
        random_input : torch.Tensor = torch.randn(random_size)
        random_output = net(random_input)
        print("myUnet random_output size is {}".format(random_output["out"].shape))
        '''
            输出的形状是[64,2,48,48]无问题
        '''
        
        
if __name__ == "__main__":
    unittest.main()
        
        