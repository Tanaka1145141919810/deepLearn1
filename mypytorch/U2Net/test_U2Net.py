from mypytorch.U2Net.U2Net import REBNCONV
from mypytorch.U2Net.U2Net import conv_and_downsample
from mypytorch.U2Net.U2Net import U2Net
import torch
import torch.nn as nn
import unittest
from typing import Tuple
from typing import Optional
from typing import TypeAlias

class TestU2Net(unittest.TestCase):
    def test_REBNCONV(self):
        '''
        构造一个随机形状的Tensor输入测试
        '''
        randomsize : Tuple[int,int,int,int] = [64,3,480,480]
        random_Tensor : torch.Tensor  = torch.randn(randomsize)
        rebnconv : nn.Module = REBNCONV(in_chan=randomsize[1],out_chan=randomsize[1],dirate=1)
        random_out : torch.Tensor = rebnconv(random_Tensor)
        assert (random_Tensor.shape == random_out.shape)
    def test_upsample_like(self):
        pass
    def test_conv_and_downsample(self):
        randomsize : Tuple[int,int,int,int] = [64,3,480,480]
        random_input : torch.Tensor = torch.randn(randomsize)
        class Test_conv_and_downsample(nn.Module):
            def __init__(self,in_chan : int , out_chan : int)->None:
                super(Test_conv_and_downsample,self).__init__()
                self.layer = conv_and_downsample(in_chan= in_chan,out_chan=out_chan)
            def forward(self,input : torch.Tensor)->torch.Tensor:
                out = self.layer(input)
                return out
        net = Test_conv_and_downsample(in_chan=3,out_chan=12)
        random_output = net(random_input)
        assert (random_output.shape[2] == random_input.shape[2] / 2 and random_output.shape[3] == random_input.shape[3] /2)
    def test_U2Net(self):
        '''
        随机的测试数据
        '''
        random_size : Tuple[int,int,int,int] = [1,3,512,512]
        random_input : torch.Tensor = torch.randn(random_size).to("cuda:0")
        net = U2Net().to("cuda:0")
        d0,d1,d2,d3,d4,d5,d6 = net(random_input)
        print("do shape is {}\n".format(d0.shape))
        print("d1 shape is {}\n".format(d1.shape))
        ### [1,3,512,512] -> [1,1,512,512]
        
        
        
if __name__ == "__main__":
   TestU2Net().test_U2Net()