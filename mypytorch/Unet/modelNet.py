import torch
import torch.nn as nn
import torch.nn.functional as F

from typing  import Tuple
from typing import Dict
from typing import Optional

### Unet中两个卷积层合并
class DoubleConv(nn.Sequential):
    '''
    Args:
        in_channel : int 输入通道数
        out_channel : int 输出通道数
        mid_channel : Optional[int] 中间的隐藏层
    Return:
        None
    '''
    def __init__(self,in_channels : int , out_channels : int ,mid_channels : Optional[int] = None)->None:
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv,self).__init__(
            ## 卷积公式 H_out = H_in + 2*padding -kernel_size /stride + 1
            ## 可见形状保持不变
            nn.Conv2d(in_channels = in_channels,
                      out_channels = mid_channels,
                      kernel_size= 3,
                      padding= 1,
                      bias= 1),
            nn.BatchNorm2d(num_features= mid_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels=mid_channels,
                      out_channels=out_channels,
                      kernel_size= 3,
                      padding= 1,
                      bias = False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace= True)
        )
## 定义下采样 H,W -> H/2,W/2
class Down(nn.Sequential):
    def __init__(self,in_channels : int ,out_channels : int)->None:
        super(Down,self).__init__(
            nn.MaxPool2d(kernel_size=2,stride=2),
            DoubleConv(in_channels=in_channels,out_channels=out_channels)
        )
## 定义上采样
class Up(nn.Module):
    def __init__(self,in_channels : int ,out_channels : int ,bilinear : bool = True)->None:
        '''
        Args:
            in_channel : int 输入
            out_channel : int 输出
            bilinear : bool 是否使用双线性插值
        '''
        super(Up,self).__init__()
        if bilinear :
            self.up = nn.Upsample(scale_factor=2,mode="bilinear",align_corners=True)
            self.conv = DoubleConv(in_channels=in_channels,out_channels=out_channels,mid_channels=in_channels //2)
        else :
            self.up = nn.ConvTranspose2d(in_channels=in_channels,
                                         out_channels= in_channels //2,
                                         kernel_size= 2,
                                         stride= 2,
                                         )
            self.conv = DoubleConv(in_channels=in_channels,out_channels=out_channels)
            
    def forward(self, x1 : torch.Tensor, x2 : torch.Tensor)->torch.Tensor:
        '''
        Args:
            x1 :  需要上采样的那个Tensor
            x2 :  拼接过来的那个Tensor
        Return:
            torch.Tensor : 返回的Tensor
        '''
        ## 上采样
        x1 = self.up(x1)
        x = self._match_size_and_concat(x1,x2)
        x = self.conv(x)
        return x
    
    def _match_size_and_concat(self,x1 : torch.Tensor , x2 : torch.Tensor) -> torch.Tensor:
        diff_x = x2.size()[2] - x1.size()[2]
        diff_y = x2.size()[3] - x1.size()[3]
        padding_size : Tuple[int,int,int,int] = (diff_x // 2,diff_x // 2, diff_y //2 ,diff_y //2)
        x1 = F.pad(x1,padding_size)
        x = torch.cat([x2,x1],dim=1)
        return x

class OutConv(nn.Sequential):
    def __init__(self,in_channels : int , num_classes : int)->None:
        super(OutConv,self).__init__(
            nn.Conv2d(in_channels=in_channels,
                      out_channels= num_classes,
                      kernel_size= 1)
        )
        
## 开始顶层的Unet
class UNet(nn.Module):
    def __init__(self,
                 in_channels : int = 1,
                 num_classes : int = 2,
                 bilinear : bool = True,
                 base_c : int = 64
                 )->None:
        super(UNet,self).__init__()
        self.in_channels  = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear
        self.base_c = base_c
        
        ### 进入图片时开始卷积
        self.in_conv = DoubleConv(in_channels=self.in_channels,
                                  out_channels=self.base_c)
        ### 三次下采样
        self.down1 = Down(in_channels= self.base_c,out_channels=self.base_c * 2)
        self.down2 = Down(in_channels= self.base_c * 2,out_channels= self.base_c * 4)
        self.down3 = Down(in_channels= self.base_c * 4,out_channels= self.base_c * 8)
        ### 确定是否采用双线性插值，再下采样
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)
        ### 四次上采样
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        ### 最后一层
        self.out_conv = OutConv(base_c, num_classes)
    def forward(self, x : torch.Tensor)-> Dict[str, torch.Tensor]:
        ### 保存到"out"字典中
        ### 经过每个层
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5,x4)
        x = self.up2(x,x3)
        x = self.up3(x,x2)
        x = self.up4(x,x1)
        x = self.out_conv(x)
        return {
            "out" : x
        }
        
        
        
        
        
        
            
        