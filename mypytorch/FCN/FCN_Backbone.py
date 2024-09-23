### 参考bilibili霹雳啪啦wz的视频
### 参考Pytorch的官方代码实现
### 导入必要的库
import torch
import torch.nn as nn

from typing import Tuple

####定义模型
####这里官方采用的是ResNet，详细原理可以去看ResNet的解析

def conv3x3(in_chan : int ,out_chan : int , stride : int = 1,groups : int = 1,dilation : int = 1):
    '''
    Args:
        in_chan : 输入
        out_chan : 输出
        stride : 步长
        dilation : 膨胀系数
    '''
    return nn.Conv2d(in_channels=in_chan,
                     out_channels= out_chan,
                     kernel_size=3,
                     stride=1,
                     padding=dilation,
                     groups=groups,
                     bias=False,
                     dilation=dilation)
### 1x1的卷积核
def conv1x1(in_chan : int , out_chan : int ,stride : int = 1):
    '''
    Args:
        in_chan : 输入
        out_chan : 输出
        stride : 步长
    '''
    return nn.Conv2d(in_channels=in_chan,
                     out_channels=out_chan,
                     kernel_size=1,
                     stride=stride,
                     bias=False)

class BottleNeck(nn.Module):
    ### expansion 表示最后经过backbone之后输出相比于输入要扩大4
    ### base_width 用来控制中间层的宽度 ？ 我也不知道为什么要这么设计
    def __init__(self, in_chan : int ,out_chan : int , stride : int = 1,downsample = None,
                 groups : int = 1,
                 dilation : int = 1,
                 norm_layer = None,
                 base_width = 64)-> None:
        super(BottleNeck,self).__init__()
        expansion : int = 4
        if norm_layer is None:
            ### 设置norm_layer为BN
            norm_layer = nn.BatchNorm2d
        width = int(in_chan * (base_width / 64.)) * groups
        ### 卷积输入和输出的维度关系
        ### H_out = (H_in + 2*padding - kernel)/ stride + 1
        ### W_out = (W_in + 2*padding - kernel)/ stride + 1
        self.conv1 = conv1x1(in_chan=in_chan,out_chan=width)
        ### base_width = 64 , groups = 1 时形状保持不变
        self.bn1 = norm_layer(width)
        ### 带膨胀的卷积输入的输出的关系
        ### H_out = (H_in + 2*padding - (kernel - 1)*dilation -1)/stride + 1
        ### W_out = (W_in + 2*padding - (kernel - 1)*dilation -1)/stride + 1
        ### 这里 H_out = H_in + 2*1 - (3 -1)*1 - 1 /1 + 1 = H_in 尺寸不变
        self.conv2 = conv3x3(in_chan=width,out_chan=width,stride=stride,groups=groups,dilation=dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(in_chan=width,out_chan=in_chan * expansion)
        self.bn3 = norm_layer(in_chan * expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    ### 定义前向函数(经典的ResNet结构)
    def forward(self,x : torch.Tensor) -> torch.Tensor :
        '''
        Args:
            x : 输入的Tensor
        Returns:
            Tensor : 输出的Tensor
        '''
        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)
        return out
### 开始ResNet的编写
class ResNet(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 num_classes : int = 1000,
                 zero_init_residual = False,
                 groups : int = 1,
                 width_per_group : int = 64,
                 replace_stride_with_dilation = None,
                 norm_layer = None
                 ) -> None:
        '''
        Args:
            block:  基本的ResNet模块,包含BasicBlock和BottleNeck,前者较浅,后者深一点,这里用BottleNeck
            layers : 表示每段的模块的数量
            num_classes : 数据集的类别
            zero_init_residual : 一种特殊的初始化 ? 
            groups : 卷积的组数
            width_per_group : 每个分组的宽度
            replace_stride_with_dilation : 是否采用dilation卷积来替换stride
            norm_layer : 归一化
        '''
        super(ResNet,self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes : int = 64
        self.dilation : int = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False,False,False]
        ###检查repalce_stride_with_dilation输入是否正确
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation 必须有3个参数,例如[False,False,False] Error {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        RGB_chan : int = 3
        ### 经过初始的卷积层
        self.conv1 = nn.Conv2d(in_channels= RGB_chan,
                               out_channels=self.inplane,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias=False
                                )
        self.bn1 = norm_layer(self.inplane)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        ### 开始resnet堆叠
        self.layer1 = self. _make_layers(block,64,layers[0])
        self.layer2 = self. _make_layers(block,128,layers[1],stride=2,
                                    replace_stride_with_dilation = replace_stride_with_dilation[0])
        self.layer3 = self. _make_layers(block,256,layers[2],stride=2,
                                    replace_stride_with_dilation = replace_stride_with_dilation[1])
        self.layer4 = self. _make_layers(block,512,layers[3],stride=2,
                                    replace_stride_with_dilation = replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion,num_classes)
        ### 完成网络搭建
        ### 权重初始化
        if m in self.modules():
            if isinstance(m,nn.Conv2d):
                ### 何恺明初始化
                nn.init.kaiming_normal_(m.weight,mode="fan_out",nonlinearity="relu")
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BottleNeck):
                    nn.init.constant_(m.bn3.weight,0)
        
    def _make_layers(self,
                    block,
                    planes,
                    blocks,
                    stride : int = 1,
                    dilate : bool = False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate :
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != block.expansion * planes:
            downsample = nn.Sequential(
                conv1x1(in_chan=self.inplanes,out_chan=planes * block.expansion,stride=stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1,blocks):
             layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
        return nn.Sequential(*layers)
    def _forward_impl(self,x : torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x
    
    def forward(self,x : torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)
    
    
def _resnet(block , layers , **kwargs):
    module = ResNet(block,layers,**kwargs)
    return module


def resnet50(**kwargs):
    return _resnet(BottleNeck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs): 
    return _resnet(BottleNeck, [3, 4, 23, 3], **kwargs)
    
         
