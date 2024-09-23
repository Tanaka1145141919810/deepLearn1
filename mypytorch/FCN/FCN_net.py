#### 完成了back_bone现在完成完整的FCN
import torch
from torch import nn
from torch.nn import functional as F 

from typing import Dict, Mapping
from collections import OrderedDict

from .FCN_Backbone import resnet50
from .FCN_Backbone import resnet101

### 提取backbone网络的中间层信息,把不要的网络层删除
class IntermediateLayerGetter(nn.ModuleDict):
    def __init__(self, model : nn.Module , return_layers : Dict[str,str]) -> None:
        '''
        Args :
            model : backbone网络
            return_layers :
        '''
        if not set(return_layers).issubset([name for name,_ in model.named_children()]):
            raise ValueError("没有这种层结构")
        orig_return_layers = return_layers
        return_layers = {str(k) : str(v) for k , v in return_layers.item()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break
        super(IntermediateLayerGetter,self).__init__(layers)
        self.return_layers = orig_return_layers
    ### 得到需要返回层的输出
    def forward(self,x : torch.Tensor)-> Dict[str,torch.Tensor]:
        out = OrderedDict()
        for name,module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


### 下面是FCNHead的函数,根据FCN网络中所述，在经过backbone网络之后便来到了FCNHead
class FCNHead(nn.Sequential):
    ### 就一个全连接
    def __init__(self,in_chan : int,out_chan : int)->None:
        inter_chan = in_chan // 4
        layers = [
            nn.Conv2d(in_channels=in_chan , out_channels= inter_chan , kernel_size= 3,
                      padding= 1 ,bias= False),
            nn.BatchNorm2d(num_features= inter_chan),
            nn.ReLU(),
            nn.Dropout(p = 0.1),
            nn.Conv2d(in_channels=inter_chan,out_channels=out_chan,kernel_size=1),
        ]
        super(FCNHead,self).__init__(*layers)

class FCN(nn.Module):
    def __init__(self, backbone : nn.Module , classifier : nn.Sequential ,aux_classifier : nn.Sequential)->None:
        '''
        Args:
            backbone : 这里是resnet
            classifier : 这里是FCNHead
            aux_classifier : 辅助分类器，加快训练用的
        ''' 
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        input_shape = x.shape[-2:]
        ### 经过主干网络
        features = self.backbone(x)

        result = OrderedDict()
        x = features["out"]
        x = self.classifier(x)
        # 原论文中虽然使用的是ConvTranspose2d，但权重是冻结的，所以就是一个bilinear插值 
        ### ConvTranspose2d冻结权重==双线性插值 ？
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        result["out"] = x

        if self.aux_classifier is not None:
            x = features["aux"]
            x = self.aux_classifier(x)
            # 原论文中虽然使用的是ConvTranspose2d，但权重是冻结的，所以就是一个bilinear插值
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            result["aux"] = x

        return result
def fcn_resnet50(aux, num_classes=21, pretrain_backbone=False):
    # 'resnet50_imagenet': 'https://download.pytorch.org/models/resnet50-0676ba61.pth'
    # 'fcn_resnet50_coco': 'https://download.pytorch.org/models/fcn_resnet50_coco-1167a1af.pth'
    backbone = resnet50(replace_stride_with_dilation=[False, True, True])

    if pretrain_backbone:
        # 载入resnet50 backbone预训练权重
        backbone.load_state_dict(torch.load("resnet50.pth", map_location='cpu'))

    out_inplanes = 2048
    aux_inplanes = 1024

    return_layers = {'layer4': 'out'}
    if aux:
        return_layers['layer3'] = 'aux'
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = None
    # why using aux: https://github.com/pytorch/vision/issues/4292
    ### 总而言之就是来改善训练的效果
    if aux:
        aux_classifier = FCNHead(aux_inplanes, num_classes)

    classifier = FCNHead(out_inplanes, num_classes)

    model = FCN(backbone, classifier, aux_classifier)

    return model


def fcn_resnet101(aux, num_classes=21, pretrain_backbone=False):
    # 'resnet101_imagenet': 'https://download.pytorch.org/models/resnet101-63fe2227.pth'
    # 'fcn_resnet101_coco': 'https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth'
    backbone = resnet101(replace_stride_with_dilation=[False, True, True])

    if pretrain_backbone:
        # 载入resnet101 backbone预训练权重
        backbone.load_state_dict(torch.load("resnet101.pth", map_location='cpu'))

    out_inplanes = 2048
    aux_inplanes = 1024

    return_layers = {'layer4': 'out'}
    if aux:
        return_layers['layer3'] = 'aux'
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = None
    # why using aux: https://github.com/pytorch/vision/issues/4292
    if aux:
        aux_classifier = FCNHead(aux_inplanes, num_classes)

    classifier = FCNHead(out_inplanes, num_classes)

    model = FCN(backbone, classifier, aux_classifier)

    return model

        
        