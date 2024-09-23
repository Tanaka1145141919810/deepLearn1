import torch
import torch.nn as nn

####test一个全连接网络 检查torch安装是否正确，如果安装正确可忽略

class FNet(nn.Module):
    def __init__(self,in_chan : int , hidden_chan : int , out_chan : int):
        super().__init__()
        self.in_chan = in_chan
        self.hidden_chan = hidden_chan
        self.out_chan = out_chan
        self.fc1 = nn.Linear(in_features= self.in_chan,
                             out_features = self.hidden_chan)
        self.ln1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features= self.hidden_chan,\
                             out_features= self.out_chan)
        self.ln2 = nn.Softmax()

    def forward(self,x : torch.Tensor)->torch.Tensor:
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.fc2(x)
        x = self.ln2(x)
        return x
    

def test_FN1():
    x : torch.Tensor = torch.randn(1024)
    fcNet = FNet(in_chan= 1024,hidden_chan= 4096,out_chan=10)
    x = fcNet(x)
    

if __name__ == "__main__":
    test_FN1()
