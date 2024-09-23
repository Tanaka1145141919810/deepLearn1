#### 官方的Guide拆分成函数形式，并加上类型标注(尝试使用类型标注功能)

#### import nessary lib
#### 导入必要的库
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from typing import Tuple

#### 导入数据库并处理
def getDatasetFashionMINIST(root : str):
    trainset = datasets.FashionMNIST(root = root,
                                     train= True,
                                     transform=ToTensor(),
                                     download=True)
    testset = datasets.FashionMNIST(root=root,
                                    train= False,
                                    transform=ToTensor(),
                                    download=True)
    return trainset , testset

#### 载入数据到Loader
def loadDataset(batch_size : int,trainset,testset) -> Tuple[type,type] :
    trainDataLoader = DataLoader(dataset=trainset,batch_size=batch_size,shuffle=True)
    testDataLoader = DataLoader(dataset=testset,batch_size=batch_size,shuffle=False)
    return trainDataLoader , testDataLoader
#### 展示加载数据的shape
def showDataLoaderShape(DataLoader : type):
    for X , y in DataLoader :
        print("X shape is {}".format(X.shape))
        print("y shape is {}".format(y.shape))
        return 
#### 对DataLoader解析可以发现XShape = [batch_size(64) ,in_chan(RGB,1),height,width ],yShape = [batch_size(64)]

#### 设置训练设备
def setDevice() -> str :
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    return device

#### 设置训练网络
class FC(nn.Module):
    def __init__(self,in_chan : int,hidden_chan : int ,out_chan : int):
        super(FC,self).__init__()
        self.in_chan = in_chan
        self.hidden_chan = hidden_chan
        self.out_chan = out_chan
        ### Flatten层 不管Batch_Size层 进入的[64,1,28,28]为[64,1*28*28]
        self.flatten = nn.Flatten() 
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=self.in_chan,out_features=hidden_chan),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_chan,out_features=self.out_chan),
            nn.Softmax(dim=-1),
        )
        return
    
    ## 前向传播函数
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.fc1(x)
        return x
### 设置主函数
def main():
    device = setDevice()
    net = FC(in_chan=28*28,hidden_chan=512,out_chan=10).to(device=device)
    ### 设置损失函数和优化器
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(),lr= 1e-3)
    ### 加载数据集
    trainset , testset = getDatasetFashionMINIST("dataset")
    ### Load数据集
    trainLoader , testLoader = loadDataset(batch_size = 64,trainset=trainset,testset=testset)
    ### 开始训练和测试
    epoches : int = 10
    for epoch in range(epoches):
        with open("runLog.txt","a") as file:
            file.write("This is {} train\n".format(epoch))
            file.close()
        train(dataloader=trainLoader,net=net,loss_func=loss,optimizer=optimizer,device=device)
        with open("runLog.txt","a") as file:
            file.write("This is {} test\n".format(epoch))
            file.close()
        test(dataLoader=testLoader,net=net,loss_func=loss,device=device)

    saveModule(net=net,path="ModuleSave")
    print("Finish ......................")



    ### 输入日志
    with open("runLog.txt","a") as file: 
        file.write("This a Pytorch QuickStart\n")
        file.write("The dataSet is FashionMINIST\n")
        file.write("device is {}\n".format(device))
        file.write("net parameter is {}\n".format(net.parameters))
        file.write("loss func is {}\n".format(loss))
        file.write("optimizer is {}\n".format(optimizer))
        file.close()
### 设置训练函数
def train(dataloader : type,net,loss_func,optimizer,device : str):
    size = len(dataloader.dataset)
    for batch ,(X,y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)

        pred = net(X)
        loss = loss_func(pred,y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            with open("runLog.txt","a") as file:
                file.write("loss is {} {}/{}".format(loss.item(),(batch + 1)/(len(X)),size))
                file.close()

    return

### 定义测试函数
def test(dataLoader,net,loss_func,device : str):
    size = len(dataLoader.dataset)
    num_batches = len(dataLoader)
    net.eval()
    test_loss : float = 0.0
    correct : float = 0.0
    with torch.no_grad():
        for X,y in dataLoader:
            X = X.to(device)
            y = y.to(device)
            pred = net(X)
            test_loss += loss_func(pred,y).item()
            correct += (pred.argmax(1) == y).type(torch.float32).sum().item()
    test_loss /= num_batches
    correct /= size
    with open("runLog.txt","a") as file:
        file.write("Test Error :\n")
        file.write("Accuracy : {}%\n".format(correct * 100))
        file.write("Avg loss : {}\n".format(test_loss))
        file.close()
def saveModule(net,path : str):
    torch.save(net.state_dict(),path+".pth")
    with open("runLog.txt","a") as file:
        file.write("Module savePath is {}\n".format(path))
        file.close()
    

if __name__ == "__main__":
    #trainset, testset =getDatasetFashionMINIST("dataset")
    #trainDataLoader, testDataLoader =loadDataset(batch_size= 64,trainset=trainset,testset=testset)
    #showDataLoaderShape(DataLoader=testDataLoader)
    #setDevice()
    main()