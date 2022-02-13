import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,  transforms
from torch.utils.data import DataLoader

class numberCNN(nn.Module):
    def __init__(self) -> None:
        super(numberCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels= 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.out = nn.Linear(32*7*7, 10)          

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        return self.out(x) 
    
def accuarcy(prediction, labels):
   # torch.max(a,0)返回每一列中最大值的那个元素，且返回索引（返回最大元素在这一列的行索引）。返回的最大值和索引各是一个tensor，
   # 一起构成元组(Tensor, LongTensor)
    pred = torch.max(prediction.data, 1)[1]
    # view_as，形状变为相同，这里的rights依然是一个tensor
    rights = pred.eq(labels.data.view_as(pred)).sum()
    print(rights)
    return rights, len(labels)


def statr():
    input_size = 28
    num_class = 10
    num_epoch = 3
    batch_size = 4

    train_dataset = datasets.MNIST(root = 'D:\study\PyTorch\PyTorchBase\CNN\data',
                                    train = True,
                                    transform = transforms.ToTensor(),
                                    download = False)
    test_dataset = datasets.MNIST(root = 'D:\study\PyTorch\PyTorchBase\CNN\data',
                                    train = False,
                                    transform = transforms.ToTensor(),
                                    download = False)

    #构建batch数据
    train_loader = DataLoader(dataset = train_dataset,
                                batch_size = batch_size,
                                shuffle=True)
    test_loader = DataLoader(dataset = test_dataset,
                                batch_size=batch_size,
                                shuffle=True)

    myModel = numberCNN()

    #损失函数
    criterion = nn.CrossEntropyLoss()

    #优化器
    optimizer  = optim.Adam(myModel.parameters(), lr=0.001)

    for epoch in range(num_epoch):
        train_right = []
        for idx, data in enumerate(train_loader):
            inputs, labels = data
            myModel.train()
            outputs = myModel(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            right = accuarcy(outputs, labels)
            #每个批次 将正确个数记录和总个数
            train_right.append(right)

            if idx % 100 == 0:
                myModel.eval()
                val_right = []
                for valData in test_loader:
                    valInputs, valLabels = valData
                    outputs = myModel(valInputs)
                    right = accuarcy(outputs, valLabels)
                    val_right.append(right)

                #准确率计算
                train_r = (sum([tup[0] for tup in train_right]), sum([tup[1] for tup in train_right]))
                val_r = (sum([tup[0] for tup in val_right]), sum([tup[1] for tup in val_right]))

                print("当前epochL", epoch, "当前训练量： ", (idx+1)*batch_size)
                print("训练正确个数：" , train_r[0].numpy(), "训练总个数" , train_r[1], "训练准确率： ",  train_r[0].numpy() / train_r[1])
                print("测试正确个数", val_r[0].numpy(), "测试总个数：", val_r[1], "准确率", val_r[0].numpy()/val_r[1])
    

if __name__=='__main__':
    print("aaa")
    statr()
    print("bbbb")