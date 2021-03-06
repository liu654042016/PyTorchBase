import copy
import json
import time

import torch
from torch import nn 
import torch.optim as optim
import torchvision
import os 
from torchvision import transforms, models, datasets
import numpy as np
import matplotlib.pyplot as plt
import ssl

#是否继续训练模型的参数
def set_parameter_requires_grad(model, feature_etracting):
    if feature_etracting:
        for param in model.parameters():
            param.requires_grad = False

def get_device()->torch.device:
    train_on_gpu = torch.cuda.is_available()

    if not train_on_gpu:
        print('cuda is not available , training on cpu...')
    else :
        print('cuda is available , training on gpu ...')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device

#选择迁移哪个模型
def initialize_model(model_name, num_class, feature_etract, use_pretrained=True):
    #选择合适的模型 不同的模型初始化方法有区别
    model_fit = None
    input_size = 0

    if model_name == "resnet":
        """
        resnet152
        """
        #pretrained表示是否需要下载
        model_ft = models.resnet152(pretrained = use_pretrained)
        print(model_ft)
        set_parameter_requires_grad(model_ft, feature_etract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(nn.Linear(num_ftrs,  num_class),
                                    nn.LogSoftmax(dim=1))
        input_size = 224
    
    elif model_name == "alexnet":
        model_ft = models.alexnet(pretrained = use_pretrained)
        set_parameter_requires_grad(model_ft, feature_etract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_class)
        input_size = 224
    
    elif model_name == "vgg":
        model_ft = models.vgg16(pretained = use_pretrained)
        set_parameter_requires_grad(model_ft, feature_etract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_class)
        input_size = 224

    elif model_name == "squeezenet":
        model_ft = models.squeezenet1_0(ptetrained = use_pretrained)
        set_parameter_requires_grad(model_ft, feature_etract)
        model_ft.classifier[1] = nn.Conv2d(512, num_class, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_class
        input_size = 224

    elif model_name == "inception":
        model_ft = models.inception_v3(pretrained = use_pretrained)
        set_parameter_requires_grad(model_ft, feature_etract)

        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_class)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_class)
        input_size = 299
    else:
        print("invalid model name exiting...")
    
    return model_ft, input_size


def wz_model_train(model, dataloaders, criterion, optimizer, scheduler, filename: str, device : torch.device, num_epochs=10, is_inception=False):
    start_time = time.time()
    best_acc = 0
    best_model_weights = copy.deepcopy(model.state_dict())
    model.to(device)

    #保存损失和准确率数据
    val_acc_history = []
    train_acc_histtory = []
    train_loss = []
    valid_loss = []

    #记录每个epoch的学习率
    LRs = [optimizer.param_groups[0]['lr']]
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('-'*10)

        #训练和验证
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                #清零
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    #inception会有辅助输出，损失函数的一个线性模型
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        print(1)
                        outputs =  model(inputs)
                        loss = criterion(outputs, labels)
                        print(2)
                    _, preds = torch.max(outputs, 1)
                    #训练阶段更新权重
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                #计算损失
                #loss计算默认都是取mean,计算批量的loss时，要乘以loss的数量
                #所以这里计算的是一个epoch里所有样本的loss和正确数量
                running_loss += loss.item()*inputs.size(0)
                running_corrects += torch.sum(preds==labels.data)
            #完整一次的Loss平均值和准确率
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() /len(dataloaders[phase].dataset)
            #一个epoch里train和valid分别花的时间和 loss和准确度
            time_elapsed = time.time() - start_time
            print('Time elapsed{: .f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed%60))
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            #得到最好的模型
            if phase=='valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())
                state = {
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(state, filename)
            if phase=='valid':
                val_acc_history.append(epoch_acc)
                valid_loss.append(epoch_loss)
                scheduler.step()
            if phase == 'train':
                train_acc_histtory.append(epoch_acc)
                train_loss.append(epoch_loss)

        print('optimizer learning rate: {:.0f}m{:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('best val acc: {:4f}'.format(best_acc))
        #训练完后用最好的一次当作模型的最终结果
        model.load_state_dict(best_model_weights)
        return model, val_acc_history, train_acc_histtory, valid_loss, train_loss, LRs




def flower_start():
    model_ft, input_size = initialize_model("resnet", 102, feature_etract=True, use_pretrained=True)
    device = get_device()
    model_ft = model_ft.to(device)
    #带name的param
    params = model_ft.named_parameters()
    print("params need to learn")
    params_need_updata = []
    for para_name, param in params:
        if param.requires_grad:
            params_need_updata.append(param)
            print(para_name)

    data_dir = 'D:\study\PyTorch\PyTorchBase\CNN\\flower_data\\'
  #  train_dir = data_dir + '\train'
  #  valid_dir = data_dir + '\valid'

    #读取对应的花名
    with open("D:\study\PyTorch\PyTorchBase\CNN\cat_to_name.json") as f:
        cat_to_name = json.load(f)

    data_transforms = {
        'train' : transforms.Compose([transforms.RandomRotation(45), #随机旋转 -45到45度之间随机
                                      transforms.CenterCrop(224), #从中心开始裁剪，只得一张图片
                                      transforms.RandomHorizontalFlip(p=0.5), #随机水平翻转，概率为0.5
                                      transforms.RandomVerticalFlip(p=0.5),#随机垂直翻转
                                      transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
                                      #参数1为亮度， 参数2为对比度， 参数3为饱和度， 参数4为色相
                                      transforms.RandomGrayscale(p=0.025),#概率转换成灰度率， 3通道R=G=B
                                      transforms.ToTensor(),
                                      #迁移学习，用别人的均值和标准差
                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])#jun
                                    ]),
        'valid' : transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     # 预处理必须和训练集一致
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                     ]),
    }
    batch_size = 8
    #train 和 vaild的图片做transform之后用字典保存

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid']}
    #批量处理，这里都是tensor格式
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size = batch_size, shuffle = True) for x in ['train', 'valid']}

    dataset_sizes = {x: len(image_datasets[x] )for x in ['train', 'valid']}    
    print(dataset_sizes)

    class_names = image_datasets['train'].classes


    #优化器设置
    optimizer_ft = optim.Adam(params_need_updata, lr=1e-2)
    #学习率衰减
    scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size = 7, gamma=0.1) # 学习率每7个epoch衰减成原来的1/10
    # 最后一层已经LogSoftmax()了，所以不能nn.CrossEntropyLoss()来计算了，nn.CrossEntropyLoss()相当于logSoftmax()和nn.NLLLoss()整合
    criterion = nn.NLLLoss()
    filename = "wa.pth"

    model_ft, val_acc_history, train_acc_history, valid_losses, train_losses, LRs = wz_model_train(model_ft, 
                                                                                                   dataloaders,
                                                                                                   criterion,
                                                                                                   optimizer_ft,
                                                                                                   scheduler,
                                                                                                   filename,
                                                                                                   device )

    for param  in model_ft.parameters():
        param.requires_grad = True

    #再继续训练所有的参数，学习率调小一点
    optimizer = optim.Adam(params_need_updata, lr = 1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 7, gamma = 0.1)

    #损失函数
    criterion = nn.NLLLoss()
    checkpoint = torch.load(filename)
    best_acc = checkpoint['best_acc']
    model_ft.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    model_ft, val_acc_history, train_acc_history, valid_losses, train_losses, LRs = wz_model_train(model_ft,
                                                                                                  dataloaders,
                                                                                                  criterion,
                                                                                                  optimizer,
                                                                                                  scheduler,
                                                                                                  filename,
                                                                                                  device,
                                                                                                  num_epochs=5)



if __name__=='__main__':
    ssl._create_default_https_context = ssl._create_unverified_context
    flower_start()