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
        model_ft = models.Resnet152(pretrained = use_pretrained)
        set_parameter_requires_grad(model_ft, feature_etract)
        num_ftrs = model_ft.fc.in_features
        model_fit.fc = nn.Sequential(nn.Linear(num_ftrs,    num_class),
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

    data_dir = ''
    train_dir = data_dir + '/train'
    valid_dir = data_dir + './valid'

    #读取对应的花名
    with open("cat_to_name.json") as f:
        cat_to_name = json.load(f)

    data_transforms = {
        'train' : transforms.Compose([transforms.RandomRotation(45), #随机旋转 -45到45度之间随机
                                      transforms.CenterCrop(224), #从中心开始裁剪，只得一张图片
                                      transforms.RandomHorizontalFlip(p=0.5), #随机水平翻转，概率为0.5
                                      transforms.RandomVerticalFlip(p=0.5),#随机垂直翻转
                                      transforms.ColorJitter(brightness=0.2, constrast=0.1, saturation=0.1, hue=0.1),
                                      #参数1为亮度， 参数2为对比度， 参数3为饱和度， 参数4为色相
                                      transforms.RandomGrayscale(p=0.025),#概率转换成灰度率， 3通道R=G=B
                                      transforms.Tensor(),
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
    