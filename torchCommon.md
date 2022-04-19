<!--
 * @Author: LIU KANG
 * @Date: 2022-04-16 23:16:22
 * @LastEditors: LIU KANG
 * @LastEditTime: 2022-04-19 10:51:01
 * @FilePath: \PyTorchBase\torchCommon.md
 * @Description: liukang
 * 
 * Copyright (c) 2022 by 用户/公司名, All Rights Reserved. 
-->
#pytorch基本操作
官网链接：https://pytorch.org/docs/stable/index.html
##基本配置
1.导入包和版本查询
`import torch`
`import torch.nn as nn`
`import torchvision`
`print(torch.__version__)`
`print(torch.version.cuda)`
`print(torch.backends.cudnn.version())`
`print(torch.cuda.get_device_name(0))`
2.可复现性
在硬件设备（CPU、GPU）不同时，完全的可复现性无法保证，即使随机种子相同。但是，在同一个设备上，应该保证可复现性。具体做法是，在程序开始的时候固定torch的随机种子，同时也把numpy的随机种子固定。
```
np.random.seed(0)
np.manual_seed(0)
torch.cuda.manual_seed_all(0)


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
```
3.显卡设置
`device = torch.device('cuda' if torch.cuda_isavaliable() else 'cpu')`
如果指定多张显卡，比如0，1号显卡
`os.environ=['CUDA_VISIBLE_DEVICES']=0,1`
在命令行代码设置显卡
`CUDA_VISIBLE_DEVICES=0, 1 python train.py`
清除显存
`torch.cuda.empty_cache()`
也可以在命令行重置gpu的命令
`nvidia-smi --gpu-reset -i [gpu_id]`
##张量处理
1.张量基本信息
```
tensor = torch.randn(3, 4, 5)
print(tensor.type())
print(tensor.size())
print(tensor.dim())
```
2.数据类型转换
```
#设置默认类型，pytorch中的floattensor远远快于doubletensor
torch.set_default_tensor_type(torch.FloatTensor)

#类型转换
tensor = tensor.cuda()
tensor = tensor.cpu()
tensor = tensor.float()
tensor = tensor.long()
```
3.torch.Tensor与np.ndarry转换
```
ndarray = tensor.cpu().numpy()
tensor = torch.from_numpy(ndarry).float()
tensor = torch.from_numpy(ndarry.copy()).float()
```