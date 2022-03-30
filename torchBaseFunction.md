<!--
 * @Author: LIU KANG
 * @Date: 2022-03-04 20:14:49
 * @LastEditors: LIU KANG
 * @LastEditTime: 2022-03-30 17:19:48
 * @FilePath: \PyTorchBase\torchBaseFunction.md
 * @Description: 
 * 
 * Copyright (c) 2022 by 用户/公司名, All Rights Reserved. 
-->
# pytorch
##tensor操作
https://alyssaasa.github.io/posts/309/
###查看维度
`a=torch.randn(3,4)`
`a.size`
###将tensor进行reshape: torch.view
定义：将 tensor 中的元素，按照顺序逐个选取，凑成 (shape) 的大小。把原先 tensor 中的数据按照行优先的顺序排成一个一维的数据（这里应该是因为要求地址是连续存储的），然后按照参数组合成其他维度的 tensor。
```
a= torch.randn(3,4)
a = a.view(2,6)
```
###tensor交换维度



###将两个tensor拼接起来：torch.cat
定义：把 2 个 tensor 按照特定的维度连接起来。
要求：除被拼接的维度外，其他维度必须相同
```
a = torch.randn(3,4)
b = torch.randn(2,4)
torch.cat([a, b], dim=0)
#返回一个shape（5，4）的tensor
#把a和b拼接成一个shape（5，4）的tensor，
```

###将两个tesnor堆叠起来：torch.stack
定义：增加一个新的维度，来表示拼接后的2个tensor。
直观些理解的话，咱们不妨把一个 2 维的 tensor 理解成一张长方形的纸张，cat 相当于是把两张纸缝合在一起，形成一张更大的纸，而 stack 相当于是把两张纸上下堆叠在一起
要求：两个 tensor 拼接前的形状完全一致
```
a=torch.randn(3,4)
b=torch.randn(3,4)

c=torch.stack([a,b],dim=0)
#返回一个shape(2,3,4)的tensor,新增的维度2分别指向a和b
d=torch.stack([a,b],dim=1)
#返回一个shape（3,2,4）的tensor，新增的维度2分别指向相应的a的第i行和b的第i行
e = torch.stack([a, b], dim=-1)
#返回一个shape(3,4,2)的tensor
```
###troch.detach()和torch.detach_()区别
参考网站：https://blog.csdn.net/qq_27825451/article/details/95498211
torch.detach()返回一个新的tensor，从当前计算图中分离下来的，但是仍指向原变量的存放位置,不同之处只是requires_grad为false，得到的这个tensor永远不需要计算其梯度，不具有grad。即使之后重新将它的requires_grad置为true,它也不会具有梯度grad。这样我们就会继续使用这个新的tensor进行计算，后面当我们进行反向传播时，到该调用detach()的tensor就会停止，不能再继续向前进行传播