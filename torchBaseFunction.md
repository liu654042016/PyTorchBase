<!--
 * @Author: LIU KANG
 * @Date: 2022-03-04 20:14:49
 * @LastEditors: LIU KANG
 * @LastEditTime: 2022-04-25 10:55:32
 * @FilePath: \PyTorchBase\torchBaseFunction.md
 * @Description: 
 * 
 * Copyright (c) 2022 by 用户/公司名，All Rights Reserved. 
-->
# pytorch
##tensor 操作
https://alyssaasa.github.io/posts/309/
###查看维度
`a=torch.randn(3,4)`
`a.size`
###将 tensor 进行 reshape: torch.view
定义：将 tensor 中的元素，按照顺序逐个选取，凑成 (shape) 的大小。把原先 tensor 中的数据按照行优先的顺序排成一个一维的数据（这里应该是因为要求地址是连续存储的），然后按照参数组合成其他维度的 tensor。
```
a= torch.randn(3,4)
a = a.view(2,6)
```
###tensor 交换维度
定义：将 tensor 的维度换位。
```
a = torch.randn(2, 3, 4) # torch.Size([2, 3, 4])
b = a.permute(2, 0, 1)   # torch.Size([4, 2, 3])

```
###对 tensor 的维度进行压缩
定义：对数据的维度进行压缩，去掉维数为 1 的的维度
```
a=torch.randn(1, 2, 1, 3, 4)
x = a.squeeze() # 去掉所有为 1 的维度：torch.Size([2, 3, 4])
y = a.squeeze(dim=2) # 去掉维度为 1 的 dim 维度：torch.Size([1, 2, 3, 4])
```
###tensor 维度扩张：tensor.expand & torch.repeat
定义：对 tensor 的维度进行扩张。如果某个维度参数是 -1，代表这个维度不改变。tensor 可以被 expand 到更大的维度，新的维度的只是前面的值的重复。新的维度参数不能为 -1。
expand 一个 tensor 并不会分配新的内存，而只是生成一个已存在的 tensor 的 view。
返回当前张量在某维扩展更大后的张量。扩展（expand）张量不会分配新的内存，只是在存在的张量上创建一个新的视图（view），一个大小（size）等于 1 的维度扩展到更大的尺寸
```
x = torch.tensor([[1], [2], [3]])
x.size() 
#torch.Size([3, 1])
x.expand(3, 4)
# tensor([[ 1,  1,  1,  1],
#         [ 2,  2,  2,  2],
#         [ 3,  3,  3,  3]])
x.expand(-1, 4)   # -1 means not changing the size of that dimension
# tensor([[ 1,  1,  1,  1],
#         [ 2,  2,  2,  2],
#         [ 3,  3,  3,  3]])
```
使用 expand 可以增加新的一个维度，但是只能在第 0 维增加一个维度，增加的维度大小可以大于 1，比如原始 t = tensor (X,Y)，可以 t.expand (Z,X,Y)，不能在其他维度上增加；expand 拓展某个已经存在的维度的时候，如果原始 t = tensor (X,Y)，则必须要求 X 或者 Y 中至少有 1 个维度为 1，且只能 expand 维度为 1 的那一维。

torch.repeat()
沿着特定的维度重复这个张量，和 expand() 不同的是，这个函数拷贝张量的数据。
```
x = torch.tensor([1, 2, 3])
x = x.repeat(3,2)
tensor([[1, 2, 3, 1, 2, 3],
        [1, 2, 3, 1, 2, 3],
        [1, 2, 3, 1, 2, 3]])

x2 = torch.randn(2,3,4)
x2.repeat(2, 1, 3).shape
torch.Tensor([4, 3, 12])
```

###将 tensor 的指定维度合并为一个维度：torch.flatten
`torch.flatten(input, start_dim=0, end_dim=-1)`
start_dim: flatten 的起始维度。end_dim: flatten 的结束维度。
```
a=torch.randn(2, 3, 4)
x = torch.flatten(a, start_dim=1) # torch.Size([2, 12])
y = torch.flatten(a, start_dim=0, end_dim=1) # torch.Size([6, 4])
```
###将 tensor 进行分割：torch.split
定义：根据长度去拆分 tensor
```
a=torch.randn(3,4)
a.split([1,2],dim=0)
#把维度 0 按照长度 [1,2] 拆分，形成 2 个 tensor，shape（1，4）和 shape（2，4）
a.split([2,2],dim=1)
#把维度 1 按照长度 [2,2] 拆分，形成 2 个 tensor，shape（3，2）和 shape（3，2）
```
###将 tensor 均等分割：torch.chunk
定义：均等分的 split，但是当维度长度不能被等分份数整除时，虽然不会报错，但可能结果与预期的不一样，建议只在可以被整除的情况下运用。
```
a=torch.randn(4,6)

a.chunk(2,dim=0)
#返回一个 shape（2，6）的 tensor
a.chunk(2,dim=1)
#返回一个 shape（4，3）的 tensor
```



###将两个 tensor 拼接起来：torch.cat
定义：把 2 个 tensor 按照特定的维度连接起来。
要求：除被拼接的维度外，其他维度必须相同
```
a = torch.randn(3,4)
b = torch.randn(2,4)
torch.cat([a, b], dim=0)
#返回一个 shape（5，4）的 tensor
#把 a 和 b 拼接成一个 shape（5，4）的 tensor，
```

###将两个 tesnor 堆叠起来：torch.stack
定义：增加一个新的维度，来表示拼接后的 2 个 tensor。
直观些理解的话，咱们不妨把一个 2 维的 tensor 理解成一张长方形的纸张，cat 相当于是把两张纸缝合在一起，形成一张更大的纸，而 stack 相当于是把两张纸上下堆叠在一起
要求：两个 tensor 拼接前的形状完全一致
```
a=torch.randn(3,4)
b=torch.randn(3,4)

c=torch.stack([a,b],dim=0)
#返回一个 shape(2,3,4) 的 tensor，新增的维度 2 分别指向 a 和 b
d=torch.stack([a,b],dim=1)
#返回一个 shape（3,2,4）的 tensor，新增的维度 2 分别指向相应的 a 的第 i 行和 b 的第 i 行
e = torch.stack([a, b], dim=-1)
#返回一个 shape(3,4,2) 的 tensor
```
###troch.detach() 和 torch.detach_() 区别
参考网站：https://blog.csdn.net/qq_27825451/article/details/95498211
torch.detach() 返回一个新的 tensor，从当前计算图中分离下来的，但是仍指向原变量的存放位置，不同之处只是 requires_grad 为 false，得到的这个 tensor 永远不需要计算其梯度，不具有 grad。即使之后重新将它的 requires_grad 置为 true，它也不会具有梯度 grad。这样我们就会继续使用这个新的 tensor 进行计算，后面当我们进行反向传播时，到该调用 detach() 的 tensor 就会停止，不能再继续向前进行传播