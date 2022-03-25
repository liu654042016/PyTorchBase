<!--
 * @Author: LIU KANG
 * @Date: 2022-03-13 23:47:19
 * @LastEditors: LIU KANG
 * @LastEditTime: 2022-03-25 17:34:15
 * @FilePath: \PyTorchBase\numpybase\readme.md
 * @Description: 
 * 
 * Copyright (c) 2022 by 用户/公司名, All Rights Reserved. 
-->
#numpy base practice
##参考网站
https://blog.csdn.net/cxmscb/article/details/54583415
###创建ndarry数组

####np.linalg.norm求范数
```
x_norm=np.linalg.norm(x, ord=None, axis=None, keepdims=False)
```
①x: 表示矩阵（也可以是一维）
②ord：范数类型
3.向量的范数
![图 1](../images/e7c22b4ad53f84846afc1fc7b63ee09b9469e88fd0ccd4c6b8179d14641a82c7.png)  
###计算方阵的逆np.linalg.inv(x)

##随机数
参考网站：https://blog.csdn.net/sinat_28576553/article/details/82926047
####生成随机整数np.random.randint()
```
temp3=np.random.randint(10,size=8)
temp4=np.random.randint(10,size=(2,4))
temp5=np.random.randint(5,10,size=(2,4))
```