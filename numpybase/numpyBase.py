'''
Author: LIU KANG
Date: 2022-03-14 15:22:44
LastEditors: LIU KANG
LastEditTime: 2022-03-15 15:02:21
FilePath: \PyTorchBase\numpybase\numpyBase.py
Description: 

Copyright (c) 2022 by 用户/公司名, All Rights Reserved. 
'''
import numpy as np


#init
zeros = np.zeros([3, 5])
#print(zeros)

ones = np.ones([3,4])
#print(ones)

empty = np.empty([3,4]);
# print(empty)
# print(empty.size)

a = np.arange(0, 9)
a = a.reshape(3, 3)
# print(a)
# print(a[:, 2])

b = np.arange(0,9)
print(b)
b = np.power(np.array(b),2)
print(b)
b = np.square(b)
print(b)
