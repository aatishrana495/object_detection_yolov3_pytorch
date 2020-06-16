# -*- coding: utf-8 -*-
"""
Created on Sat May 16 00:28:40 2020

@author: devid
"""

import os
import numpy as np
from sklearn import model_selection

path= os.getcwd()
file_list=os.listdir(os.path.join(path,"images"))
#train_path=os.path.join(path,'train.txt')

os.remove(os.path.join(path, "train.txt"))
os.remove(os.path.join(path, "valid.txt"))

os.mknod("train.txt")
os.mknod("valid.txt")

# making training and validation set
train, test= model_selection.train_test_split(file_list,test_size=0.2, random_state=0)

with open('train.txt', 'w') as f:
	for t in train:
		file_path = os.path.join(path,"images", t)
		f.write(file_path + "\n")
	f.close()

with open('valid.txt', 'w') as f:
	for v in test:
		file_path = os.path.join(path,"images", v)
		f.write(file_path + "\n")
	f.close()




# a=["2.0","2.7","3.6"]
# for x in a:
#     n=int(float(x))
#     print(n)
# print(a[(1.0)int])
# print(a[1])

# path= os.getcwd()
# file_list=os.listdir(os.path.join(path,"yolo_labels2"))
# #train_path=os.path.join(path,'train.txt')
# bad=[]
# for f in file_list:
#     file_path=os.path.join(path,"yolo_labels2",f)
#     file=open(file_path,"r")
#     for x in file:
#         print(f)
#     label=x.split(" ")
#     if(label[len(label)-1]==''):
#         bad.append(f)
#     file.close()
# print(bad)
# print(len(bad))






    
