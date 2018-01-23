'''
Created on 2018年1月18日

@author: user
'''
from sklearn.model_selection import train_test_split
import numpy as np
import sklearn


data = np.loadtxt('validation.txt',delimiter=',') 
label = data[:,0]
value = data[:,1::]
for i in range(label.shape[0]) :
    if label[i]<0:
        label[i] = 0

train_value, test_value, train_label, test_label = train_test_split(value, label, test_size=0.33)
#为训练,测试集添加标签列
train = np.c_[train_value,train_label].astype(float)
test = np.c_[test_value,test_label].astype(float)
# print (train_value.shape)
# print (test_value.shape)
# print (train_label.shape)
# print (test_label.shape)
# print(train.shape)
# print (test.shape)
#写入txt,fmt指定数据精度(默认的科学记数法很浪费空间),
np.savetxt("train.txt",train,fmt='%f',delimiter=',')
np.savetxt("test.txt",test,fmt='%f',delimiter=',')
print ('secess')

