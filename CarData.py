###CarData.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
#from onehot import onehot
 
#指定数据转换的格式：变成张量、归一化
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
#采用独热编码，为矩阵增加一个维度，1表示背景，0表示车
def onehot(data, n):
    #初始化一个全零矩阵
    buf = np.zeros(data.shape + (n, ))
    #多维数组转化为一维数组
    nmsk = np.arange(data.size)*n + data.ravel()
    #这样的结果就是把(a,b)的label投射到(a,b)*2的长度中
    buf.ravel()[nmsk-1] = 1
    #返回
    return buf
#读取车辆图片，制作数据集
class CarDataset(Dataset):
    #创建类时，自动调用该方法，指定数据转换格式
    def __init__(self, transform=None):
        self.transform = transform
    #获得./train文件夹下的文件数 
    def __len__(self):
        return len(os.listdir('train'))
    #获得指定索引的图片和标签
    def __getitem__(self, idx):
        #获取文件路径
        tempfilename = os.listdir('train')[idx]
        #获得文件名和后缀
        (filename,extension) = os.path.splitext(tempfilename)
        #根据文件名读取./train中的图片
        imgA = cv2.imread('train/'+filename+'.jpg')
        #将图片分辨率缩小10倍
        imgA = cv2.resize(imgA, (192, 128))
        #根据文件名读取./train_masks中的标签，读取格式为灰度图
        imgB = Image.open('train_masks/'+filename+'_mask.gif').convert("L")        
        #灰度图转化为矩阵
        imgB = np.asarray(imgB)
        #图片分辨率缩小10倍
        imgB = cv2.resize(imgB, (192, 128))
        #归一化，1表示背景，0表示车
        imgB = imgB/255
        #设置格式为uint8
        imgB = imgB.astype('uint8')
        #用独热编码增加一个维度，因为此代码是二分类问题，即分割出手提包和背景两样就行，因此这里参数是2
        imgB = onehot(imgB, 2)    
        #imgB不经过transform处理，所以要手动把(H,W,C)转成(C,H,W)
        imgB = imgB.transpose(2,0,1) 
        #将numpy转化为tensor
        imgB = torch.FloatTensor(imgB)
        if self.transform:
            #一转成向量后，imgA通道就变成(C,H,W)
            imgA = self.transform(imgA) 
        #返回
        return imgA, imgB
#获取车辆数据集
car = CarDataset(transform)
#整个训练集中，百分之75为训练集
train_size = int(0.75 * len(car))  
#剩下的是验证集  
verify_size = len(car) - train_size
#划分训练集和验证集
train_dataset, verify_dataset = random_split(car, [train_size, verify_size]) 
 #DataLoader操作数据的类，参数batch_size(每个batch的大小), shuffle(是否进行shuffle操作), num_workers(加载数据的时候使用几个子进程)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
verify_dataloader = DataLoader(verify_dataset, batch_size=4, shuffle=True, num_workers=2)
#主函数 
if __name__ =='__main__':
    #打印训练集信息
    for train_batch in train_dataloader:
        print(train_batch)
    #打印验证集信息
    for verify_batch in verify_dataloader:
        print(verify_batch)
