#######predict.py
 
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
import os
import cv2
from CarData import verify_dataloader, train_dataloader
from FCN import FCNs, VGGNet
from PIL import Image


# 对数据集进行标准化处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# 预测测试集中的图片分割结果
def predict(model):
    # 测试集中图片数量
    image_num=len(os.listdir('test'))
    # 循环遍历所有测试集的图片
    for idx in range(image_num):
        # 获取每一张图片的路径
        tempfilename = os.listdir('test')[idx]
        # 分割图片的名称和后缀
        (filename,extension) = os.path.splitext(tempfilename)
        # 读取测试集图片
        imgA = cv2.imread('test/'+filename+'.jpg')
        # 缩小图片
        imgA = cv2.resize(imgA, (192, 128))
        # 对图片进行标准化处理
        imgA = transform(imgA)
        # 将图片送到GPU处理
        imgA = imgA.to(device)
        # 给图片增加维度
        imgA = imgA.unsqueeze(0)
        # 使用神经网络模型对测试集进行预测
        output = model(imgA)
        # 函数sigmoid()的用法：神经网络中的激活函数。 功能：完成逻辑回归的软判决。
        output = torch.sigmoid(output)
        # 如果想把CUDA tensor格式的数据改成numpy时，需要先将其转换成cpu float-tensor随后再转到numpy格式。
        output_np = output.cpu().detach().numpy().copy()  # output_np.shape = (4, 2,  128, 192)
        # 将张量降维变成图片
        output_np = np.argmin(output_np, axis=1)
        # 去掉维度为1的轴
        imgB  = np.squeeze(output_np)
        # 改变图片像素点的格式
        imgB = imgB.astype('uint8')
        # 将0/1像素点变成黑白图
        imgB = imgB * 255
        # 把图片数组转化为PIL格式的灰度图
        imgB = Image.fromarray(imgB,'L')
        #imgB.show()
        # 图片保存到predict文件夹中
        imgB.save('predict/'+ filename +'.gif')
    print('predict finished.')



# mIoU实现
def getmIou():
    # 初始化一个list存每个预测结果的iou评价指标
    ious = []
    # 获取./predict文件夹下文件的数目
    image_num=len(os.listdir('predict'))
    # 便利./predict下所有的预测图
    for idx in range(image_num):
        # 获得文件路径
        tempfilename = os.listdir('predict')[idx]
        # 获得文件名和后缀
        (filename,extension) = os.path.splitext(tempfilename)
        # 读取预测图
        pred = Image.open('predict/'+filename+'.gif')
        # 读取标签
        target = Image.open('test_masks/'+filename+'_mask.gif')
        # 图片尺寸缩小10倍
        target=target.resize((192,128),Image.NEAREST)
        # 获得两个图片的并集
        intersection = np.logical_and(target,pred)
        # 获得两个图片的交集
        union = np.logical_or(target,pred)
        # 获得iou，即交并比
        iou_score=np.sum(intersection)/np.sum(union)
        #print('idx',idx,'iou_score',iou_score)
        # iou添加到list中
        ious.append(iou_score)
    # 求所有预测图的iou指标的均值
    return np.mean(ious)

# 主函数
if __name__ =='__main__': 
    # 用一个list存储vgg网络结构层数。   
    vggs=[11,13,16,19]
    # 比较四种VGG网络
    for i in range(4):
        # 比较不同迭代次数
        for j in range(20):
            # 使用cuda或者使用cpu
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # 不同模型的比较
            model = torch.load('checkpoints--vgg'+str(vggs[i])+'/fcn_model_'+str(4+5*j)+'.pt')
            # 根据已经训练好的模型预测测试集车辆分割结果
            predict(model)
            # 计算测试集预测结果的miou指标，评估分割准确率
            print('vgg',vggs[i],' fcn_model_',(4+5*j),' miou:',getmIou())