########train.py
from datetime import datetime
 
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import visdom
 
from CarData import verify_dataloader, train_dataloader
from FCN import FCNs, VGGNet
 
# 训练模型，迭代次数默认50次
def train(epo_num=50, show_vgg_params=False):
 
    #vis = visdom.Visdom()
    #CUDA加速
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #初始化VGG网络
    vgg_model = VGGNet(requires_grad=True, show_params=show_vgg_params)
    #初始化FCN网络
    fcn_model = FCNs(pretrained_net=vgg_model, n_class=2)
    #传入设备参数
    fcn_model = fcn_model.to(device)
    #定义损失函数
    criterion = nn.BCELoss().to(device)
    #优化器
    optimizer = optim.SGD(fcn_model.parameters(), lr=1e-2, momentum=0.7)
    #每次训练迭代的损失值
    all_train_iter_loss = []
    #每次验证迭代的损失值
    all_verify_iter_loss = []
 
    #标记开始训练时间
    prev_time = datetime.now()
    #迭代训练过程
    for epo in range(epo_num):
        #初始化训练损失值
        train_loss = 0
        #定义训练模型
        fcn_model.train()
        #从训练集中取数据和标签
        for index, (car, car_msk) in enumerate(train_dataloader):
            # car.shape is torch.Size([4, 3, 160, 160])
            # car_msk.shape is torch.Size([4, 2, 160, 160])
            #将数据加载到GPU
            car = car.to(device)
            #将标签加载到GPU
            car_msk = car_msk.to(device)
            #初始化优化器，参数设成零
            optimizer.zero_grad()
            #正向传播
            output = fcn_model(car)
            #激活函数
            output = torch.sigmoid(output) # output.shape is torch.Size([4, 2, 160, 160])
            # print(output)
            # print(car_msk)
            #将输出与标签比较，计算出损失值
            loss = criterion(output, car_msk)
            #反向传播
            loss.backward()
            #将损失值从张量类型转化为浮点型
            iter_loss = loss.item()
            #将转化后的损失值加入到列表中
            all_train_iter_loss.append(iter_loss)
            #更新总损失值
            train_loss += iter_loss
            #更新优化器
            optimizer.step()
            #将输出从张量类型转化为矩阵类型
            #output_np = output.cpu().detach().numpy().copy() # output_np.shape = (4, 2, 160, 160)

            #output_np = np.argmin(output_np, axis=1)
            
            #car_msk_np = car_msk.cpu().detach().numpy().copy() # car_msk_np.shape = (4, 2, 160, 160) 
           # car_msk_np = np.argmin(car_msk_np, axis=1)
        #初始化验证集损失
        verify_loss = 0
        #model.eval()是模型的某些特定层/部分的一种开关，这些层/部分在训练和推断（评估）期间的行为不同。
        fcn_model.eval()
        #torch.no_grad() 是一个上下文管理器，被该语句 wrap 起来的部分将不会track 梯度
        with torch.no_grad():
            #遍历验证集中的索引和车
            for index, (car, car_msk) in enumerate(verify_dataloader):
                # 车辆样本送到GPU
                car = car.to(device)
                # 车辆标签送到GPU
                car_msk = car_msk.to(device)
                # optimizer.zero_grad()意思是把梯度置零，也就是把loss关于weight的导数变成0.
                optimizer.zero_grad() 
                # 记录模型的预测值
                output = fcn_model(car)
                # 函数sigmoid()的用法：神经网络中的激活函数。 功能：完成逻辑回归的软判决。
                output = torch.sigmoid(output) # output.shape is torch.Size([4, 2, 160, 160])
                # 将输出与标签比较，计算出损失值
                loss = criterion(output, car_msk)
                # 将损失值从张量类型转化为浮点型
                iter_loss = loss.item()
                # 将转化后的损失值加入到列表中
                all_verify_iter_loss.append(iter_loss)
                # 更新总损失值
                verify_loss += iter_loss
 
                #output_np = output.cpu().detach().numpy().copy() # output_np.shape = (4, 2, 160, 160)  
                #output_np = np.argmin(output_np, axis=1)
                '''
                car_msk_np = car_msk.cpu().detach().numpy().copy() # car_msk_np.shape = (4, 2, 160, 160) 
                car_msk_np = np.argmin(car_msk_np, axis=1)
        '''
        #获得系统时间
        cur_time = datetime.now()
        #拆分小时
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        #拆分分秒
        m, s = divmod(remainder, 60)
        #按照时分秒进行显示
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        prev_time = cur_time
        #输出每次迭代的损失率，以及耗时
        print('epoch train loss = %f, epoch verify loss = %f, %s'
                %(train_loss/len(train_dataloader), verify_loss/len(verify_dataloader), time_str))
        
        #每迭代五次
        if np.mod(epo+1, 5) == 0:
            #保存一次模型
            torch.save(fcn_model, 'checkpoints/fcn_model_{}.pt'.format(epo))
            #并输出迭代次数
            print('saveing checkpoints/fcn_model_{}.pt'.format(epo))
 
 #主函数
if __name__ == "__main__":
    #训练模型，迭代次数设置为100次
    train(epo_num=100, show_vgg_params=False)
