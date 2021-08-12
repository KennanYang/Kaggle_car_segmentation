#####FCN.py
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.vgg import VGG
 
 
class FCNs(nn.Module):
 
    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class  # 定义分类数，其中包括背景，识别车别的话就是背景+车辆一共两类
        self.pretrained_net = pretrained_net    # 定义卷积部分
        self.relu    = nn.ReLU(inplace=True)    # 定义激活函数
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)   # 定义反卷积层，输入通道数为512，输出通道数512，反卷积核尺寸为3，步长2
        self.bn1     = nn.BatchNorm2d(512)  # 归一化操作
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)   # 定义反卷积层，输入通道数为512，输出通道数256，反卷积核尺寸为3，步长2
        self.bn2     = nn.BatchNorm2d(256)  # 归一化操作
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)   # 定义反卷积层，输入通道数为256，输出通道数128，反卷积核尺寸为3，步长2
        self.bn3     = nn.BatchNorm2d(128)  # 归一化操作
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)   # 定义反卷积层，输入通道数为128，输出通道数64，反卷积核尺寸为3，步长2
        self.bn4     = nn.BatchNorm2d(64)  # 归一化操作
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)   # 定义反卷积层，输入通道数为64，输出通道数32，反卷积核尺寸为3，步长2
        self.bn5     = nn.BatchNorm2d(32)  # 归一化操作
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)     # 通过正向卷积最终得出像素级分类结果，通道数即为分类数，由于onehot编码，结果中类别对应的通道为1，其余为0
        # classifier is 1x1 conv, to reduce channels from 32 to n_class

    # 前向传播计算分类结果过程
    def forward(self, x):
        # output是在后面预训练网络函数中的输出，其中存放了该网络五个池化层的输出结果
        output = self.pretrained_net(x)
        x5 = output['x5']  # 最后一层池化层输出
        x4 = output['x4']  # 第四层池化层输出
        x3 = output['x3']  # 第三层池化层输出
        x2 = output['x2']  # 第二层池化层输出
        x1 = output['x1']  # 第一层池化层输出
 
        score = self.bn1(self.relu(self.deconv1(x5)))     # 最后一个池化层的输出也就是卷积部分的输出，对其进行反卷积尺寸扩大一定程度，并进行激活、归一化
        score = score + x4                                # 将上一句的结果与第四层池化层输出结果（对应元素相加）
        score = self.bn2(self.relu(self.deconv2(score)))  # 将上一句的输出进行反卷积，尺寸扩大一定程度，并进行激活、归一化
        score = score + x3                                # 将上一句的结果与第三层池化层输出结果（对应元素相加）
        score = self.bn3(self.relu(self.deconv3(score)))  # 将上一句的输出进行反卷积，尺寸扩大一定程度，并进行激活、归一化
        score = score + x2                                # 将上一句的结果与第二层池化层输出结果（对应元素相加）
        score = self.bn4(self.relu(self.deconv4(score)))  # 将上一句的输出进行反卷积，尺寸扩大一定程度，并进行激活、归一化
        score = score + x1                                # 将上一句的结果与第一层池化层输出结果（对应元素相加）
        score = self.bn5(self.relu(self.deconv5(score)))  # 将上一句的输出进行反卷积，尺寸扩大一定程度，并进行激活、归一化，此时尺寸以及还原成了原图片的尺寸
        score = self.classifier(score)                    # 将上一句的输出进行正卷积，输出像素级的分类结果
 
        return score  
 
 
class VGGNet(VGG):
    def __init__(self, model='vgg16', remove_fc=True):
        super().__init__(make_layers(cfg[model])) # 调用父类函数，将本类中定义的网络结构实际搭建起来
        self.ranges = ranges[model] # ranges定义的是卷积部分卷积层的范围
 
        # delete redundant fully-connected layer params, can save memory
        # 去掉vgg最后的全连接层(classifier)
        if remove_fc:  
            del self.classifier

    # 前向传播
    def forward(self, x):
        output = {}
        # 得到每一个池化层输出 (VGG net一共有五个最大池化层)
        for idx, (begin, end) in enumerate(self.ranges):
            # self.ranges = ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)) (vgg16)
            # 对于卷积层，就让它单纯地前向传播，每一个range意义是两个池化层之间的几个卷积层，包括后一个池化层
            for layer in range(begin, end):
                x = self.features[layer](x)
            # 跳出上述的小循环说明一个range已经前向完毕，此时得到的是一个池化层的输出，将这个池化层的输出放入output中
            output["x%d"%(idx+1)] = x
 
        return output

# 定义卷积层-池化层、卷积层-池化层。。。这样的间隔
ranges = {
    'vgg11': ((0, 3), (3, 6),  (6, 11),  (11, 16), (16, 21)),
    'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
    'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
    'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
}


# Vgg网络结构，数字代表通道数，M代表最大池化层，此处定义了四种VGG网络结构
cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

# 由cfg构建vgg-Net
def make_layers(cfg, batch_norm=False):
    layers = []     # 收纳网络各层
    in_channels = 3     # 定义输入通道数
    for v in cfg:
        # 如果当前应该往网络里加一个最大池化层，则layers增加一共核尺寸为2、步长为2的最大池化层
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        # 如果当前应该往网络里加一卷积层，则先定义一个核尺寸为3、进行填充的二维卷积层
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:  # 如果需要归一化，则layers添加：卷积层、归一化层、激活层
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:           # 如果不需要归一化，则layers添加：卷积层、激活层
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v # 将通道数改为该层输出的通道数
    return nn.Sequential(*layers)
 
 
if __name__ == "__main__":
    pass
