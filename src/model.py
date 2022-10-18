# coding = gbk
"""
定义网络
"""
import torch.nn as nn
from torchvision import models


def waste_removal_model(num_classes=2):
    """
    清运审核模型
    输入三通道图像
    :param num_classes:
    :return: 多标签，每个标签对应0-1之间的置信度
    """
    # 迁移学习模型
    model = models.mobilenet_v2(pretrained=False)
    # 冻结参数，不参与训练
    # for param in model.parameters():
    #     param.requires_grad = True

    # print(model)
    # 修改全连接层
    model.classifier[1] = SigmodLinear(in_features=1280, out_features=num_classes)

    return model


def abnormal_picture_model(num_classes=2):
    """
    异常图片识别模型
    输入多通道图像
    :param num_classes:
    :return: 多标签，每个标签对应0-1之间的置信度
    """
    # 迁移学习模型
    model = models.mobilenet_v2(pretrained=False)
    # 冻结参数，不参与训练
    # for param in model.parameters():
    #     param.requires_grad = True

    # print(model)
    # 修改全连接层
    # model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1), bias=False)
    model.classifier[1] = SigmodLinear(in_features=1280, out_features=num_classes)

    return model


def machine_full_verify_model(num_classes=3):
    """
    满仓审核模型
    :param num_classes:
    :return: 多类别，输出再经过softmax函数
    """
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes, bias=True)

    return model


class SigmodLinear(nn.Module):
    """
    将sigmoid函数作用到模型的最后一个全连接层输出，将每个类别的得分归一化到0-1之间，用于多标签分类模型
    """

    def __init__(self, in_features, out_features):
        super(SigmodLinear, self).__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features, bias=True)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        return self.sigmod(self.linear(x))
