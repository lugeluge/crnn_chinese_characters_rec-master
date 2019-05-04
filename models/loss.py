import torch.nn.functional as F
from torch import nn


class CapsuleLoss(nn.Module):
    """其中

k 是分类
Tk 是分类的指示函数 (k 类存在为 1，不存在为 0)
m+ 为上界，惩罚假阳性 (false positive) ，即预测 k 类存在但真实不存在，识别出来但错了
m- 为下界，惩罚假阴性 (false negative) ，即预测 k 类不存在但真实存在，没识别出来
λ 是比例系数，调整两者比重
总的损失是各个样例损失之和。论文中 m+= 0.9, m-= 0.1, λ = 0.5，用大白话说就是

如果 k 类存在，||vk|| 不会小于 0.9
如果 k 类不存在，||vk|| 不会大于 0.1
惩罚假阳性的重要性大概是惩罚假阴性的重要性的 2 倍"""
    def __init__(self):
        super(CapsuleLoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss(reduction='sum')

    def forward(self, images, labels, classes, reconstructions):
        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2

        margin_loss = labels * left + 0.5 * (1. - labels) * right
        margin_loss = margin_loss.sum()
        images  = images.view(-1,28*28)
        reconstruction_loss = self.reconstruction_loss(reconstructions, images)

        return (margin_loss + 0.0005 * reconstruction_loss) / images.size(0)


if __name__ == "__main__":
    digit_loss = CapsuleLoss()
    print(digit_loss)
