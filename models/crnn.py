import torch.nn as nn


class BidirectionalLSTM(nn.Module):
    # Inputs hidden units Out
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class CRNN(nn.Module):
    #                   32    1   37     256  nc一开始输入为1 因为是gray图片
    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2] #核大小
        ps = [1, 1, 1, 1, 1, 1, 0] #填充0
        ss = [1, 1, 1, 1, 1, 1, 1] #步长
        nm = [64, 128, 256, 256, 512, 512, 512] #有多少个filter

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(in_channels=nIn, out_channels=nOut, kernel_size=ks[i], stride=ss[i], padding=ps[i]))
                                                                                    #padding=(kernel_size-1)/2 if stride=1
            if batchNormalization:#每一层对输出进行批量正则化
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu: #ReLU是将所有的负值都设为零，相反，Leaky ReLU是给所有负值赋予一个非零斜率
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        # windows 大小2*2 宽方向的步长为2，高方向的步长的1，左右填充为零，上下填充为1
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d(kernel_size=(2,2),stride=(2,1),padding=(0,1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))
        self.linear = nn.Linear(512,nclass)
    def forward(self, input):
        # conv features
        #print('---forward propagation---')
        conv = self.cnn(input)
        # print(conv.size()) batch_size*512*1*width
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2) # b *512 * width  维度压缩
        conv = conv.permute(2, 0, 1)  # [w, b, c] #维度变换

        #修改
        conv = conv.contiguous().view(w*b,-1)
        output  = self.linear(conv)
        output = output.contiguous().view(w,b,-1)

        #原始
        # #print(conv.size()) # width batch_size channel
        # # rnn features
        # output = self.rnn(conv)
        # #print(output.size(0))
        # # print(output.size())# width*batch_size*nclass


        return output
