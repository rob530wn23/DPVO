import torch
import torch.nn as nn
import torch.nn.functional as F

def tenarize(tensor):
    output = torch.zeros(tensor.size())
    delta = delta(tensor)
    alpha = alpha(tensor,delta)
    for i in range(tensor.size()[0]):
        for w in tensor[i].view(1,-1):
            pos_one = (w > delta[i]).type(torch.FloatTensor)
            neg_one = -1 * (w < -delta[i]).type(torch.FloatTensor)
        out = torch.add(pos_one,neg_one).view(tensor.size()[1:])
        output[i] = torch.add(output[i],torch.mul(out,alpha[i]))
    return output


def alpha(tensor,delta):
        alpha = []
        for i in range(tensor.size()[0]):
            count = 0
            abssum = 0
            absvalue = tensor[i].view(1,-1).abs()
            for w in absvalue:
                truth_value = w > delta[i]
            count = truth_value.sum()
            abssum = torch.matmul(absvalue,truth_value.type(torch.FloatTensor).view(-1,1))
            alpha.append(abssum/count)
        alpha = alpha[0]
        for i in range(len(alpha) - 1):
            alpha = torch.cat((alpha,alpha[i+1]))
        return alpha

def delta(tensor):
    n = tensor[0].nelement()
    # conv layer
    if(len(tensor.size()) == 4):
        delta = 0.7 * tensor.norm(1,3).sum(2).sum(1).div(n)
    # linear layer
    elif(len(tensor.size()) == 2):
        delta = 0.7 * tensor.norm(1,1).div(n)
    return delta


class TernaryLinear(nn.Linear):
    def __init__(self,*args,**kwargs):
        super(TernaryLinear,self).__init__(*args,**kwargs)

    def forward(self,input):
        self.weight.data = tenarize(self.weight.data)
        out = F.linear(input,self.weight,self.bias)
        return out

class TernaryConv2d(nn.Conv2d):
    def __init__(self,*args,**kwargs):
        super(TernaryConv2d,self).__init__(*args,**kwargs)

    def forward(self,input):
        self.weight.data = tenarize(self.weight.data)
        out = F.conv2d(input, self.weight, self.bias, self.stride,self.padding, self.dilation, self.groups)
        return out
