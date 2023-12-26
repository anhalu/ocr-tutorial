import math 

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from torchvision.models.vgg import vgg19_bn 
from torchvision.models.resnet import resnet34 


class CNN(nn.Module): 
    def __init__(self, pretrained = False) -> None:
        super(CNN, self).__init__()
        
        self.cnn = vgg19_bn(weights = None).features[:33] 
    
    def forward(self, x): 
        x = self.cnn(x) 
        return x 
    
class OneHot(nn.Module):
    def __init__(self, depth):
        super(OneHot, self).__init__()
        emb = nn.Embedding(depth, depth)
        emb.weight.data = torch.eye(depth)
        emb.weight.requires_grad = False
        self.emb = emb

    def forward(self, input_):
        return self.emb(input_) 
    
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size

        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size), requires_grad=True)
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(1)
        h = hidden.expand(timestep, -1, -1).transpose(0, 1)
        attn_energies = self.score(h, encoder_outputs)
        return attn_energies.softmax(2)

    def score(self, hidden, encoder_outputs):
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(1, 2)
        v = self.v.expand(encoder_outputs.size(0), -1).unsqueeze(1)
        energy = torch.bmm(v, energy)
        return energy


model = CNN() 
input = torch.randn((1,3,32,500)) 
x = model(input) 
print(x.shape)










        