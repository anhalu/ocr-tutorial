import torch 
from torchvision.models.vgg import vgg19_bn 


input = torch.randn((1,3,32,500))


model = vgg19_bn().features[:33]
x = model(input) 
print(x.shape) 
