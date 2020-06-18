import torch
import numpy as np

def gelu(x):
    return 0.5*x*(1+torch.tanh(np.sqrt(2/np.pi)*(x+0.044715*torch.pow(x, 3))))









print('x\tgelu\tbackward')

x = -4.0
while x < 4.0:
    x_ = torch.tensor([x], requires_grad=True)
    y = gelu(x_)
    y.backward()
    print('{}\t{}\t{}'.format(x, y.item(), x_.grad.item()))

    x += 0.1