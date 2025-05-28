import torch
import torch.nn as nn
import torch.nn.functional as F
from HybridTensor.routers.router_utils import CONFIG


class Router(torch.nn.Module):
    def __init__(self, model_dim, inner_dim = 1024, out_dim = None, norm="layer", hidden_activation="none"):
        super(Router, self).__init__()
        self.fc1 = torch.nn.Linear(model_dim, inner_dim, bias=None)
        self.use_act = True
        
        if norm == 'batch':
            self.norm = torch.nn.BatchNorm1d(inner_dim)
        elif norm == 'layer':
            self.norm = torch.nn.LayerNorm(inner_dim)
        else:
            self.norm = nn.Identity()
            
        self.activation_fn(hidden_activation)
        
        if out_dim is None:
            out_dim = model_dim * 4
        self.fc2 = torch.nn.Linear(inner_dim, out_dim, bias=None)
        
    def activation_fn(self, hidden_activation):
        if hidden_activation == "relu":
            self.activation = torch.nn.ReLU()
        elif hidden_activation == 'relu_squared':
            self.activation = ReLUSquared()
        elif hidden_activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(negative_slope=0.01)
        elif hidden_activation == 'swish':
            self.activation = nn.SiLU()
        elif hidden_activation == 'elu':
            self.activation = nn.ELU(alpha=1.0)
        elif hidden_activation == 'gelu':
            self.activation = nn.GELU()
        elif hidden_activation == 'selu':
            self.activation = nn.SELU()
        elif hidden_activation == 'mish':
            self.activation = Mish()
        elif hidden_activation == 'none':
            self.activation = nn.Identity()
            
    def use_dropout(self, args):
        if args.dropout > 0:
            self.use_dropout = True
            self.dropout = torch.nn.Dropout(p=args.dropout)
        else:
            self.use_dropout = False
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.norm(x)
        x = self.fc2(x)
        
        return x
            
    
# Custom activation functions
class ReLUSquared(nn.Module):
    def forward(self, x):
        return F.relu(x) ** 2

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self, dim):
        super(SwiGLU, self).__init__()
        self.dim = dim
        self.activation = nn.SiLU()

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return self.activation(x1) * x2

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))