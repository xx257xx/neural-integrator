

import torch
import torch.nn as nn


## parameters for Neural Network
width = 256
depth = 3
P = 64


class DON(nn.Module):
    def __init__(self, M, width, depth, P):
        super(DON, self).__init__()
        
        self.M = M
        
        
        ## trunk net
        self.trunk_s = nn.Linear(M, width)        
        self.trunk_list = torch.nn.ModuleList([])
        
        for i in range(depth):
            self.trunk_list.append(nn.Linear(width, width))
            
        self.trunk_t = nn.Linear(width, P)
        
        ## branch net
        self.branch_s = nn.Linear(1, width)
        self.branch_list = torch.nn.ModuleList([])
        
        for i in range(depth):
            self.branch_list.append(nn.Linear(width, width))
        
        self.branch_t = nn.Linear(width, P)
        
        
        ## PReLU's
        self.prelus_t = torch.nn.ModuleList([])
        for i in range(depth + 1):
            self.prelus_t.append(torch.nn.PReLU(width))
            
        self.prelus_b = torch.nn.ModuleList([])
        for i in range(depth + 1):
            self.prelus_b.append(torch.nn.PReLU(width))
            
        
        
    def forward(self, x):
    
        u0 = x[:,0:self.M]
        t = x[:,self.M].reshape(x.shape[0],1)
        
    
    
        u0 = self.prelus_t[0](self.trunk_s(u0))
        
        for i in range(1, depth+1):
            
            u0 = self.prelus_t[i](self.trunk_list[i-1](u0)) + u0
            
        u0 = self.trunk_t(u0)
        
        t = self.prelus_b[0](self.branch_s(t))
        
        for i in range(1, depth+1):
            
            t = self.prelus_b[i](self.branch_list[i-1](t)) + t
            
        t = self.branch_t(t)
        
        
        y = torch.sum(t * u0, dim = -1)
        
   
        return y

