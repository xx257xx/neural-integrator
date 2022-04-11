import numpy as np
from scipy.integrate import odeint
import torch
import torch.utils.data as data_0

## Chebyshev polynomial function for ODE solver
def func(y, t, c):
    
    return np.polynomial.chebyshev.Chebyshev(c)(t)


def raw_data_generator(Num, ord, D, y0):
    
    train_x = []
    train_y = []


    for i in range(Num):
    
        c = [2*np.random.rand(ord)-1]
        x = np.polynomial.chebyshev.Chebyshev(c[0])(D)
        y = odeint(func, y0, D, args = tuple(c))

        for j in D:
            train_x.append(np.append(x, j))
    
        for j in range(len(D)):
            train_y.append(y[j])
    
    train_x = torch.as_tensor(train_x).reshape(Num*len(D), len(D) + 1)
    train_y = torch.as_tensor(train_y).reshape(Num*len(D))
    
    print('done')

    result = {'train_X' : train_x, 'train_Y' : train_y, 'M' : len(D)}
    
    return result



 
    
class Dataset(data_0.Dataset):
    def __init__(self, x_tensor, y_tensor):
        super(Dataset, self).__init__()

        self.x = x_tensor
        self.y = y_tensor
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)
