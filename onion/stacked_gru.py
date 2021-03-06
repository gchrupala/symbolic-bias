import torch.nn as nn
import torch
import onion.util as util
import torch.nn.functional as F



class StackedGRU(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers, residual=False, bidirectional=False, **kwargs):
        super(StackedGRU, self).__init__()
        assert num_layers > 0
        util.autoassign(locals())
        self.bottom = nn.GRU(input_size, hidden_size, 1, bidirectional=bidirectional, **kwargs)
        self.layers = nn.ModuleList()
        if bidirectional:
            self.downscale = nn.Linear(hidden_size * 2, hidden_size)
     
        for i in range(num_layers-1):
            layer = nn.GRU(hidden_size, hidden_size, 1, bidirectional=self.bidirectional, **kwargs)
            self.layers.append(layer)

    def forward(self, x, h0):
        j = 2 if self.bidirectional else 1
        out, _ = self.bottom(x, h0[0:1,:,:])
        out = self.proj(out)
        for i in range(self.num_layers-1):
            out =  self.Residual(self.layers[i], out, h0[i+1:i+2,:,:])
        return out
    
    def proj(self, x):
        if self.bidirectional:
            return self.downscale(x)
        else:
            return x        
        
    
    def Residual(self, f, x, *args):
        fx, _ = f(x, *args)
        #print(f)
        #print(" x:", x.size())
        #print("fx:", fx.size())
        px = self.proj(fx)    
        #print("px:", px.size())
        if self.residual:
            return x + px
        else:
            return px
