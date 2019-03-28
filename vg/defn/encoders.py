import torch
import torch.nn as nn
from onion import attention, conv
import onion.util as util
import torch.nn.functional as F

def l2normalize(x):
    return F.normalize(x, p=2, dim=1)

#LEGACY
class TextEncoder(nn.Module):

    def __init__(self, size_feature, size, size_embed=64, depth=1, size_attn=512, dropout_p=0.0):
        super(TextEncoder, self).__init__()
        util.autoassign(locals())
        self.h0   = torch.autograd.Variable(torch.zeros(self.depth, 1, self.size))
        self.Embed  = nn.Embedding(self.size_feature, self.size_embed) 
        self.Dropout = nn.Dropout(p=self.dropout_p)
        self.RNN  = nn.GRU(self.size_embed, self.size, self.depth, batch_first=True)
        self.Attn = attention.SelfAttention(self.size, size=self.size_attn)
    
    def forward(self, text):
        h0 = self.h0.expand(self.depth, text.size(0), self.size).cuda()
        out, last = self.RNN(self.Dropout(self.Embed(text)), h0)
        return l2normalize(self.Attn(out))


class TextEncoderBottom(nn.Module):

    def __init__(self, size_feature, size, size_embed=64, depth=1, dropout_p=0.0):
        super(TextEncoderBottom, self).__init__()
        util.autoassign(locals())
        self.h0   = torch.autograd.Variable(torch.zeros(self.depth, 1, self.size))
        self.Embed  = nn.Embedding(self.size_feature, self.size_embed) 
        self.Dropout = nn.Dropout(p=self.dropout_p)
        self.RNN  = nn.GRU(self.size_embed, self.size, self.depth, batch_first=True)

    
    def forward(self, text):
        h0 = self.h0.expand(self.depth, text.size(0), self.size).cuda()
        out, last = self.RNN(self.Dropout(self.Embed(text)), h0)
        return out


class TextEncoderTop(nn.Module):

    def __init__(self, size_feature, size, depth=1, size_attn=512, dropout_p=0.0):
        super(TextEncoderTop, self).__init__()
        util.autoassign(locals())
        if self.depth > 0:
            self.h0   = torch.autograd.Variable(torch.zeros(self.depth, 1, self.size))
            self.Dropout = nn.Dropout(p=self.dropout_p)
            self.RNN  = nn.GRU(self.size_feature, self.size, self.depth, batch_first=True)
        self.Attn = attention.SelfAttention(self.size, size=self.size_attn)
    
    def forward(self, x):
        if self.depth > 0:
            h0 = self.h0.expand(self.depth, x.size(0), self.size).cuda()
            out, _last = self.RNN(self.Dropout(x), h0)
        else:
            out = x
        return l2normalize(self.Attn(out))


class SpeechEncoder(nn.Module):

    def __init__(self, size_vocab, size, depth=1, filter_length=6, filter_size=64, stride=2, size_attn=512, dropout_p=0.0):
        super(SpeechEncoder, self).__init__()
        util.autoassign(locals())
        self.h0 = torch.autograd.Variable(torch.zeros(self.depth, 1, self.size))
        self.Conv = conv.Convolution1D(self.size_vocab, self.filter_length, self.filter_size, stride=self.stride)
        self.Dropout = nn.Dropout(p=self.dropout_p)
        self.RNN = nn.GRU(self.filter_size, self.size, self.depth, batch_first=True)
        self.Attn = attention.SelfAttention(self.size, size=self.size_attn)

    def forward(self, input):
        h0 = self.h0.expand(self.depth, input.size(0), self.size).cuda()
        out, last = self.RNN(self.Dropout(self.Conv(input)), h0)
        return l2normalize(self.Attn(out))


class GRUStack(nn.Module):
    """GRU stack with separate GRU modules so that full intermediate states can be accessed."""
    def __init__(self, size_in, size, depth):
        super(GRUStack, self).__init__()
        self.bottom = nn.GRU(size_in, size, 1, batch_first=True)
        self.layers = nn.ModuleList([nn.GRU(size, size, 1, batch_first=True) for i in range(depth-1) ])
    
    def forward(self, x):
        hidden = []
#        print("rnn x", x.size())
        output, _ = self.bottom(x)
#        print("rnn bottom", output.size())
        hidden.append(output)
        for rnn in self.layers:
            output, _ = rnn(hidden[-1])
#            print("rnn middle", output.size())
            hidden.append(output)
        return torch.stack(hidden)

class SpeechEncoderBottom(nn.Module):
    def __init__(self, size_vocab, size, depth=1, filter_length=6, filter_size=64, stride=2, dropout_p=0.0):
        super(SpeechEncoderBottom, self).__init__()
        util.autoassign(locals())
        self.Conv = conv.Convolution1D(self.size_vocab, self.filter_length, self.filter_size, stride=self.stride)
        if self.depth > 0:
            self.h0 = torch.autograd.Variable(torch.zeros(self.depth, 1, self.size))
            self.Dropout = nn.Dropout(p=self.dropout_p)
            self.RNN = nn.GRU(self.filter_size, self.size, self.depth, batch_first=True)

    def forward(self, x):
        if self.depth > 0:
            h0 = self.h0.expand(self.depth, x.size(0), self.size).cuda()
            out, last = self.RNN(self.Dropout(self.Conv(x)), h0)
        else:
            out = self.Conv(x)
        return out

class SpeechEncoderBottomStack(nn.Module):
    def __init__(self, size_vocab, size, depth=1, filter_length=6, filter_size=64, stride=2, dropout_p=0.0):
        super(SpeechEncoderBottomStack, self).__init__()
        util.autoassign(locals())
        self.Conv = conv.Convolution1D(self.size_vocab, self.filter_length, self.filter_size, stride=self.stride)
        if self.depth > 0:
            self.Dropout = nn.Dropout(p=self.dropout_p)
            self.RNN = GRUStack(self.filter_size, self.size, self.depth)

    def forward(self, x):
        return self.states(x)[-1]

    def states(self, x):
        if self.depth > 0:
#            print("x", x.size())
#            x = self.Conv(x)
#            print("conv", x.size())
#            x = self.Dropout(x)
#            print("dropout", x.size())
#            x = self.RNN(x)
#            print("rnn.forward", x.size())
#            out = x
            out = self.RNN(self.Dropout(self.Conv(x)))       
        else:
            out = self.Conv(x)
        return out
    

class SpeechEncoderBottomNoConv(nn.Module):
    def __init__(self, size_vocab, size, depth=1, dropout_p=0.0):
        super(SpeechEncoderBottomNoConv, self).__init__()
        util.autoassign(locals())
        if self.depth > 0:
            self.h0 = torch.autograd.Variable(torch.zeros(self.depth, 1, self.size))
            self.Dropout = nn.Dropout(p=self.dropout_p)
            self.RNN = nn.GRU(self.size_vocab, self.size, self.depth, batch_first=True)

    def forward(self, x):
        if self.depth > 0:
            h0 = self.h0.expand(self.depth, x.size(0), self.size).cuda()
            out, last = self.RNN(self.Dropout(x), h0)
        else:
            out = x
        return out

class SpeechEncoderBottomBidi(nn.Module):
    def __init__(self, size_vocab, size, depth=1, dropout_p=0.0):
        super(SpeechEncoderBottomBidi, self).__init__()
        util.autoassign(locals())
        if self.depth > 0:
            self.Dropout = nn.Dropout(p=self.dropout_p)
            self.RNN = nn.GRU(self.size_vocab, self.size, self.depth, batch_first=True, bidirectional=True)
            self.Down = nn.Linear(self.size * 2, self.size)

    def forward(self, x):
        if self.depth > 0:
            out, last = self.RNN(self.Dropout(x))
            out = self.Down(out)
        else:
            out = x
        return out

class SpeechEncoderTop(nn.Module):
    def __init__(self, size_input, size, depth=1, size_attn=512, dropout_p=0.0):
        super(SpeechEncoderTop, self).__init__()
        util.autoassign(locals())
        if self.depth > 0:
            self.h0 = torch.autograd.Variable(torch.zeros(self.depth, 1, self.size))
            self.Dropout = nn.Dropout(p=self.dropout_p)
            self.RNN = nn.GRU(self.size_input, self.size, self.depth, batch_first=True)
        self.Attn = attention.SelfAttention(self.size, size=self.size_attn)
        

    def forward(self, x):
        if self.depth > 0:
            h0 = self.h0.expand(self.depth, x.size(0), self.size).cuda()
            out, _last = self.RNN(self.Dropout(x), h0)
        else:
            out = x
        return l2normalize(self.Attn(out))

    def states(self, x):
        if self.depth > 0:
            h0 = self.h0.expand(self.depth, x.size(0), self.size).cuda()
            out, _last = self.RNN(self.Dropout(x), h0)
        else:
            out = x
        return out, l2normalize(self.Attn(out))

class SpeechEncoderTopStack(nn.Module):
    def __init__(self, size_input, size, depth=1, size_attn=512, dropout_p=0.0):
        super(SpeechEncoderTopStack, self).__init__()
        util.autoassign(locals())
        if self.depth > 0:
            self.Dropout = nn.Dropout(p=self.dropout_p)
            self.RNN = GRUStack(self.size_input, self.size, self.depth)
        self.Attn = attention.SelfAttention(self.size, size=self.size_attn)

    def states(self, x):
        if self.depth > 0:
            out = self.RNN(self.Dropout(x))
        else:
            out = x
        return out

    def forward(self, x):
        out = self.states(x)
        return l2normalize(self.Attn(out[-1]))

    
class SpeechEncoderTopBidi(nn.Module):
    def __init__(self, size_input, size, depth=1, size_attn=512, dropout_p=0.0):
        super(SpeechEncoderTopBidi, self).__init__()
        util.autoassign(locals())
        if self.depth > 0:
            self.Dropout = nn.Dropout(p=self.dropout_p)
            self.RNN = nn.GRU(self.size_input, self.size, self.depth, batch_first=True, bidirectional=True)
            self.Down = nn.Linear(self.size * 2, self.size)
        self.Attn = attention.SelfAttention(self.size, size=self.size_attn)
        

    def forward(self, x):
        if self.depth > 0:
            out, _last = self.RNN(self.Dropout(x))
            out = self.Down(out)
        else:
            out = x
        return l2normalize(self.Attn(out))

    def states(self, x):
        if self.depth > 0:
            out, _last = self.RNN(self.Dropout(x))
            out = self.Down(out)
        else:
            out = x
        return out, l2normalize(self.Attn(out))
    
class ImageEncoder(nn.Module):
    
    def __init__(self, size, size_target):
        super(ImageEncoder, self).__init__()
        self.Encoder = util.make_linear(size_target, size)
    
    def forward(self, img):
        return l2normalize(self.Encoder(img))


