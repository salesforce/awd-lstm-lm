import torch
from torch.nn import Parameter
from functools import wraps

class WeightDrop(torch.nn.Module):
    def __init__(self, module, weights, dropout=0, variational=False):
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self.variational = variational
        self._setup()

    def _setup(self):
        for name_w in self.weights:
            print('Applying weight drop of {} to {}'.format(self.dropout, name_w))
            w = getattr(self.module, name_w)
            del self.module._parameters[name_w]
            self.module.register_parameter(name_w + '_raw', Parameter(w.data))

    def _setweights(self):
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + '_raw')
            w = None
            if self.variational:
                mask = torch.autograd.Variable(torch.ones(raw_w.size(0), 1))
                if raw_w.is_cuda: mask = mask.cuda()
                mask = torch.nn.functional.dropout(mask, p=self.dropout, training=True)
                w = mask.expand_as(raw_w) * raw_w
            else:
                w = torch.nn.functional.dropout(raw_w, p=self.dropout, training=self.training)
            setattr(self.module, name_w, w)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)

if __name__ == '__main__':
    x = torch.nn.Linear(10, 10)
    x.bias.data *= 0
    y = torch.nn.functional.dropout(torch.autograd.Variable(torch.ones(10, 1)), p=0.5, training=True)
    z = torch.autograd.Variable(torch.rand(2, 10))
    print(x.weight)
    print(y)
    print(x(z))
    x.weight.data = (y.expand_as(x.weight) * x.weight).data
    print(x.weight)
    print(x(z))
