import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F
class EmbeddingDropout(torch.nn.Embedding):

  def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None,
               norm_type=2, scale_grad_by_freq=False,
               sparse = False, dropout=0.1, scale = None):
    super(EmbeddingDropout, self).__init__(num_embeddings, embedding_dim, padding_idx,
                            max_norm, norm_type, scale_grad_by_freq, sparse)
    self.dropout=dropout
    self.padding_idx = padding_idx if padding_idx else -1
    assert dropout>=0.0 and dropout<1.0 , "Dropout must be >= 0.0 and < 1.0"
    self.scale = scale
    
  def forward(self, input):
    if self.training:
      dropout = self.dropout
    else:
      dropout = 0
      
    if dropout:
      mask = self.weight.data.new(self.weight.size(0), 1)
      mask.bernoulli_(1 - dropout)
      mask = mask.expand_as(self.weight)
      mask = mask / (1 - dropout)
      masked_weight = self.weight * Variable(mask)
    else:
      masked_weight = self.weight
    if self.scale and self.scale != 1:
      masked_weight = masked_weight * scale

    X = F.embedding(input, masked_weight,
                                       self.padding_idx, self.max_norm, self.norm_type,
                                       self.scale_grad_by_freq, self.sparse
    )

    return X
      
if __name__ == '__main__':
  V = 50
  h = 4
  bptt = 10
  batch_size = 2

  embed = EmbeddingDropout(V, h)

  words = np.random.random_integers(low=0, high=V-1, size=(batch_size, bptt))
  words = torch.LongTensor(words)
  words = Variable(words)

  origX = embed(words)

  print(origX)
