import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Implement the positional encoding (PE) function.
    PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
    PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))
    """

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, input):
        """
        Args:
            input: N x T x D
        """
        length = input.size(1)
        return self.pe[:, :length]


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    d_model = 512
    max_len = 5000
    pe = PositionalEncoding(d_model, max_len)
    mat = pe.pe.numpy()[0]  # (5000, 512)
    mat = np.transpose(mat, (1, 0))
    print(mat.shape)
    print(mat)
    plt.imshow(mat)
    plt.colorbar()
    plt.show()
