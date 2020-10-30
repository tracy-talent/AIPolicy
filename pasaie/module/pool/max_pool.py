import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PieceMaxPool(nn.Module):
    """
    Piecewise MaxPooling
    """
    def __init__(self, kernel_size, piece_num=None):
        """
        Args:
            kernel_size: kernel_size for CNN
            piece_num: piece_num of PCNN, None for common pool
        """
        super().__init__()
        self.piece_num = piece_num
        if self.piece_num != None:
            self.mask_embedding = nn.Embedding(piece_num + 1, piece_num)
            self.mask_embedding.weight.data.copy_(torch.FloatTensor(np.concatenate([np.zeros((1, piece_num)), np.identity(piece_num)], axis=0)))
            self.mask_embedding.weight.requires_grad = False
            self._minus = -100
        self.pool = nn.MaxPool1d(kernel_size)

    def forward(self, x, mask=None):
        """
        Args:
            input features: (B, I_EMBED, L)
        Return:
            output features: (B, H_EMBED)
        """
        # Check size of tensors
        if mask is None or self.piece_num is None or self.piece_num == 1:
            x = self.pool(x).squeeze(-1) # (B, I_EMBED, 1) -> (B, I_EMBED)
            return x
        else:
            B, I_EMBED, L = x.size()[:3]
            mask_embedded = 1 - self.mask_embedding(mask).transpose(1, 2).unsqueeze(2) # (B, L) -> (B, L, S) -> (B, S, L) -> (B, S, 1, L)
            x = x.unsqueeze(1) # (B, I_EMBED, L) -> (B, 1, I_EMBED, L)
            x = (x + self._minus * mask_embedded).contiguous().view([-1, I_EMBED, L]) # (B, S, I_EMBED, L) -> (B * S, I_EMBED, L)
            x = self.pool(x).squeeze(-1) # (B * S, I_EMBED, 1) -> (B * S, I_EMBED)
            x = x.view([B, -1])  # (B, S * I_EMBED)
            return x
            # mask_embedded = 1 - self.mask_embedding(mask).transpose(1, 2) # (B, L) -> (B, L, S) -> (B, S, L)
            # x = x.transpose(1, 2) # (B, L, I_EMBED) -> (B, I_EMBED, L)
            # pool1 = self.pool(x + self._minus * mask_embedded[:, 0:1, :]) # (B, I_EMBED, L) -> (B, I-EMBED)
            # pool2 = self.pool(x + self._minus * mask_embedded[:, 1:2, :])
            # pool3 = self.pool(x + self._minus * mask_embedded[:, 2:3, :])

            # x = torch.cat([pool1, pool2, pool3], 1) # (B, 3 * I-EMBED)
            # x = x.squeeze(-1)
            # return  x
