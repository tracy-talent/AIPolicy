import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PieceAvgPool(nn.Module):
    """
    Piecewise AveragePooling
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
            self.mask_embedding.weight.data.copy_(torch.FloatTensor(np.concatenate([np.zeros(piece_num), np.identity(piece_num)], axis = 0)))
            self.mask_embedding.weight.requires_grad = False

    def forward(self, x, mask=None):
        """
        Args:
            input features: (B, I_EMBED, L)
        Return:
            output features: (B, H_EMBED)
        """
        # Check size of tensors
        if mask == None or self.piece_num == None or self.piece_num == 1:
            x = self.pool(x).squeeze(-1) # (B, I_EMBED, 1) -> (B, I_EMBED)
            return x
        else:
            B, I_EMBED, L = x.size()[:3]
            mask_embedded = self.mask_embedding(mask).transpose(1, 2) # (B, L) -> (B, L, S) -> (B, S, L)
            piece_len = torch.sum(mask_embedded, dim=-1, keepdim=True) # (B, S, L) -> (B, S, 1)
            piece_len[piece_len == 0] = 1e-5  # 防止除0，能保证分母为0的分子为0所以随便给个不为0的数即可
            mask_embedded = mask_embedded.unsqueeze(2) # (B, S, L) -> (B, S, 1, L)
            x = x.unsqueeze(1) # (B, I_EMBED, L) -> (B, 1, I_EMBED, L)
            x = x * mask # (B, S, I_EMBED, L)
            x = torch.sum(x, dim=-1) / piece_len # (B, S, I_EMBED, L) -> (B, S, I_EMBED)
            x = x.view([B, -1]) # (B, S * I_EMBED)
            return x