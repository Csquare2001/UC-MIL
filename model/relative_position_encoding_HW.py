import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import pdb

class RelativePositionEmbedding(nn.Module):

    # input-dependent relative position
    def __init__(self, dim, shape):
        super().__init__()

        self.dim = dim
        self.shape = shape

        self.key_rel_w = nn.Parameter(torch.randn((2 * self.shape - 1, dim)) * 0.02)
        self.key_rel_h = nn.Parameter(torch.randn((2 * self.shape - 1, dim)) * 0.02)

        coords = torch.arange(self.shape)
        relative_coords = coords[None, :] - coords[:, None]  # h, h
        relative_coords += self.shape - 1  # shift to start from 0

        self.register_buffer('relative_position_index', relative_coords)

    def forward(self, q, Nh, H, W, dim_head):
        # q: B, Nh, HW, dim
        q = q[:,:,1:,:]
        B, _, _, dim = q.shape

        # q: B, Nh, H, W, dim_head
        q = rearrange(q, 'b heads (h w) dim_head -> b heads h w dim_head', b=B, dim_head=dim_head, heads=Nh, h=H, w=W)

        rel_logits_w = self.relative_logits_1d(q, self.key_rel_w, 'w')

        rel_logits_h = self.relative_logits_1d(q.permute(0, 1, 3, 2, 4), self.key_rel_h, 'h')

        return rel_logits_w, rel_logits_h

    def relative_logits_1d(self, q, rel_k, case):

        B, Nh, H, W, dim = q.shape

        rel_logits = torch.einsum('bhxyd,md->bhxym', q, rel_k)  # B, Nh, H, W, 2*shape-1
        if case == 'w':
            if W != self.shape:
                relative_index = torch.repeat_interleave(self.relative_position_index, W // self.shape,   dim=0)  # W, shape
            else:
                relative_index = self.relative_position_index
            relative_index = relative_index.view(1, 1, 1, W, self.shape)


        elif case == 'h':
            if H != self.shape:
                relative_index = torch.repeat_interleave(self.relative_position_index, H // self.shape,   dim=0)  # H, shape
            else:
                relative_index = self.relative_position_index
            relative_index = relative_index.view(1, 1, 1, H, self.shape)

        relative_index = relative_index.repeat(B, Nh, H, 1, 1)
        rel_logits = torch.gather(rel_logits, 4, relative_index)  # B, Nh, H, W, shape
        rel_logits = rel_logits.unsqueeze(3)
        rel_logits = rel_logits.repeat(1, 1, 1, self.shape, 1, 1)
        rel_logits = rearrange(rel_logits, 'b heads H h W w -> b heads (H W) (h w)')
        rel_logits = torch.nn.functional.pad(rel_logits, (1, 0, 1, 0))
        return rel_logits
if __name__ == "__main__":
    model = RelativePositionEmbedding(64,14)
    a = torch.randn(4,12,197,64)
    b,c = model(a,12,14,14,64)
    print(b.shape)
