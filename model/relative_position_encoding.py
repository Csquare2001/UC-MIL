import torch
import torch.nn as nn
class RelativePositionBias(nn.Module):
    # input-independent relative position attention
    # As the number of parameters is smaller, so use 2D here
    def __init__(self, num_heads, h, w):  # (4,16,16)
        super().__init__()
        self.num_heads = num_heads  # 4
        self.h = h  # 16
        self.w = w  # 16

        self.relative_position_bias_table = nn.Parameter(
            torch.randn((2 * h - 1) * (2 * w - 1), num_heads) * 0.02)  # (961,4)

        coords_h = torch.arange(self.h)  # [0,16]
        coords_w = torch.arange(self.w)  # [0,16]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # (2, 16, 16)
        coords_flatten = torch.flatten(coords, 1)  # (2, 256)

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2,256,256)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (256,256,2)
        # 转换到大于0
        relative_coords[:, :, 0] += self.h - 1  # (256,256,2)
        relative_coords[:, :, 1] += self.w - 1
        relative_coords[:, :, 0] *= 2 * self.h - 1
        # 二维转换到一维
        relative_position_index = relative_coords.sum(-1)  # (256, 256)

        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, H, W):
        # relative_position_index->(256,256)
        # relative_position_bias_table->(961,4)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.h,self.w ,self.h * self.w ,-1)  # h, w, hw, nH (16,16,256,4)

        relative_position_bias_expand_h = torch.repeat_interleave(relative_position_bias, H // self.h ,dim=0)  # (在dim=0维度重复7次)->(112,16,256,4)

        relative_position_bias_expanded = torch.repeat_interleave(relative_position_bias_expand_h, W // self.w  ,dim=1)  # HW, hw, nH #(在dim=1维度重复7次)

        relative_position_bias_expanded = relative_position_bias_expanded.view(H * W, self.h * self.w, self.num_heads).permute(2, 0  ,1).contiguous().unsqueeze(0)
        # 进行填充操作
        relative_position_bias_expanded = torch.nn.functional.pad(relative_position_bias_expanded, (1, 0, 1, 0))

        return relative_position_bias_expanded
if __name__=="__main__":
    model = RelativePositionBias(8,14,14)
    c = model(14,14)
    print(c.size())