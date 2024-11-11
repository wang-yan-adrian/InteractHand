import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
import numpy as np

class CTransformer(nn.Module):
    def __init__(self, inp_res=32, dim=256, depth=2, num_heads=4, mlp_ratio=4., injection=True):
        super().__init__()

        self.injection=injection
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, injection=injection))

        if self.injection:
            self.conv1 = nn.Sequential(
                nn.Conv2d(dim*2, dim, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(dim, dim, 3, padding=1),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(dim*2, dim, 1, padding=0),
            )

    def forward(self, query, key):
        output = query
        for i, layer in enumerate(self.layers):
            output = layer(query=output, key=key)
        
        if self.injection:
            output = torch.cat([key, output], dim=1)
            output = self.conv1(output) + self.conv2(output)

        return output

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self._init_weights()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)


class LePEAttention(nn.Module):
    def __init__(self, dim, resolution, idx, split_size=7, dim_out=None, num_heads=8, attn_drop=0., proj_drop=0., qk_scale=None):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        if idx == -1:
            H_sp, W_sp = self.resolution, self.resolution
        elif idx == 0:
            H_sp, W_sp = self.resolution, self.split_size
        elif idx == 1:
            W_sp, H_sp = self.resolution, self.split_size
        else:
            print ("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp
        stride = 1
        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,groups=dim)

        self.attn_drop = nn.Dropout(attn_drop)

    def im2cswin(self, x):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2,-1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp* self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def get_lepe(self, x, func):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2,-1).contiguous().view(B, C, H, W)

        H_sp, W_sp = self.H_sp, self.W_sp
        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp) ### B', C, H', W'

        lepe = func(x) ### B', C, H', W'
        lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()

        x = x.reshape(-1, self.num_heads, C // self.num_heads, self.H_sp* self.W_sp).permute(0, 1, 3, 2).contiguous()
        return x, lepe

    def forward(self, qkv):
        """
        x: B L C
        """
        q,k,v = qkv

        ### Img2Window
        H = W = self.resolution
        B, L, C = q.shape
        assert L == H * W, "flatten img_tokens has wrong size"
        
        q = self.im2cswin(q)
        k = self.im2cswin(k)
        v, lepe = self.get_lepe(v, self.get_v)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v) + lepe
        x = x.transpose(1, 2).reshape(-1, self.H_sp* self.W_sp, C)  # B head N N @ B head N C

        ### Window2Img
        x = windows2img(x, self.H_sp, self.W_sp, H, W).view(B, -1, C)  # B H' W' C

        return x
def img2windows(img, H_sp, W_sp):
    """
    img: B C H W
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp* W_sp, C)
    return img_perm

def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    img_splits_hw: B' H W C
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img
class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., act_layer=nn.GELU, norm_layer=nn.LayerNorm, injection=True):
        super().__init__()

        self.injection = injection

        self.channels = dim

        self.encode_value = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1, padding=0)
        self.encode_query = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1, padding=0)
        self.encode_key = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1, padding=0)

        if self.injection:
            self.encode_query2 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1, padding=0)
            self.encode_key2 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1, padding=0)

        self.attn = LePEAttention(dim, num_heads=num_heads, resolution=32, idx=1, split_size=2, dim_out=dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.q_embedding = nn.Parameter(torch.randn(1, 256, 32, 32))
        self.k_embedding = nn.Parameter(torch.randn(1, 256, 32, 32))

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, query, key, query_embed=None, key_embed=None):
        b, c, h, w = query.shape
        # 就是沿着批次维度复制b次而(n,c,b)保持不变，这样在后续进行位置嵌入的时候，可以匹配上数据的维度
        query_embed = repeat(self.q_embedding, '() n c d -> b n c d', b = b)
        key_embed = repeat(self.k_embedding, '() n c d -> b n c d', b = b)

        q_embed = self.with_pos_embed(query, query_embed)
        k_embed = self.with_pos_embed(key, key_embed)

        v = self.encode_value(key).view(b, self.channels, -1)
        v = v.permute(0, 2, 1)

        q = self.encode_query(q_embed).view(b, self.channels, -1)
        q = q.permute(0, 2, 1)

        k = self.encode_key(k_embed).view(b, self.channels, -1)
        k = k.permute(0, 2, 1)
        
        query = query.view(b, self.channels, -1).permute(0, 2, 1)

        if self.injection:
            q2 = self.encode_query2(q_embed).view(b, self.channels, -1)
            q2 = q2.permute(0, 2, 1)

            k2 = self.encode_key2(k_embed).view(b, self.channels, -1)
            k2 = k2.permute(0, 2, 1)

            query = self.attn(query=q, key=k, value=v,query2 = q2, key2 = k2, use_sigmoid=True)
        else:
            q2 = None
            k2 = None
            qkv = (q, k, v)

            query = query + self.attn(qkv)
 
        # 感觉这里像是残差连接
        query = query + self.mlp(self.norm2(query))
        # query(batch_size,N,C)--(B,C,N)permute将N置于最后，为后续的view做准备
        # contiguous()确保张量在内存中是连续存储的，在某些操作之后，张量额可以变得非连续存储以节省内存，然后某些操作如view要求张量必须是连续的
        query = query.permute(0, 2, 1).contiguous().view(b, self.channels, h, w)

        return query
