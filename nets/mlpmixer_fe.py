import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

from einops import rearrange

from timm.models.layers import DropPath


class RelativePosition(nn.Module):

    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = torch.LongTensor(final_mat).cuda()
        embeddings = self.embeddings_table[final_mat].cuda()

        return embeddings


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.max_relative_position = 2

        self.relative_position_k = RelativePosition(self.head_dim, self.max_relative_position)
        self.relative_position_v = RelativePosition(self.head_dim, self.max_relative_position)

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))
        self.scale = self.head_dim ** -0.5
        
    def forward(self, query, key, value, mask = None):
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
        batch_size = query.shape[0]
        len_k = key.shape[1]
        len_q = query.shape[1]
        len_v = value.shape[1]

        query = self.fc_q(query)
        key = self.fc_k(key)
        value = self.fc_v(value)

        r_q1 = query.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        r_k1 = key.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        attn1 = torch.matmul(r_q1, r_k1.permute(0, 1, 3, 2)) 

        r_q2 = query.permute(1, 0, 2).contiguous().view(len_q, batch_size*self.n_heads, self.head_dim)
        r_k2 = self.relative_position_k(len_q, len_k)
        attn2 = torch.matmul(r_q2, r_k2.transpose(1, 2)).transpose(0, 1)
        attn2 = attn2.contiguous().view(batch_size, self.n_heads, len_q, len_k)
        attn = (attn1 + attn2) / self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e10)

        attn = self.dropout(torch.softmax(attn, dim = -1))

        #attn = [batch size, n heads, query len, key len]
        r_v1 = value.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        weight1 = torch.matmul(attn, r_v1)
        r_v2 = self.relative_position_v(len_q, len_v)
        weight2 = attn.permute(2, 0, 1, 3).contiguous().view(len_q, batch_size*self.n_heads, len_k)
        weight2 = torch.matmul(weight2, r_v2)
        weight2 = weight2.transpose(0, 1).contiguous().view(batch_size, self.n_heads, len_q, self.head_dim)

        x = weight1 + weight2 # x = [batch size, n heads, query len, head dim]
        
        x = x.permute(0, 2, 1, 3).contiguous() # x = [batch size, query len, n heads, head dim]
        
        x = x.view(batch_size, -1, self.hid_dim) # x = [batch size, query len, hid dim]
        
        x = self.fc_o(x) # x = [batch size, query len, hid dim]
        
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads):
        super(MultiHeadAttention, self).__init__()
        self.dim = dim
        self.heads = heads
        self.scale = (dim/heads) ** -0.5
        self.attend = nn.Softmax(dim=-1)

    def attend_with_rpe(self, Q, K):
        Q = rearrange(Q, 'b i (heads d) -> b heads i d', heads=self.heads)
        K = rearrange(K, 'b j (heads d) -> b heads j d', heads=self.heads)

        dots = einsum('bhid, bhjd -> bhij', Q, K) * self.scale # (b hw) heads 1 pointnum

        return self.attend(dots)

    def forward(self, Q, K, V):
        attn = self.attend_with_rpe(Q, K)
        B, HW, _ = Q.shape

        V = rearrange(V, 'b j (heads d) -> b heads j d', heads=self.heads)

        out = einsum('bhij, bhjd -> bhid', attn, V)
        out = rearrange(out, 'b heads hw d -> b hw (heads d)', b=B, hw=HW)

        return out


class CrossAttentionLayer(nn.Module):
    def __init__(self, qk_dim, v_dim, query_token_dim, tgt_token_dim, num_heads=8, attn_drop=0., proj_drop=0., drop_path=0., dropout=0.):
        super(CrossAttentionLayer, self).__init__()
        assert qk_dim % num_heads == 0, f"dim {qk_dim} should be divided by num_heads {num_heads}."
        assert v_dim % num_heads == 0, f"dim {v_dim} should be divided by num_heads {num_heads}."
        """
            Query Token:    [N, C]  -> [N, qk_dim]  (Q)
            Target Token:   [M, D]  -> [M, qk_dim]  (K),    [M, v_dim]  (V)
        """
        self.num_heads = num_heads
        head_dim = qk_dim // num_heads
        self.scale = head_dim ** -0.5

        self.norm1 = nn.LayerNorm(query_token_dim)
        self.norm2 = nn.LayerNorm(query_token_dim)
        # self.multi_head_attn = MultiHeadAttention(qk_dim, num_heads)
        self.multi_head_attn = MultiHeadAttentionLayer(qk_dim, num_heads, dropout)
        self.q, self.k, self.v = nn.Linear(query_token_dim, qk_dim, bias=True), nn.Linear(tgt_token_dim, qk_dim, bias=True), nn.Linear(tgt_token_dim, v_dim, bias=True)

        self.proj = nn.Linear(v_dim, query_token_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.ffn = nn.Sequential(
            nn.Linear(query_token_dim, query_token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(query_token_dim, query_token_dim),
            nn.Dropout(dropout)
        )

    def forward(self, query, tgt_token, weight):
        """
            x: [BH1W1, H3W3, D]
        """
        short_cut = query
        query = self.norm1(query)

        q, k, v = self.q(query), self.k(tgt_token), self.v(tgt_token)

        x = self.multi_head_attn(q, k, v)

        # x = short_cut + self.proj_drop(self.proj(x)) * weight
        x = self.proj_drop(self.proj(x))

        x = x + self.drop_path(self.ffn(self.norm2(x)))

        return x


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    # go to 0,1 then 0,2 then -1,1
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def fmaps_sample(coords, fmaps):
        r = 3
        H5 = W5 = 2*r+1
        B, S, N, D = coords.shape
        assert(D==2)

        B, S, C, H, W = fmaps.shape

        # fmaps = fmaps.unsqueeze(2).repeat(1, 1, N, 1, 1, 1) # B S N C H W

        dx = torch.linspace(-r, r, 2*r+1)
        dy = torch.linspace(-r, r, 2*r+1)
        delta = torch.stack(torch.meshgrid(dy, dx, indexing='ij'), axis=-1).to(coords.device) 

        centroid_lvl = coords.reshape(B*S, N, 1, 2)
        delta_lvl = delta.view(1, 1, H5 ** 2, 2)
        coords_lvl = centroid_lvl + delta_lvl

        out = bilinear_sampler(fmaps.reshape(B*S, C, H, W), coords_lvl)
        out = rearrange(out.float(), '(B S) C N (H5 W5) -> B S N H5 W5 C', B=B, H5=H5)
        # out = out.view(B, S, N, H5, W5, C).contiguous().float()

        return out


class SimplifiedFeatureEnhancerLayer(nn.Module):
    def __init__(self, feature_input_dim=128, feature_output_dim=128, dropout=0.0) -> None:
        super().__init__()

        query_token_dim, tgt_token_dim = feature_input_dim, feature_output_dim
        qk_dim, v_dim = query_token_dim, query_token_dim

        self.ffeats_updater = CrossAttentionLayer(qk_dim, v_dim, query_token_dim, tgt_token_dim, dropout=dropout)

    def forward(self, ffeats, fmaps, weight):
        ffeats = self.ffeats_updater(ffeats, fmaps, weight) # BS N C

        return ffeats, fmaps


class SimplifiedFeatureEnhancer(nn.Module):
    def __init__(self, depth=3, feature_input_dim=128) -> None:
        super().__init__()
        self.depth = depth
        self.layers = nn.ModuleList([SimplifiedFeatureEnhancerLayer() for i in range(self.depth)])
        self.weight = nn.Sequential(
            nn.Linear(feature_input_dim, feature_input_dim),
            nn.Sigmoid()
        )

    def forward(self, ffeats, fmaps):
        B, S, N, C = ffeats.shape
        ffeats = ffeats.unsqueeze(3) # B S N Q=1 C
        ffeats = rearrange(ffeats, 'B S N Q C -> (B S N) Q C')
        fmaps = rearrange(fmaps, 'B S N H5 W5 C -> (B S N) (H5 W5) C')
        weight = self.weight(ffeats) # BSN Q C

        for i in range(self.depth):
            ffeats, fmaps = self.layers[i](ffeats, fmaps, weight)

        ffeats = rearrange(ffeats, '(B S N) Q C -> B S N Q C', B=B, S=S)
        ffeats = ffeats.squeeze(3)

        return ffeats