import torch
from torch import nn, einsum
import torch.nn.functional as F
from functools import partial
from inspect import isfunction

# helpers

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else (val,) * depth

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

def top_k(logits, thres = 0.9):
    k = int((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

# keyword argument helpers

def pick_and_pop(keys, d):
    values = list(map(lambda key: d.pop(key), keys))
    return dict(zip(keys, values))

def group_dict_by_key(cond, d):
    return_val = [dict(),dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)

def string_begins_with(prefix, str):
    return str.startswith(prefix)

def group_by_key_prefix(prefix, d):
    return group_dict_by_key(partial(string_begins_with, prefix), d)

# classes

class Scale(nn.Module):
    def __init__(self, value, fn):
        super().__init__()
        self.value = value
        self.fn = fn

    def forward(self, x, **kwargs):
        x, *rest = self.fn(x, **kwargs)
        return (x * self.value, *rest)

class Rezero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.zeros(1))

    def forward(self, x, **kwargs):
        x, *rest = self.fn(x, **kwargs)
        return (x * self.g, *rest)

class ScaleNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        norm = torch.norm(x, dim = -1, keepdim = True) * self.scale
        return x / norm.clamp(min = self.eps) * self.g

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# feedforward

class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim = -1)
        return x * F.gelu(gate)

class FeedForward(nn.Module):
    def __init__(self, dim, dim_out = None, mult = 4, glu = False, dropout = 0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)

# attention.

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        causal = False,
        num_mem_kv = 0,
        talking_heads = False,
        sparse_topk = 0,
        use_entmax15 = False,
        dropout = 0.,
        on_attn = False
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.causal = causal
        self.num_mem_kv = num_mem_kv
        self.to_mem_k = nn.Parameter(torch.randn(num_mem_kv, inner_dim))
        self.to_mem_v = nn.Parameter(torch.randn(num_mem_kv, inner_dim))

        self.talking_heads = talking_heads
        if talking_heads:
            self.pre_softmax_proj = nn.Parameter(torch.randn(heads, heads))
            self.post_softmax_proj = nn.Parameter(torch.randn(heads, heads))

        self.sparse_topk = sparse_topk

        self.attn_fn = F.softmax
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.on_attn = on_attn
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context = None, mask = None, context_mask = None, rel_pos = None, sinusoids = None):
        b, n, _, h, device = *x.shape, self.heads, x.device
        kv_input = default(context, x)

        q = self.to_q(x)
        k, v = self.to_kv(kv_input).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        if self.num_mem_kv > 0:
            mk, mv = map(lambda t: repeat(t, 'n d -> b h n d', b = b, h = h), (self.to_mem_k, self.to_mem_v))
            k = torch.cat((mk, k), dim = 2)
            v = torch.cat((mv, v), dim = 2)

        content_q = q if not self.on_attn else k
        dots = einsum('b h i d, b h j d -> b h i j', content_q, k) * self.scale

        if exists(sinusoids):
            sinusoid_q, sinusoid_k = sinusoids
            sinusoid_q, sinusoid_k = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (sinusoid_q, sinusoid_k))
            dots = dots + einsum('b h i d, b h j d -> b h i j', sinusoid_q, sinusoid_k)

        if exists(rel_pos):
            dots = dots + rel_pos

        if self.talking_heads:
            dots = einsum('b h i j, h k -> b k i j', dots, self.pre_softmax_proj)

        mask_value = max_neg_value(dots)

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            dots.masked_fill_(~mask, mask_value)

        if exists(context_mask):
            context_mask = rearrange(context_mask, 'b j -> b 1 1 j')
            dots.masked_fill_(~context_mask, mask_value)

        if self.causal:
            i, j = dots.shape[-2:]
            mask = torch.ones(i, j, device = device).triu_(j - i + 1).bool()
            dots.masked_fill_(mask, mask_value)

        if self.sparse_topk > 0:
            topk = self.sparse_topk
            topk_dots, _ = dots.topk(topk, dim = -1)
            dots.masked_fill_(dots < topk_dots[..., -1:], mask_value)

        attn = self.attn_fn(dots, dim = -1)

        if self.talking_heads:
            attn = einsum('b h i j, h k -> b k i j', attn, self.post_softmax_proj)

        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x):
        t = torch.arange(x.shape[1], device = x.device)
        return self.emb(t)

class FixedPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        position = torch.arange(0, max_seq_len, dtype=torch.float)
        sinusoid_inp = torch.einsum("i,j->ij", position, inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        self.register_buffer('emb', emb)

    def forward(self, x):
        return self.emb[None, :x.shape[1], :].to(x)

class Encoder(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads = 8,
        causal = False,
        num_mem_kv = 0,
        rel_pos_emb = False,
        rezero = False,
        talking_heads = False,
        sparse_topk = 0,
        use_entmax15 = False,
        num_global_tokens = 1,
        gate_residual = False,
        attn_dropout = 0.,
        ff_dropout = 0.,
        ff_glu = False
    ):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([])
        self.global_tokens = None

        if num_global_tokens > 0:
            self.global_tokens = nn.Parameter(torch.randn(num_global_tokens, dim))

        for _ in range(depth):
            attn = Attention(dim, heads = heads, causal = causal, num_mem_kv = num_mem_kv, talking_heads = talking_heads, sparse_topk = sparse_topk, use_entmax15 = use_entmax15, dropout = attn_dropout)
            ff = FeedForward(dim, glu = ff_glu, dropout = ff_dropout)

            if rezero:
                attn, ff = map(Rezero, (attn, ff))

            self.layers.append(nn.ModuleList([
                PreNorm(dim, attn),
                PreNorm(dim, ff)
            ]))

    def forward(self, x, context = None, mask = None, context_mask = None, rel_pos = None, sinusoids = None):
        if exists(self.global_tokens):
            global_tokens = repeat(self.global_tokens, 'n d -> b n d', b = x.shape[0])
            x = torch.cat((global_tokens, x), dim = 1)

        for attn, ff in self.layers:
            x = attn(x, context = context, mask = mask, context_mask = context_mask, rel_pos = rel_pos, sinusoids = sinusoids) + x
            x = ff(x) + x

        if exists(self.global_tokens):
            x = x[:, self.global_tokens.shape[0]:]

        return x

class Decoder(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads = 8,
        num_mem_kv = 0,
        rel_pos_emb = False,
        rezero = False,
        talking_heads = False,
        sparse_topk = 0,
        use_entmax15 = False,
        attn_dropout = 0.,
        ff_dropout = 0.,
        ff_glu = False,
        cross_attend = False
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            attn = Attention(dim, heads = heads, causal = True, num_mem_kv = num_mem_kv, talking_heads = talking_heads, sparse_topk = sparse_topk, use_entmax15 = use_entmax15, dropout = attn_dropout)
            ff = FeedForward(dim, glu = ff_glu, dropout = ff_dropout)

            if rezero:
                attn, ff = map(Rezero, (attn, ff))

            if cross_attend:
                cross_attn = Attention(dim, heads = heads, causal = False, num_mem_kv = num_mem_kv, talking_heads = talking_heads, sparse_topk = sparse_topk, use_entmax15 = use_entmax15, dropout = attn_dropout)
                if rezero:
                    cross_attn = Rezero(cross_attn)
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, attn),
                    PreNorm(dim, cross_attn),
                    PreNorm(dim, ff)
                ]))
            else:
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, attn),
                    PreNorm(dim, ff)
                ]))

    def forward(self, x, context = None, mask = None, context_mask = None, rel_pos = None, sinusoids = None):
        for layer in self.layers:
            if len(layer) == 3:
                attn, cross_attn, ff = layer
                x = attn(x, mask = mask, rel_pos = rel_pos, sinusoids = sinusoids) + x
                x = cross_attn(x, context = context, mask = mask, context_mask = context_mask, rel_pos = rel_pos, sinusoids = sinusoids) + x
            else:
                attn, ff = layer
                x = attn(x, mask = mask, rel_pos = rel_pos, sinusoids = sinusoids) + x
            x = ff(x) + x
        return x

class TransformerWrapper(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        max_seq_len,
        attn_layers,
        emb_dim = None,
        max_mem_len = 0,
        emb_dropout = 0.,
        num_memory_tokens = None,
        tie_embedding = False,
        use_pos_emb = True
    ):
        super().__init__()
        assert isinstance(attn_layers, Encoder) or isinstance(attn_layers, Decoder), 'attn_layers must be an instance of Encoder or Decoder'

        dim = attn_layers.dim
        emb_dim = default(emb_dim, dim)

        self.max_seq_len = max_seq_len
        self.max_mem_len = max_mem_len
        self.num_memory_tokens = num_memory_tokens

        self.token_emb = nn.Embedding(num_tokens, emb_dim)
        self.pos_emb = AbsolutePositionalEmbedding(emb_dim, max_seq_len) if (use_pos_emb and not isinstance(attn_layers, Decoder)) else None
        self.emb_dropout = nn.Dropout(emb_dropout)

        self.project_emb = nn.Linear(emb_dim, dim) if emb_dim != dim else nn.Identity()
        self.attn_layers = attn_layers
        self.norm = nn.LayerNorm(dim)

        self.init_ = nn.Parameter(torch.randn(1, 1, dim)) if isinstance(attn_layers, Decoder) else None
        self.to_logits = nn.Linear(dim, num_tokens) if not tie_embedding else lambda t: t @ self.token_emb.weight.t()
        
        self.memory_tokens = None
        if num_memory_tokens is not None:
            self.memory_tokens = nn.Parameter(torch.randn(num_memory_tokens, dim))

    def forward(self, x, return_embeddings = False, mask = None):
        x = self.token_emb(x)
        if exists(self.pos_emb):
            x = x + self.pos_emb(x)

        x = self.emb_dropout(x)
        x = self.project_emb(x)

        if exists(self.memory_tokens):
            m = repeat(self.memory_tokens, 'n d -> b n d', b = x.shape[0])
            x = torch.cat((x, m), dim = 1)
            if exists(mask):
                mask = F.pad(mask, (0, self.num_memory_tokens), value = True)

        x = self.attn_layers(x, mask = mask)
        x = self.norm(x)

        if return_embeddings:
            return x

        return self.to_logits(x)

from einops import rearrange, repeat