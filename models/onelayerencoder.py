import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def modified_scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)

    # no softmax in the attention
    # attention = F.softmax(attn_logits, dim=-1)
    attention = attn_logits
    values = torch.matmul(attention, v)
    return values, attention

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)

    # softmax in the attention
    attention = F.softmax(attn_logits, dim=-1)
    attention = attn_logits
    values = torch.matmul(attention, v)
    return values, attention


class ModifiedMultiheadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim, bias=False)
        # self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        # self.qkv_proj.bias.data.fill_(0)
        # nn.init.xavier_uniform_(self.o_proj.weight)
        # self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = modified_scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        # o = self.o_proj(values)
        o = values

        if return_attention:
            return o, attention
        else:
            return o

class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim, bias=False)
        # self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        # self.qkv_proj.bias.data.fill_(0)
        # nn.init.xavier_uniform_(self.o_proj.weight)
        # self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        # o = self.o_proj(values)
        o = values

        if return_attention:
            return o, attention
        else:
            return o

class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm, self).__init__()

    def forward(self, input_tensor):
        l2_norm = torch.norm(input_tensor, p=2, dim=2, keepdim=True)
        return l2_norm

class NormLastColumn(nn.Module):
    def __init__(self):
        super(NormLastColumn, self).__init__()

    def forward(self, input_matrix):
        norm_values = torch.norm(input_matrix, p=2, dim=2, keepdim=True).squeeze()
        output_matrix = torch.zeros_like(input_matrix)
        # print(output_matrix.size())
        # print(norm_values.size())
        output_matrix[:,:,-1] = norm_values
        return output_matrix

class MLP(nn.Module):
    def __init__(self, input_dim, dim_feedforward, dropout=0.0):
        super().__init__()

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # MLP part
        linear_out = self.linear_net(x)
        return linear_out
    
class OnelayerEncoder(nn.Module):
    def __init__(self, self_attn, linear_net, args):
        """
        Args:
            input_dim: Dimensionality of the input
            num_heads: Number of heads to use in the attention block
            dim_feedforward: Dimensionality of the hidden layer in the MLP
            dropout: Dropout probability to use in the dropout layers
        """
        super().__init__()
        # Attention layer
        self.self_attn = self_attn
        self.linear_net = linear_net
        self.norm1 = nn.LayerNorm(args.input_dim)
        self.norm2 = nn.LayerNorm(args.input_dim)


    def forward(self, x, args, mask=None):
        # Attention part
        x = self.self_attn(x, mask=mask)
        if args.residual:
            x = x + self.dropout(x)
        if args.layer_norm:
            x = self.norm1(x)

        # MLP part
        x = self.linear_net(x)
        if args.residual:
            x = x + self.dropout(x)
        if args.layer_norm:
            x = self.norm1(x)

        return x
    
class OnelayerEncoderNoResidualNoLayerNorm(nn.Module):
    def __init__(self, self_attn, linear_net, args):
        """
        Args:
            input_dim: Dimensionality of the input
            num_heads: Number of heads to use in the attention block
            dim_feedforward: Dimensionality of the hidden layer in the MLP
            dropout: Dropout probability to use in the dropout layers
        """
        super().__init__()
        # Attention layer
        self.self_attn = self_attn
        self.linear_net = linear_net
        self.norm1 = nn.LayerNorm(args.input_dim)
        self.norm2 = nn.LayerNorm(args.input_dim)

    def forward(self, x, mask=None):
        # Attention part
        x = self.self_attn(x, mask=mask)
        # MLP part
        x = self.linear_net(x)
        return x
    
class OnelayerEncoderNoResidual(nn.Module):
    def __init__(self, self_attn, linear_net, args):
        """
        Args:
            input_dim: Dimensionality of the input
            num_heads: Number of heads to use in the attention block
            dim_feedforward: Dimensionality of the hidden layer in the MLP
            dropout: Dropout probability to use in the dropout layers
        """
        super().__init__()
        # Attention layer
        self.self_attn = self_attn
        self.linear_net = linear_net
        self.norm1 = nn.LayerNorm(args.input_dim)
        self.norm2 = nn.LayerNorm(args.input_dim)

    def forward(self, x, mask=None):
        # Attention part
        attn_out = self.self_attn(x, mask=mask)
        x = self.norm1(attn_out)
        # MLP part
        linear_out = self.linear_net(x)
        x = self.norm2(linear_out)
        return x
    
class OnelayerEncoderNoLayerNorm(nn.Module):
    def __init__(self, self_attn, linear_net, args):
        """
        Args:
            input_dim: Dimensionality of the input
            num_heads: Number of heads to use in the attention block
            dim_feedforward: Dimensionality of the hidden layer in the MLP
            dropout: Dropout probability to use in the dropout layers
        """
        super().__init__()
        # Attention layer
        self.self_attn = self_attn
        self.linear_net = linear_net
        self.norm1 = nn.LayerNorm(args.input_dim)
        self.norm2 = nn.LayerNorm(args.input_dim)

    def forward(self, x, mask=None):
        # Attention part
        attn_out = self.self_attn(x, mask=mask)
        x = x + self.dropout(attn_out)
        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        return x    