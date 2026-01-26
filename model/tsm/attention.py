import torch
from torch import nn

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.uniform_(m.weight, -0.1, 0.1)
        m.bias.data.fill_(0.01)

class SelfAttention(nn.Module):
    def __init__(self, 
                 attention_size,
                 batch_first=False,
                 layers=1,
                 dropout=.0,
                 non_linearity="tanh"):
        super(SelfAttention, self).__init__()

        self.batch_first = batch_first

        if non_linearity == "relu":
            activation = nn.ReLU()
        else:
            activation = nn.Tanh()

        modules = []
        for i in range(layers - 1):
            modules.append(nn.Linear(attention_size, attention_size))
            modules.append(activation)
            modules.append(nn.Dropout(dropout))

        # last attention layer must output a scalar
        modules.append(nn.Linear(attention_size, 1))
        modules.append(activation)
        modules.append(nn.Dropout(dropout))

        self.attention = nn.Sequential(*modules)
        self.attention.apply(init_weights) 
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, inputs):

        ##################################################################
        # STEP 1 - perform dot product
        # of the attention vector and each hidden state
        ##################################################################

        # inputs is a 3D Tensor: batch, len, hidden_size
        # scores is a 2D Tensor: batch, len
        scores = self.attention(inputs).squeeze()
        scores = self.softmax(scores)

        ##################################################################
        # Step 2 - Weighted sum of hidden states, by the attention scores
        ##################################################################

        # multiply each hidden state with the attention weights
        weighted = torch.mul(inputs, scores.unsqueeze(-1).expand_as(inputs))

        # sum the hidden states
        # representations = weighted.sum(1).squeeze()
        representations = weighted.sum(1).squeeze()
        return representations, scores

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        """
        d_model: 词向量/隐藏维度大小
        n_heads: Multi-Head 注意力头数 一般是 8
        d_ff   : 前馈网络中间维度 (一般是 4 * d_model)
        dropout: dropout 概率
        """
        super().__init__()

        # Multi-Head Self-Attention
        # batch_first=True => 输入输出形状都是 [batch, seq_len, d_model]
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        # 前馈网络 FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),          # 如果想和 Transformer 原版更像，可以换成 GELU
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

        # 两个 LayerNorm + Dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x,
        attn_mask=None,
        key_padding_mask=None
    ):
        """
        x: [batch, seq_len, d_model]
        attn_mask: [seq_len, seq_len] 或 [batch*n_heads, seq_len, seq_len]
                   用于做因果 mask 或其它自定义 mask
        key_padding_mask: [batch, seq_len]，为 True 的位置表示 padding，需要被 mask 掉
        """

        # ====== 1. Multi-Head Self-Attention 子层 + 残差 + LayerNorm ======
        # 注意：MultiheadAttention 的输入为 (query, key, value)，这里都是 x（自注意力）
        attn_output, attn_weights = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False  # 如果你不关心注意力权重可以关掉，节省一点开销
        )
        x = x + self.dropout(attn_output)  # 残差连接
        x = self.norm1(x)

        # ====== 2. 前馈网络 FFN 子层 + 残差 + LayerNorm ======
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)   # 残差连接
        x = self.norm2(x)
        # x = x.mean(dim=1)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, num_layers=4, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = x.mean(dim=1)  # 最后再做 pooling
        return x  # [batch, d_model]
