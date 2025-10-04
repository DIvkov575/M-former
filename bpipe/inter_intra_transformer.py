
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x is (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class InterIntraSequenceAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

    def forward(self, x, src_key_padding_mask=None):
        # x is (batch, num_seqs, seq_len, d_model)
        b, n, s, d = x.shape
        
        # Flatten num_seqs and seq_len dimensions
        x_flat = x.view(b, n * s, d)
        
        # Prepare mask
        if src_key_padding_mask is not None:
            # src_key_padding_mask is (batch, num_seqs, seq_len)
            mask = src_key_padding_mask.view(b, n * s)
        else:
            mask = None
            
        attn_output, _ = self.attn(x_flat, x_flat, x_flat, key_padding_mask=mask)
        
        return attn_output.view(b, n, s, d)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return x

class InterIntraTransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super().__init__()
        self.attn = InterIntraSequenceAttention(d_model, nhead, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, src_key_padding_mask=None):
        # x is (batch, num_seqs, seq_len, d_model)
        x = x + self.dropout1(self.attn(self.norm1(x), src_key_padding_mask=src_key_padding_mask))
        x = x + self.dropout2(self.ff(self.norm2(x)))
        return x

class InterIntraTransformer(nn.Module):
    def __init__(self, ntoken, d_model, nhead, d_ff, nlayers, dropout=0.1):
        super().__init__()
        self.encoder = nn.Embedding(ntoken, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        self.blocks = nn.ModuleList([
            InterIntraTransformerBlock(d_model, nhead, d_ff, dropout) for _ in range(nlayers)
        ])
        
        self.decoder = nn.Linear(d_model, ntoken)
        self.d_model = d_model

    def forward(self, src, src_key_padding_mask=None):
        # src is (batch, num_seqs, seq_len)
        x = self.encoder(src) * math.sqrt(self.d_model)
        
        # Add positional encoding to each sequence in the MSA
        b, n, s, d = x.shape
        x_pos = x.view(b * n, s, d)
        x_pos = self.pos_encoder(x_pos)
        x = x_pos.view(b, n, s, d)

        for block in self.blocks:
            x = block(x, src_key_padding_mask=src_key_padding_mask)
            
        return self.decoder(x)
