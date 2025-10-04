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

class IntraSequenceAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

    def forward(self, x, src_key_padding_mask=None):
        # x is (batch, num_seqs, seq_len, d_model)
        b, n, s, d = x.shape
        
        # Reshape for attention over sequence length (intra-sequence)
        x_flat = x.view(b * n, s, d)
        
        # Prepare mask for flattened sequences
        mask_flat = None
        if src_key_padding_mask is not None:
            mask_flat = src_key_padding_mask.view(b * n, s)
            
        attn_output, _ = self.attn(x_flat, x_flat, x_flat, key_padding_mask=mask_flat)
        
        return attn_output.view(b, n, s, d)

class InterSequenceAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

    def forward(self, x, src_key_padding_mask=None):
        # x is (batch, num_seqs, seq_len, d_model)
        b, n, s, d = x.shape
        
        # Permute and reshape for attention over number of sequences (inter-sequence)
        x_permuted = x.permute(0, 2, 1, 3).contiguous() # -> (b, s, n, d)
        x_flat = x_permuted.view(b * s, n, d)
        
        # Prepare mask for flattened sequences
        mask_flat = None
        if src_key_padding_mask is not None:
            # The mask needs to be applied to the num_seqs dimension
            # Original mask: (batch, num_seqs, seq_len)
            # We need to create a mask of shape (b * s, n)
            # This means for each residue position (s), we need to mask sequences (n)
            # If a residue is padded in the original mask, then all sequences at that position are masked.
            # This is a bit tricky. For simplicity, let's assume src_key_padding_mask is for residues.
            # If a residue is masked, it means it's a padding token. We don't want to attend to it.
            # For inter-sequence attention, we are attending across sequences for a given residue position.
            # So, if a residue position is padded, all sequences at that position should be masked.
            # Let's assume src_key_padding_mask is (batch, num_seqs, seq_len) where True means padded.
            # For inter-sequence attention, we are essentially doing attention for each (batch, seq_len) pair over num_seqs.
            # So, the mask should be (batch * seq_len, num_seqs)
            # If src_key_padding_mask[b, n_idx, s_idx] is True, then sequence n_idx at position s_idx is padded.
            # When we do inter-sequence attention, for a given s_idx, we are attending over n_idx.
            # So, the mask should indicate which sequences are padded at that specific s_idx.
            # This means we need to transpose the mask as well.
            mask_flat = src_key_padding_mask.permute(0, 2, 1).contiguous().view(b * s, n)
            
        attn_output, _ = self.attn(x_flat, x_flat, x_flat, key_padding_mask=mask_flat)
        
        attn_output = attn_output.view(b, s, n, d)
        return attn_output.permute(0, 2, 1, 3).contiguous() # -> (b, n, s, d)

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

class SophisticatedTransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super().__init__()
        self.intra_attn = IntraSequenceAttention(d_model, nhead, dropout)
        self.inter_attn = InterSequenceAttention(d_model, nhead, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, src_key_padding_mask=None):
        # x is (batch, num_seqs, seq_len, d_model)
        # Apply intra-sequence attention
        x = x + self.dropout1(self.intra_attn(self.norm1(x), src_key_padding_mask=src_key_padding_mask))
        
        # Apply inter-sequence attention
        x = x + self.dropout2(self.inter_attn(self.norm2(x), src_key_padding_mask=src_key_padding_mask))
        
        # Apply feed-forward
        x = x + self.dropout3(self.ff(self.norm3(x)))
        return x

class SophisticatedTransformer(nn.Module):
    def __init__(self, ntoken, d_model, nhead, d_ff, nlayers, dropout=0.1):
        super().__init__()
        self.encoder = nn.Embedding(ntoken, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        self.blocks = nn.ModuleList([
            SophisticatedTransformerBlock(d_model, nhead, d_ff, dropout) for _ in range(nlayers)
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
