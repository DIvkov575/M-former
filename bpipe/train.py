
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from inter_intra_transformer import InterIntraTransformer
import glob
from tqdm import tqdm
import torch.nn.utils.rnn as rnn_utils

def load_data(data_dir):
    msas = []
    for f in glob.glob(f"{data_dir}/*.npz"):
        data = np.load(f, allow_pickle=True)
        sequences_data = data['sequences']
        residues_data = data['residues']['res_type']
        
        msa_sequences = []
        for seq_info in sequences_data:
            res_start, res_end = seq_info['res_start'], seq_info['res_end']
            sequence = residues_data[res_start:res_end]
            msa_sequences.append(torch.from_numpy(sequence).long())
        
        if not msa_sequences:
            continue

        msa = rnn_utils.pad_sequence(msa_sequences, batch_first=True, padding_value=0)
        msas.append(msa)
    return msas

class ProteinMSADataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    # batch is a list of MSAs (2D tensors)
    max_num_seqs = max(msa.shape[0] for msa in batch)
    max_seq_len = max(msa.shape[1] for msa in batch)
    
    padded_batch = torch.zeros(len(batch), max_num_seqs, max_seq_len).long()
    
    for i, msa in enumerate(batch):
        n, s = msa.shape
        padded_batch[i, :n, :s] = msa
        
    return padded_batch

def train():
    # Model parameters
    ntokens = 23 # 21 amino acids + 1 gap + 1 mask
    d_model = 128 # embedding dimension
    nhead = 4 # number of heads
    d_ff = 256 # feedforward dimension
    nlayers = 4 # number of layers
    dropout = 0.1

    # Training parameters
    batch_size = 4 # smaller batch size due to larger model and data
    epochs = 10
    lr = 0.001
    mask_prob = 0.15

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = InterIntraTransformer(ntokens, d_model, nhead, d_ff, nlayers, dropout).to(device)
    model = torch.compile(model)
    criterion = nn.CrossEntropyLoss(ignore_index=0) # Ignore padding index
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    data = load_data('data')
    dataset = Prot teinMSADataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch in progress_bar:
            batch = batch.to(device)
            
            # Create mask for MLM
            mask = torch.rand(batch.shape, device=device) < mask_prob
            mask = mask & (batch != 0) # Do not mask padding tokens
            
            masked_msa = batch.clone()
            masked_msa[mask] = ntokens - 1 # Use a special token for masked positions
            targets = batch.clone()

            # Create padding mask for the attention
            src_key_padding_mask = (masked_msa == 0)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                output = model(masked_msa, src_key_padding_mask=src_key_padding_mask)
                loss = criterion(output.view(-1, ntokens), targets.view(-1))
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'Loss': total_loss / (progress_bar.n + 1)})

if __name__ == '__main__':
    train()
