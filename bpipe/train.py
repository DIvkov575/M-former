import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformer import TransformerModel
import glob
from tqdm import tqdm
import torch.nn.utils.rnn as rnn_utils

def load_data(data_dir):
    msas = []
    for f in glob.glob(f"{data_dir}/*.npz"):
        data = np.load(f, allow_pickle=True)
        sequences_data = data['sequences']
        residues_data = data['residues']['res_type']
        
        for seq_info in sequences_data:
            res_start, res_end = seq_info['res_start'], seq_info['res_end']
            sequence = residues_data[res_start:res_end]
            msas.append(torch.from_numpy(sequence).long())
    return msas

class ProteinDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    batch = rnn_utils.pad_sequence(batch, batch_first=True, padding_value=0)
    return batch

def train():
    # Model parameters
    ntokens = 23 # 21 amino acids + 1 gap + 1 mask
    ninp = 256 # embedding dimension
    nhead = 8 # number of heads in the multiheadattention models
    nhid = 512 # dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 6 # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    dropout = 0.1

    # Training parameters
    batch_size = 16
    epochs = 10
    lr = 0.001
    mask_prob = 0.15

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TransformerModel(ntokens, ninp, nhead, nhid, nlayers, dropout).to(device)
    model = torch.compile(model) # PyTorch 2.0+ feature for JIT compilation
    criterion = nn.CrossEntropyLoss(ignore_index=0) # Ignore padding index
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    data = load_data('data')
    dataset = ProteinDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch in progress_bar:
            batch = batch.to(device)
            
            # Create mask
            mask = torch.rand(batch.shape, device=device) < mask_prob
            # Ensure that we don't mask padding tokens
            mask = mask & (batch != 0)
            
            masked_seq = batch.clone()
            masked_seq[mask] = ntokens - 1 # use a special token for masked positions
            targets = batch.clone()

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                output = model(masked_seq)
                loss = criterion(output.view(-1, ntokens), targets.view(-1))
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'Loss': total_loss / (progress_bar.n + 1)})

if __name__ == '__main__':
    train()