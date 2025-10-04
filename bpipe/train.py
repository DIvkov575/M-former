
import torch
import torch.nn as nn
import numpy as np
from transformer import TransformerModel
import glob
from tqdm import tqdm

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
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    data = load_data('data')

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(range(0, len(data), batch_size), desc=f'Epoch {epoch+1}/{epochs}')
        for i in progress_bar:
            batch = data[i:i+batch_size]
            # This is a simplified batching, proper batching would require padding
            for seq in batch:
                seq = seq.to(device)
                
                # Create mask
                mask = torch.rand(seq.shape) < mask_prob
                masked_seq = seq.clone()
                masked_seq[mask] = ntokens - 1 # use a special token for masked positions
                targets = seq.clone()

                optimizer.zero_grad()
                output = model(masked_seq.unsqueeze(1)) # add batch dimension
                loss = criterion(output.view(-1, ntokens), targets.view(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            progress_bar.set_postfix({'Loss': total_loss / (i + len(batch))})

if __name__ == '__main__':
    train()
