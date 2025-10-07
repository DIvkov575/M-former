import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import math
import os

# --- msa_transformer.py content ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class MSATransformer(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(MSATransformer, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output

# --- data_loader.py content ---
def load_msa(npz_path):
    """
    Loads and reconstructs an MSA from a given .npz file.

    The .npz file is expected to have a specific structure:
    - 'sequences': A structured array with fields like 'res_start', 'res_end', 
                   'del_start', 'del_end'.
    - 'residues': A structured array with the field 'res_type'.
    - 'deletions': A structured array with fields 'res_idx' and 'deletion'.

    Returns a NumPy array of shape (num_sequences, seq_length).
    """
    print(f"[data_loader] Attempting to load: {npz_path}")
    if not os.path.exists(npz_path):
        print(f"[data_loader] File does not exist: {npz_path}")
    data = np.load(npz_path)
    
    sequences_data = data['sequences']
    residues_data = data['residues']['res_type']
    deletions_data = data['deletions']

    reconstructed_msa = []
    
    # Determine the length of the longest sequence after deletions are applied.
    # This is needed to pad all sequences to the same length.
    max_len = 0
    for seq_info in sequences_data:
        seq_len = (seq_info['res_end'] - seq_info['res_start'])
        
        dels_slice = deletions_data[seq_info['del_start']:seq_info['del_end']]
        num_deletions = np.sum(dels_slice['deletion'])
        
        total_len = seq_len + num_deletions
        if total_len > max_len:
            max_len = total_len

    # 21 is often used as a gap token in protein sequence analysis
    gap_token = 21 

    for seq_info in sequences_data:
        # Extract the base sequence of residues
        sequence = list(residues_data[seq_info['res_start']:seq_info['res_end']])
        
        # Get the deletions for this specific sequence
        dels_slice = deletions_data[seq_info['del_start']:seq_info['del_end']]
        
        # Apply deletions by inserting gap tokens
        # We iterate in reverse to avoid messing up indices as we insert.
        # We also need to sort deletions by res_idx in descending order.
        sorted_dels = np.sort(dels_slice, order='res_idx')[::-1]
        
        for del_info in sorted_dels:
            res_idx = del_info['res_idx']
            num_dels = del_info['deletion']
            for _ in range(num_dels):
                sequence.insert(res_idx, gap_token)

        # Pad the sequence to max_len
        padding_needed = max_len - len(sequence)
        sequence.extend([gap_token] * padding_needed)
        
        reconstructed_msa.append(sequence)
        
    return np.array(reconstructed_msa, dtype=np.int32)

from torch.utils.data import Dataset

class StreamMSADataset(Dataset):
    """
    A PyTorch Dataset to stream MSA data from a directory of .npz files.
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_list = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npz')]
        print(f"Found {len(self.file_list)} files in {data_dir}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        npz_path = self.file_list[idx]
        msa_data = load_msa(npz_path)
        
        # Convert to PyTorch tensor
        msa_tensor = torch.from_numpy(msa_data).long()
        
        if msa_tensor.max().item() >= VOCAB_SIZE:
            raise ValueError(f"Max value in MSA from {npz_path} ({msa_tensor.max().item()}) is >= VOCAB_SIZE ({VOCAB_SIZE}).")
            
        return msa_tensor

def collate_fn(batch):
    """
    Pads sequences in a batch to the same length.
    'batch' is a list of tensors, where each tensor is an MSA from a file.
    """
    # 21 is the gap token, used for padding
    gap_token = 21
    
    # Find the maximum sequence length in this batch
    max_len = 0
    for msa in batch:
        if msa.shape[1] > max_len:
            max_len = msa.shape[1]
            
    # Pad each MSA to the max_len and stack them
    padded_batch = []
    for msa in batch:
        padding_needed = max_len - msa.shape[1]
        if padding_needed > 0:
            # Pad on the right (at the end of the sequence)
            padding = torch.full((msa.shape[0], padding_needed), gap_token, dtype=msa.dtype)
            padded_msa = torch.cat([msa, padding], dim=1)
            padded_batch.append(padded_msa)
        else:
            padded_batch.append(msa)
            
    # Concatenate all MSAs in the batch along a new dimension (the batch dimension)
    # This assumes each file contains multiple sequences that are treated as a single unit.
    # If we want to mix sequences from different files, the logic would be different.
    # Here, we'll just concatenate them, creating a larger batch.
    return torch.cat(padded_batch, dim=0)


# --- train.py content ---
# --- Training Setup ---
# We use a vocab size of 22 to be safe (20 AA + gap + mask)
VOCAB_SIZE = 23 
BATCH_SIZE = 4 # Number of files to load per batch

# Model parameters
EMBEDDING_DIM = 128
NUM_HEADS = 8
HIDDEN_DIM = 512
NUM_LAYERS = 6
DROPOUT = 0.1

# Training parameters
NUM_EPOCHS = 5
LEARNING_RATE = 0.001

# --- Main Training Loop ---
if __name__ == "__main__":
    print("Starting MSA Transformer training...")

    # 1. Load Data
    data_dir = "/Users/dmitriyivkov/programming/bpipe/bpipe/data/"
    print(f"Loading data from directory: {data_dir}")

    # Create a streaming dataset and dataloader
    dataset = StreamMSADataset(data_dir=data_dir)
    # We use a batch_size of 1 at the DataLoader level because each "item" from the
    # dataset is a full MSA from a file, which can have many sequences.
    # The collate_fn will then combine these (potentially variable-length) MSAs.
    # Let's adjust BATCH_SIZE to mean files per batch.
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    print(f"Data loader created. Number of batches: {len(dataloader)}")

    # 2. Model Initialization
    model = MSATransformer(
        ntoken=VOCAB_SIZE,
        ninp=EMBEDDING_DIM,
        nhead=NUM_HEADS,
        nhid=HIDDEN_DIM,
        nlayers=NUM_LAYERS,
        dropout=DROPOUT
    )

    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Training on device: {device}")

    # 3. Loss and Optimizer
    # We use ignore_index so that padding tokens don't contribute to the loss
    criterion = nn.CrossEntropyLoss(ignore_index=21) 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Training Loop
    model.train() # Set the model to training mode
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for i, sequences in enumerate(dataloader):
            # `sequences` is now a batch of sequences from one or more files,
            # padded and collated into a single tensor.
            sequences = sequences.to(device)
            
            # Our model expects input of shape (seq_len, batch_size)
            input_seq = sequences.transpose(0, 1)
            
            # The target is the same as the input.
            targets = input_seq

            optimizer.zero_grad()

            # Forward pass
            output = model(input_seq)

            # Reshape output and targets for loss calculation
            # Output: (SEQ_LEN * BATCH_SIZE, VOCAB_SIZE)
            # Targets: (SEQ_LEN * BATCH_SIZE)
            loss = criterion(output.view(-1, VOCAB_SIZE), targets.reshape(-1))

            # Backward pass and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) # Gradient clipping
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Average Loss: {avg_loss:.4f}")

    print("Training finished.")

    # --- Example of how to use the trained model for inference --- 
    model.eval() # Set the model to evaluation mode
    with torch.no_grad():
        # Get a single batch for inference
        try:
            sample_batch = next(iter(dataloader)).to(device)
            
            # Prepare input
            sample_input = sample_batch.transpose(0, 1)
            
            # Get the model's prediction
            prediction = model(sample_input)
            
            # Get the predicted token indices
            predicted_indices = torch.argmax(prediction, dim=-1)
            
            print("\n--- Inference Example ---")
            print(f"Input shape: {sample_input.shape}")
            print(f"Prediction shape: {prediction.shape}")
            print(f"Predicted indices shape: {predicted_indices.shape}")
            print("Example predicted sequence (first sequence in batch):")
            print(predicted_indices[:, 0])
        except StopIteration:
            print("\nCould not get a batch for inference, the dataloader is empty.")

