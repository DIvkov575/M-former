import torch
import torch.nn as nn
import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import math
import os
from torch.cuda.amp import GradScaler, autocast

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
    gap_token = 21  # Standard gap token

    for seq_info in sequences_data:
        # Extract the base sequence of residues
        sequence = list(residues_data[seq_info['res_start']:seq_info['res_end']])
        
        # Get the deletions for this specific sequence
        dels_slice = deletions_data[seq_info['del_start']:seq_info['del_end']]
        
        # Apply deletions by inserting gap tokens
        if len(dels_slice) > 0:
            sorted_dels = np.sort(dels_slice, order='res_idx')[::-1]
            for del_info in sorted_dels:
                res_idx = del_info['res_idx']
                num_dels = del_info['deletion']
                # Insert gap tokens at the specified residue index
                for _ in range(num_dels):
                    # The res_idx is relative to the start of the *un-modified* sequence
                    sequence.insert(res_idx, gap_token)

        reconstructed_msa.append(np.array(sequence, dtype=np.int32))
        
    # Return a list of numpy arrays, not a single padded array
    return reconstructed_msa

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
        msa_data = load_msa(npz_path)  # This now returns a list of np.arrays
        
        # Convert each sequence to a PyTorch tensor
        msa_tensors = [torch.from_numpy(seq).long() for seq in msa_data]
        
        # It's inefficient to check vocab size here, let's move it to collate_fn
        # or assume the data is clean.
        # For now, we'll keep it to be safe, but this is a performance consideration.
        for tensor in msa_tensors:
            if tensor.max().item() >= VOCAB_SIZE:
                raise ValueError(f"Max value in MSA from {npz_path} ({tensor.max().item()}) is >= VOCAB_SIZE ({VOCAB_SIZE}).")
            
        return msa_tensors

def collate_fn(batch, max_tokens_per_batch=4096):
    """
    Collates a batch of MSAs, creating mini-batches that do not exceed
    a certain number of tokens (sequences * length).
    'batch' is a list of lists of tensors.
    """
    gap_token = 21
    all_sequences = [seq for msa_list in batch for seq in msa_list]
    all_sequences.sort(key=len, reverse=True)

    batches = []
    current_batch = []
    current_max_len = 0

    for seq in all_sequences:
        if not current_batch:
            current_batch.append(seq)
            current_max_len = len(seq)
            continue

        potential_tokens = max(current_max_len, len(seq)) * (len(current_batch) + 1)

        if potential_tokens > max_tokens_per_batch:
            # Finalize current batch
            padded_sequences = []
            for s in current_batch:
                padding_needed = current_max_len - len(s)
                padding = torch.full((padding_needed,), gap_token, dtype=s.dtype)
                padded_sequences.append(torch.cat([s, padding], dim=0))
            batches.append(torch.stack(padded_sequences, dim=0))

            # Start new batch
            current_batch = [seq]
            current_max_len = len(seq)
        else:
            current_batch.append(seq)
            current_max_len = max(current_max_len, len(seq))

    if current_batch:
        # Finalize the last batch
        padded_sequences = []
        for s in current_batch:
            padding_needed = current_max_len - len(s)
            padding = torch.full((padding_needed,), gap_token, dtype=s.dtype)
            padded_sequences.append(torch.cat([s, padding], dim=0))
        batches.append(torch.stack(padded_sequences, dim=0))

    return batches


# --- train.py content ---
# --- Training Setup ---
# We use a vocab size of 22 to be safe (20 AA + gap + mask)
VOCAB_SIZE = 23 
BATCH_SIZE = 1 # Number of files to load per batch

# Model parameters
EMBEDDING_DIM = 128
NUM_HEADS = 8
HIDDEN_DIM = 512
NUM_LAYERS = 6
DROPOUT = 0.1
MAX_LEN = 10000

# Training parameters
NUM_EPOCHS = 5
LEARNING_RATE = 0.001

# --- Main Training Loop ---
if __name__ == "__main__":
    print("Starting MSA Transformer training...")
    data_dir = "/home/dima/data/boltz/rcsb_processed_msa/"
    print(f"Loading data from directory: {data_dir}")

    dataset = StreamMSADataset(data_dir=data_dir)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    print(f"Data loader created. Number of batches: {len(dataloader)}")

    # 2. Model Initialization
    model = MSATransformer(
        ntoken=VOCAB_SIZE,
        ninp=EMBEDDING_DIM,
        nhead=NUM_HEADS,
        nhid=HIDDEN_DIM,
        nlayers=NUM_LAYERS,
        dropout=DROPOUT,
        max_len=MAX_LEN
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Training on device: {device}")

    # 3. Loss and Optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=21) 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Training Loop
    model.train() # Set the model to training mode
    scaler = GradScaler()

    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        num_batches = 0
        for i, sequence_chunks in tqdm.tqdm(enumerate(dataloader), total=dataloader.__len__()):
            for sequences in sequence_chunks:
                torch.cuda.empty_cache()
                sequences = sequences.to(device)
                # print(f"  - File group {i+1}/{len(dataloader)}, Sub-batch shape: {sequences.shape}")

                input_seq = sequences.transpose(0, 1)
                targets = input_seq

                optimizer.zero_grad()

                # Use autocast for mixed precision
                with autocast():
                    output = model(input_seq)
                    loss = criterion(output.view(-1, VOCAB_SIZE), targets.reshape(-1))

                # Scale loss and backpropagate
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
                num_batches += 1

        if num_batches > 0:
            avg_loss = total_loss / num_batches
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Average Loss: {avg_loss:.4f}")
        else:
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], No data processed.")

    print("Training finished.")

    torch.save(model.state_dict(), "model_weights.pth")

    # --- Example of how to use the trained model for inference --- 
    model.eval() # Set the model to evaluation mode
    with torch.no_grad():
        # Get a single batch for inference
        try:
            # The dataloader now returns a list of tensors (chunks)
            sample_chunks = next(iter(dataloader))
            if sample_chunks:
                sample_batch = sample_chunks[0].to(device) # Use the first chunk for inference
                
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
            else:
                print("\nCould not get a batch for inference, the dataloader returned an empty list.")
        except StopIteration:
            print("\nCould not get a batch for inference, the dataloader is empty.")

