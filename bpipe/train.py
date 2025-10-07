

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from msa_transformer import MSATransformer
from data_loader import load_msa

# --- Training Setup ---
# We use a vocab size of 22 to be safe (20 AA + gap + mask)
VOCAB_SIZE = 23 
BATCH_SIZE = 64 # Number of sequences per batch

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
    print("Loading and processing MSA data from 1a0a_c.npz...")
    msa_data = load_msa('/Users/dmitriyivkov/programming/bpipe/bpipe/data/1a0a_c.npz')
    SEQ_LEN = msa_data.shape[1]
    
    # Convert to PyTorch tensor
    msa_tensor = torch.from_numpy(msa_data).long()
    print(f"[train.py] Max value in msa_tensor: {msa_tensor.max().item()}")
    if msa_tensor.max().item() >= VOCAB_SIZE:
        raise ValueError(f"Max value in MSA ({msa_tensor.max().item()}) is >= VOCAB_SIZE ({VOCAB_SIZE}). Adjust VOCAB_SIZE.")
    
    # Create a dataset and dataloader
    dataset = TensorDataset(msa_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"Data loaded. MSA shape: {msa_data.shape}")
    print(f"Sequence length: {SEQ_LEN}")
    print(f"Number of batches: {len(dataloader)}")


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
        for i, batch in enumerate(dataloader):
            # batch is a list containing one tensor of shape (BATCH_SIZE, SEQ_LEN)
            sequences = batch[0].to(device)
            
            # Our model expects input of shape (seq_len, batch_size)
            input_seq = sequences.transpose(0, 1)
            
            # The target is the same as the input.
            # We are doing masked language modeling implicitly.
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
        sample_batch = next(iter(dataloader))[0].to(device)
        
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
