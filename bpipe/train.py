import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from msa_transformer import MSATransformer
from data_loader import MSADirectoryLoader

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

    # 1. Load Data using the new MSADirectoryLoader
    data_dir = "/home/dima/data/boltz/rcsb_processed_data/"
    print(f"Loading and processing MSA data from {data_dir}...")
    msa_loader = MSADirectoryLoader(data_dir)
    
    if len(msa_loader) == 0:
        print("No data found. Exiting.")
        exit()

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

    # 4. Training Loop - now iterates over each MSA from the loader
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch [{epoch+1}/{NUM_EPOCHS}] ---")
        total_epoch_loss = 0
        num_msas_processed = 0

        for i, msa_data in enumerate(msa_loader):
            if msa_data is None:
                print(f"Skipping MSA file {i+1}/{len(msa_loader)} due to loading error.")
                continue

            SEQ_LEN = msa_data.shape[1]
            
            # Convert to PyTorch tensor
            msa_tensor = torch.from_numpy(msa_data).long()
            if msa_tensor.max().item() >= VOCAB_SIZE:
                print(f"Skipping MSA file {i+1} due to vocab size issue.")
                continue
            
            # Create a dataset and dataloader for the current MSA
            dataset = TensorDataset(msa_tensor)
            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
            
            print(f"Processing MSA {i+1}/{len(msa_loader)} | Shape: {msa_data.shape} | Batches: {len(dataloader)}")

            model.train() # Set the model to training mode
            total_msa_loss = 0
            for batch in dataloader:
                sequences = batch[0].to(device)
                input_seq = sequences.transpose(0, 1)
                targets = input_seq

                optimizer.zero_grad()
                output = model(input_seq)
                loss = criterion(output.view(-1, VOCAB_SIZE), targets.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

                total_msa_loss += loss.item()
            
            avg_msa_loss = total_msa_loss / len(dataloader)
            print(f"MSA {i+1} Average Loss: {avg_msa_loss:.4f}")
            total_epoch_loss += avg_msa_loss
            num_msas_processed += 1

        avg_epoch_loss = total_epoch_loss / num_msas_processed if num_msas_processed > 0 else 0
        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}] Average Epoch Loss: {avg_epoch_loss:.4f}")

    print("\nTraining finished.")

    # --- Example of how to use the trained model for inference ---
    print("\n--- Running Inference Example ---")
    model.eval() # Set the model to evaluation mode
    with torch.no_grad():
        # Get a single MSA for inference
        try:
            first_msa = msa_loader[0]
            if first_msa is not None:
                sample_tensor = torch.from_numpy(first_msa).long()
                sample_dataset = TensorDataset(sample_tensor)
                sample_dataloader = DataLoader(sample_dataset, batch_size=BATCH_SIZE)
                
                sample_batch = next(iter(sample_dataloader))[0].to(device)
                sample_input = sample_batch.transpose(0, 1)
                
                prediction = model(sample_input)
                predicted_indices = torch.argmax(prediction, dim=-1)
                
                print(f"Input shape: {sample_input.shape}")
                print(f"Prediction shape: {prediction.shape}")
                print(f"Predicted indices shape: {predicted_indices.shape}")
                print("Example predicted sequence (first sequence in batch):")
                print(predicted_indices[:, 0])
            else:
                print("Could not load the first MSA for inference example.")
        except IndexError:
            print("No data available to run inference example.")