import torch
import os
import argparse
from merged_msa_transformer import MSATransformer, load_msa, VOCAB_SIZE, EMBEDDING_DIM, NUM_HEADS, HIDDEN_DIM, NUM_LAYERS, DROPOUT, MAX_LEN

# --- Vocabulary Definition ---
# Based on the training script (VOCAB_SIZE=23), we define a consistent vocabulary.
# 0-20: 21 canonical tokens (20 standard amino acids + UNK for unknown)
# 21: Gap token ('-'), used for padding and indicating missing residues.
# 22: Mask token ('<MASK>'), used to prompt the model for a prediction.
CANONICAL_TOKENS = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
    "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
    "TYR", "VAL", "UNK",
]
VOCAB = CANONICAL_TOKENS + ["-", "<MASK>"]
TOKEN_TO_INDEX = {token: i for i, token in enumerate(VOCAB)}
INDEX_TO_TOKEN = {i: token for i, token in enumerate(VOCAB)}

GAP_TOKEN_ID = TOKEN_TO_INDEX["-"]
MASK_TOKEN_ID = TOKEN_TO_INDEX["<MASK>"]

def decode_sequence(indices):
    """Converts a sequence of token indices to a human-readable string."""
    return " ".join([INDEX_TO_TOKEN.get(i, "?") for i in indices])

def reconstruct_from_file(model_weights_path, npz_file_path, sequence_index_in_msa=0, extract_embeddings=False):
    """
    Loads a trained MSATransformer model and uses it to reconstruct gaps
    in a single sequence from a specified .npz MSA file.
    """
    # 1. Model Initialization
    model = MSATransformer(
        ntoken=VOCAB_SIZE,
        ninp=EMBEDDING_DIM,
        nhead=NUM_HEADS,
        nhid=HIDDEN_DIM,
        nlayers=NUM_LAYERS,
        dropout=DROPOUT,
        max_len=MAX_LEN
    )

    # 2. Load Model Weights
    if not os.path.exists(model_weights_path):
        print(f"Error: Model weights file not found at {model_weights_path}")
        print("Please ensure a trained model file (e.g., '1_model_weights.pt') exists.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded from {model_weights_path} and running on {device}")

    # 3. Load Data from .npz File
    if not os.path.exists(npz_file_path):
        print(f"Error: Data file not found at {npz_file_path}")
        return

    msa_data = load_msa(npz_file_path)
    if not msa_data or sequence_index_in_msa >= len(msa_data):
        print(f"Error: Not enough sequences in {npz_file_path}. Requested index {sequence_index_in_msa}, but only {len(msa_data)} found.")
        return

    original_sequence = torch.from_numpy(msa_data[sequence_index_in_msa]).long().to(device)
    
    # The model expects input of shape (sequence_length, batch_size)
    original_sequence_batch = original_sequence.unsqueeze(1)

    # 4. Prepare Input for Reconstruction
    gap_mask = (original_sequence_batch == GAP_TOKEN_ID)
    
    if not gap_mask.any():
        print("\nNo gaps ('-') found in the selected sequence. Nothing to reconstruct.")
        print(f"Original sequence: {decode_sequence(original_sequence.tolist())}")
        return

    # Replace gaps with the MASK token to ask the model for predictions
    masked_input = original_sequence_batch.clone()
    masked_input[gap_mask] = MASK_TOKEN_ID

    # 5. Run Model and Get Predictions
    with torch.no_grad():
        if extract_embeddings:
            prediction, embeddings = model(masked_input, return_embed=True)
            # Save embeddings
            output_embedding_path = f"embeddings_{os.path.basename(npz_file_path)}_{sequence_index_in_msa}.pt"
            torch.save(embeddings, output_embedding_path)
            print(f"\nEmbeddings saved to {output_embedding_path}")
        else:
            prediction = model(masked_input)

        _, predicted_indices = torch.max(prediction, dim=-1)

    # 6. Create the Reconstructed Sequence
    reconstructed_sequence_batch = original_sequence_batch.clone()
    # Fill in the gaps with the model's predictions
    reconstructed_sequence_batch[gap_mask] = predicted_indices[gap_mask]

    # Squeeze the batch dimension for decoding
    original_seq_list = original_sequence_batch.squeeze(1).tolist()
    reconstructed_seq_list = reconstructed_sequence_batch.squeeze(1).tolist()

    # 7. Print Results
    print(f"\n--- Reconstructing sequence {sequence_index_in_msa} from {os.path.basename(npz_file_path)} ---")
    print(f"Found {gap_mask.sum().item()} gaps to reconstruct.")
    print("\nOriginal sequence:")
    print(decode_sequence(original_seq_list))
    print("\nReconstructed sequence:")
    print(decode_sequence(reconstructed_seq_list))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstruct sequences and optionally extract embeddings from an MSA transformer model.")
    parser.add_argument("model_weights", help="Path to the trained model weights (.pth file).")
    parser.add_argument("npz_file", help="Path to the .npz file containing the MSA data.")
    parser.add_argument("--sequence_index", type=int, default=0, help="The index of the sequence to reconstruct within the MSA file.")
    parser.add_argument("--extract_embeddings", action="store_true", help="If set, extract and save the embeddings.")

    args = parser.parse_args()

    reconstruct_from_file(
        model_weights_path=args.model_weights,
        npz_file_path=args.npz_file,
        sequence_index_in_msa=args.sequence_index,
        extract_embeddings=args.extract_embeddings
    )