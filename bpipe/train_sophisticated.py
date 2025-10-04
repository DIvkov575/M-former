import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sophisticated_transformer import SophisticatedTransformer
from data_pipeline import (
    BoltzTokenizer, BoltzCropper, BoltzFeaturizer, 
    DatasetConfig, DataConfig, Dataset, TrainingDataset, 
    Manifest, RandomSampler, collate, token_ids, num_tokens, get_symmetries
)
import glob
from tqdm import tqdm
import os
from pathlib import Path # Added this import

# This load_data is simplified as the data_pipeline handles the complex loading
def load_data_manifest(data_dir, manifest_filename="manifest.json"):
    manifest_path = Path(data_dir) / manifest_filename
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found at {manifest_path}")
    return Manifest.load(manifest_path)


def train():
    # Model parameters
    d_model = 128 # embedding dimension
    nhead = 4 # number of heads
    d_ff = 256 # feedforward dimension
    nlayers = 4 # number of layers
    dropout = 0.1

    # Training parameters
    batch_size = 1 # smaller batch size due to larger model and data
    epochs = 1
    lr = 0.001
    mask_prob = 0.15
    max_tokens = 256 # Example max tokens
    max_atoms = 2000 # Example max atoms
    max_seqs = 512 # Example max sequences
    samples_per_epoch = 10 # Small number for testing

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize data pipeline components
    tokenizer = BoltzTokenizer()
    cropper = BoltzCropper()
    featurizer = BoltzFeaturizer()
    sampler = RandomSampler()

    # Load manifest
    data_dir = os.path.join(os.getcwd(), "bpipe", "data")
    manifest = load_data_manifest(data_dir)

    # Load manifest
    data_dir_path = Path(data_dir)
    manifest = load_data_manifest(data_dir)

    # Create a single Dataset object
    single_dataset = Dataset(
        target_dir=data_dir_path, # Assuming structure npz are here if needed
        msa_dir=data_dir_path,
        manifest=manifest,
        prob=1.0,
        sampler=sampler,
        cropper=cropper,
        tokenizer=tokenizer,
        featurizer=featurizer,
    )

    # Create DataConfig with the single Dataset
    data_cfg = DataConfig(
        datasets=[DatasetConfig(
            target_dir=data_dir,
            msa_dir=data_dir,
            prob=1.0,
            sampler=sampler,
            cropper=cropper,
            filters=[],
            manifest_path=os.path.join(data_dir, "manifest.json")
        )],
        filters=[],
        featurizer=featurizer,
        tokenizer=tokenizer,
        max_atoms=max_atoms,
        max_tokens=max_tokens,
        max_seqs=max_seqs,
        samples_per_epoch=samples_per_epoch,
        batch_size=batch_size,
        num_workers=0, # Set to 0 for debugging, can increase later
        random_seed=42,
        pin_memory=False,
        symmetries="", # Dummy value
        atoms_per_window_queries=32,
        min_dist=2.0,
        max_dist=22.0,
        num_bins=64,
        overfit=None,
        pad_to_max_tokens=True,
        pad_to_max_atoms=False,
        pad_to_max_seqs=True,
        crop_validation=False,
        return_train_symmetries=False,
        return_val_symmetries=False,
        train_binder_pocket_conditioned_prop=0.0,
        val_binder_pocket_conditioned_prop=0.0,
        binder_pocket_cutoff=6.0,
        binder_pocket_sampling_geometric_p=0.0,
        val_batch_size=1,
        compute_constraint_features=False,
    )

    # Create training dataset
    train_dataset = TrainingDataset(
        datasets=[single_dataset],
        samples_per_epoch=data_cfg.samples_per_epoch,
        symmetries=get_symmetries(data_cfg.symmetries), # Using dummy get_symmetries
        max_atoms=data_cfg.max_atoms,
        max_tokens=data_cfg.max_tokens,
        max_seqs=data_cfg.max_seqs,
        pad_to_max_atoms=data_cfg.pad_to_max_atoms,
        pad_to_max_tokens=data_cfg.pad_to_max_tokens,
        pad_to_max_seqs=data_cfg.pad_to_max_seqs,
        # symmetries=symmetries, # This was causing an error, symmetries is already passed to get_symmetries
        atoms_per_window_queries=data_cfg.atoms_per_window_queries,
        min_dist=data_cfg.min_dist,
        max_dist=data_cfg.max_dist,
        num_bins=data_cfg.num_bins,
        overfit=data_cfg.overfit,
        binder_pocket_conditioned_prop=data_cfg.train_binder_pocket_conditioned_prop,
        binder_pocket_cutoff=data_cfg.binder_pocket_cutoff,
        binder_pocket_sampling_geometric_p=data_cfg.binder_pocket_sampling_geometric_p,
        return_symmetries=data_cfg.return_train_symmetries,
        compute_constraint_features=data_cfg.compute_constraint_features,
    )

    dataloader = DataLoader(
        train_dataset,
        batch_size=data_cfg.batch_size,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
        shuffle=False,
        collate_fn=collate,
    )

    model = SophisticatedTransformer(num_tokens, d_model, nhead, d_ff, nlayers, dropout).to(device)
    # model = torch.compile(model) # Commented out for broader compatibility
    criterion = nn.CrossEntropyLoss(ignore_index=token_ids["<pad>"]) # Ignore padding index
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch in progress_bar:
            if not batch: # Check if batch is empty
                print("Skipping empty batch.")
                continue
            # Ensure all tensors in batch are on the correct device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            # The model expects src (batch, num_seqs, seq_len)
            # We need to extract this from the featurized batch
            # For now, let's assume the target for prediction is also msa
            src = batch["msa"][:, :, :, 0] # Taking the first feature of one-hot msa as input
            targets = batch["msa"][:, :, :, 0] # Target is also the msa input for MLM

            # Create mask for MLM
            mask = torch.rand(src.shape, device=device) < mask_prob
            mask = mask & (src != token_ids["<pad>"]) # Do not mask padding tokens
            
            masked_msa = src.clone()
            masked_msa[mask] = token_ids["UNK"] # Use UNK token for masked positions

            # Create padding mask for the attention
            src_key_padding_mask = (src == token_ids["<pad>"])

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                output = model(masked_msa, src_key_padding_mask=src_key_padding_mask)
                
                # Reshape output and targets for CrossEntropyLoss
                # output shape: (batch, num_seqs, seq_len, ntokens)
                # targets shape: (batch, num_seqs, seq_len)
                loss = criterion(output.view(-1, num_tokens), targets.view(-1))
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'Loss': total_loss / (progress_bar.n + 1)})

if __name__ == '__main__':
    train()
