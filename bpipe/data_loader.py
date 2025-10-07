import numpy as np
import os
import glob

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
        return None
    data = np.load(npz_path)
    
    sequences_data = data['sequences']
    residues_data = data['residues']['res_type']
    deletions_data = data['deletions']

    reconstructed_msa = []
    
    max_len = 0
    for seq_info in sequences_data:
        seq_len = (seq_info['res_end'] - seq_info['res_start'])
        
        dels_slice = deletions_data[seq_info['del_start']:seq_info['del_end']]
        num_deletions = np.sum(dels_slice['deletion'])
        
        total_len = seq_len + num_deletions
        if total_len > max_len:
            max_len = total_len

    gap_token = 21 

    for seq_info in sequences_data:
        sequence = list(residues_data[seq_info['res_start']:seq_info['res_end']])
        
        dels_slice = deletions_data[seq_info['del_start']:seq_info['del_end']]
        
        sorted_dels = np.sort(dels_slice, order='res_idx')[::-1]
        
        for del_info in sorted_dels:
            res_idx = del_info['res_idx']
            num_dels = del_info['deletion']
            for _ in range(num_dels):
                sequence.insert(res_idx, gap_token)

        padding_needed = max_len - len(sequence)
        sequence.extend([gap_token] * padding_needed)
        
        reconstructed_msa.append(sequence)
        
    return np.array(reconstructed_msa, dtype=np.int32)

class MSADirectoryLoader:
    """
    A data loader that iterates over .npz files in a directory.
    """
    def __init__(self, directory_path, file_extension='*.npz'):
        """
        Initializes the loader by finding all .npz files in the directory.
        """
        self.directory_path = directory_path
        self.file_paths = glob.glob(os.path.join(directory_path, file_extension))
        if not self.file_paths:
            print(f"[MSADirectoryLoader] No files found in: {self.directory_path}")
        self.current_index = 0

    def __len__(self):
        """
        Returns the total number of MSA files.
        """
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Loads and returns the MSA at the given index.
        """
        if idx >= len(self.file_paths):
            raise IndexError("Index out of range")
        npz_path = self.file_paths[idx]
        return load_msa(npz_path)

    def __iter__(self):
        """
        Returns an iterator for the MSA files.
        """
        self.current_index = 0
        return self

    def __next__(self):
        """
        Loads the next MSA file in the directory.
        """
        if self.current_index >= len(self.file_paths):
            raise StopIteration
        msa = self[self.current_index]
        self.current_index += 1
        return msa

if __name__ == '__main__':
    # Example usage of the MSADirectoryLoader
    data_dir = "/home/dima/data/boltz/rcsb_processed_data/"
    
    print(f"Initializing data loader for directory: {data_dir}")
    msa_loader = MSADirectoryLoader(data_dir)
    
    print(f"Found {len(msa_loader)} MSA files to load.")

    # Iterate through the data loader and load each MSA
    for i, msa in enumerate(msa_loader):
        if msa is not None:
            print(f"--- MSA {i+1}/{len(msa_loader)} ---")
            print(f"Successfully loaded MSA from: {msa_loader.file_paths[i]}")
            print(f"Shape of the reconstructed MSA: {msa.shape}")
            print(f"Data type: {msa.dtype}")
            print("\nFirst 5 rows and 30 columns of the MSA:")
            print(msa[:5, :30])
            print("-" * 20)
        else:
            print(f"Failed to load MSA from: {msa_loader.file_paths[i]}")

    print("Finished iterating through all MSAs.")