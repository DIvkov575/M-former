

import numpy as np

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
    import os
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

if __name__ == '__main__':
    # Example usage and verification
    msa = load_msa('1a0a_c.npz')
    print(f"Successfully loaded MSA.")
    print(f"Shape of the reconstructed MSA: {msa.shape}")
    print(f"Data type: {msa.dtype}")
    print("\nFirst 5 rows and 30 columns of the MSA:")
    print(msa[:5, :30])

