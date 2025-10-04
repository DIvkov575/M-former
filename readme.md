
Data
---
To run training, you will need to download a few pre-processed datasets. Note that you will need ~250G of storage for all the data. If instead you want to re-run the preprocessing pipeline or processed your own raw data for training, please see the [instructions](#processing-raw-data) at the bottom of this page.

- The pre-processed RCSB (i.e PDB) structures:
```bash
wget https://boltz1.s3.us-east-2.amazonaws.com/rcsb_processed_targets.tar
tar -xf rcsb_processed_targets.tar
rm rcsb_processed_targets.tar
```

- The pre-processed RCSB (i.e PDB) MSA's:
```bash
wget https://boltz1.s3.us-east-2.amazonaws.com/rcsb_processed_msa.tar
tar -xf rcsb_processed_msa.tar
rm rcsb_processed_msa.tar
```

- The pre-processed OpenFold structures:
```bash
wget https://boltz1.s3.us-east-2.amazonaws.com/openfold_processed_targets.tar
tar -xf openfold_processed_targets.tar
rm openfold_processed_targets.tar
```

- The pre-processed OpenFold MSA's:
```bash
wget https://boltz1.s3.us-east-2.amazonaws.com/openfold_processed_msa.tar
tar -xf openfold_processed_msa.tar
rm openfold_processed_msa.tar
```

- The pre-computed symmetry files for ligands:
```bash
wget https://boltz1.s3.us-east-2.amazonaws.com/symmetry.pkl
```
