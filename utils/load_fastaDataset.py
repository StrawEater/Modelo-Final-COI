from torch.utils.data import Dataset
from typing import List
from sklearn.model_selection import train_test_split
import os
import zipfile
from tqdm import tqdm
from pathlib import Path
import torch

class FastaDataset(Dataset):
    """Dataset for loading sequences from FASTA files organized by phylum"""

    def __init__(self, sequences: List[str], labels: List[int], max_length: int = 750):
        """
        Args:
            sequences: List of DNA sequences
            labels: List of integer labels (phylum indices)
            max_length: Maximum sequence length to use
        """
        self.sequences = sequences
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]

        # Truncate if too long
        if len(sequence) > self.max_length:
            sequence = sequence[:self.max_length]

        return sequence, label

def separate_train_val_test(sequences, labels, test_size, val_size, random_state = 42):

  relative_val_size = val_size/(1-test_size)

  X_temp, X_test, y_temp, y_test = train_test_split(
    sequences, labels, test_size=test_size, random_state=random_state, stratify=labels
  )

  X_train, X_val, y_train, y_val = train_test_split(
      X_temp, y_temp, test_size=relative_val_size, random_state=random_state, stratify=y_temp
  )

  return X_train, X_val, X_test, y_train, y_val, y_test



def parse_fasta_from_string(fasta_string) -> List[str]:
    """
    Objetive:
        Parse FASTA format string and return list of sequences

    Args:
        fasta_string: String containing FASTA formatted data

    Returns:
        List of DNA sequences (e.g: ['ATCG', 'GCTA'])
    """

    sequences = []
    current_sequence_parts = []

    for line in fasta_string.splitlines():

        line = line.strip()

        if not line: #Skip
            continue

        if line.startswith('>'): #Header line
            if current_sequence_parts:
                sequences.append(''.join(current_sequence_parts)) #concat and save
                current_sequence_parts = []

        else:
            current_sequence_parts.append(line.upper())

    # Add final sequence
    if current_sequence_parts:
        sequences.append(''.join(current_sequence_parts))

    return sequences


def add_fasta_sequences_to_dataset(zip_file, fasta_file, sequences, labels, taxonomic_unit_to_label, label_to_taxonomic_unit):
    # Extract taxonomic unit name from filename (without extension)
    taxonomic_unit = Path(fasta_file).stem

    # Assign label to taxon if not already assigned
    if taxonomic_unit not in taxonomic_unit_to_label:
      label = len(taxonomic_unit_to_label)
      taxonomic_unit_to_label[taxonomic_unit] = label
      label_to_taxonomic_unit[label] = taxonomic_unit

    label = taxonomic_unit_to_label[taxonomic_unit]

    # Read file content directly from zip
    with zip_file.open(fasta_file) as fasta_file_obj:

      content = fasta_file_obj.read().decode('utf-8')
      unit_sequences = parse_fasta_from_string(content)

      sequences.extend(unit_sequences)
      labels.extend([label] * len(unit_sequences))

def load_data_from_zip(zip_path: str):

  sequences = []
  labels = []
  phylum_to_label = {}
  label_to_phylum = {}

  # Leemos el zip file con los .fasta
  with zipfile.ZipFile(zip_path, 'r') as zip_ref:

    # Get list of all files in zip
    fasta_files = zip_ref.namelist()

    print(f"Found {len(fasta_files)} FASTA files")

    # Process each FASTA file directly from zip
    for fasta_filename in tqdm(fasta_files, desc="Loading FASTA files"):
        add_fasta_sequences_to_dataset(zip_ref,
                                       fasta_filename,
                                       sequences,
                                       labels,
                                       phylum_to_label,
                                       label_to_phylum
                                      )

  return sequences, labels, phylum_to_label, label_to_phylum


import torch
from torch.utils.data import Dataset

nt2idx = {'A':0, 'C':1, 'G':2, 'T':3}

class MultiTaxaFastaDataset(Dataset):
    def __init__(self, df, max_length=900, taxon_cols=['phylum','order','family','genus','species']):
        self.df = df
        self.max_length = max_length
        self.taxon_cols = taxon_cols

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = row['sequence'][:self.max_length]
    
        # Labels por taxon - ⚠️ CONVERSIÓN EXPLÍCITA A INT
        labels = {
            taxon: torch.tensor(int(row[taxon]), dtype=torch.long)  # ⭐ int()
            for taxon in self.taxon_cols
        }
    
        # True tokens para reconstrucción global
        true_tokens = torch.tensor(
            [nt2idx.get(nt, 0) for nt in seq] + [0]*(self.max_length - len(seq)), 
            dtype=torch.long
        )
    
        # Recon targets por taxon
        recon_targets = {taxon: true_tokens.clone() for taxon in self.taxon_cols}
    
        return seq, labels, recon_targets, true_tokens