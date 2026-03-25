import os
import random
import re
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class DataProvider(Dataset):
    def __init__(
            self,
            epi_path=None, epi_args=None,
            hla_path=None,
            hla_args=None,
            shuffle=False,
            specific_hla=None,
            num_folds=None,
            epi_dataframe=None,
            ):
        self.epi_path = epi_path
        self.epi_dataframe = epi_dataframe  # Support for in-memory DataFrame
        self.epi_args = epi_args
        self.hla_path = hla_path
        self.hla_args = hla_args
        self.shuffle = shuffle
        self.specific_hla = specific_hla
        self.num_folds = num_folds
        self.custom_hla = self.hla_args.get('custom_hla', False) if self.hla_args else False
        self.hla_seq_map = self.make_hla_seq_map() if not self.specific_hla else None
        self.samples = self.get_samples()

    def normalize_hla_name(self, hla_name):
        hla_name = re.sub(r'\*|:|-', '', hla_name)
        return hla_name

    def make_hla_seq_map(self):
        hla_header = self.hla_args['hla_header']
        seq_header = self.hla_args['seq_header']
        separator = self.hla_args['separator']

        df_hla = pd.read_csv(self.hla_path, sep=separator)
        df_hla = df_hla.dropna(subset=[hla_header, seq_header])
        hla_seq_map = dict(zip(df_hla[hla_header], df_hla[seq_header]))
        # print(f'Number of HLA alleles: {len(hla_seq_map)}')
        return hla_seq_map

    def get_samples(self):
        hla_header = self.epi_args['hla_header']
        epi_header = self.epi_args['epi_header']
        fld_header = self.epi_args.get('fld_header', 'Fold')
        separator = self.epi_args['separator']
    
        # Use DataFrame directly if provided (in-memory), otherwise read from file
        if self.epi_dataframe is not None:
            df_epi = self.epi_dataframe.copy()
        else:
            df_epi = pd.read_csv(self.epi_path, sep=separator)
        
        df_epi = df_epi.dropna(subset=[hla_header, epi_header])
        self.df_epi = df_epi.copy()
        if self.num_folds is not None:
            self.fold_indices = [df_epi[df_epi[fld_header] == fold].index.tolist() for fold in range(self.num_folds)]

        samples = list(zip(df_epi[hla_header], df_epi[epi_header]))
        # print(f'Number of samples: {len(samples)}')
        if self.shuffle:
            random.shuffle(samples)
        return samples
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        hla_name, epi_seq = self.samples[idx]
        if "_" in hla_name:
            hla_name_ = hla_name.split("_")
            hla_seq = (self.hla_seq_map[hla_name_[0]], self.hla_seq_map[hla_name_[1]]) if not self.custom_hla else hla_name_
        else:
            hla_seq = self.hla_seq_map[hla_name] if self.hla_seq_map else hla_name
        return hla_name, epi_seq, hla_seq
