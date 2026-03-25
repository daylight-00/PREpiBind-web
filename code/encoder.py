import logging
import time
import torch
import numpy as np
from torch.utils.data import Dataset
from esm.tokenization import EsmSequenceTokenizer
from esm.utils import encoding
from pathlib import Path

logger = logging.getLogger(__name__)

#%% PLM
def get_plm_emb(hla_emb_dir, hla_name, start_idx_a=None, end_idx_a=None, max_retries=5, retry_delay=0.1):
    for attempt in range(max_retries):
        try:
            path = hla_emb_dir / f"{hla_name}.npy"
            embedding = np.load(path, mmap_mode='r')
            embedding = torch.tensor(embedding, dtype=torch.float16)
            if start_idx_a is not None and end_idx_a is not None:
                embedding = embedding[start_idx_a:end_idx_a]
            return embedding
        except OSError as e:
            logger.warning("[get_plm_emb] OSError (attempt %d/%d) hla_name=%s: %s", attempt + 1, max_retries, hla_name, e)
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise

class plm_plm_mask_msa_pair_inf(Dataset):
    def __init__(self, data_provider, hla_emb_dir):
        self.data_provider = data_provider
        self.hla_emb_dir = Path(hla_emb_dir)
        self.tokenizer = EsmSequenceTokenizer()

    def __len__(self):
        return len(self.data_provider)

    def __getitem__(self, idx):
        hla_name, epi_seq, _ = self.data_provider[idx]
        hla_name_a, hla_name_b = hla_name.split("_")
        hla_emb_a = get_plm_emb(self.hla_emb_dir, hla_name_a)
        hla_emb_b = get_plm_emb(self.hla_emb_dir, hla_name_b)
        hla_emb = torch.cat([hla_emb_a, hla_emb_b], dim=0)
        epi_seq = encoding.tokenize_sequence(epi_seq, self.tokenizer, add_special_tokens=True)
        return hla_emb, epi_seq

class plm_plm_mask_msa_pair_inf_custom_hla(Dataset):
    def __init__(self, data_provider, hla_emb_dir=None):
        self.data_provider = data_provider
        # self.hla_emb_dir = Path(hla_emb_dir)
        self.tokenizer = EsmSequenceTokenizer()

    def __len__(self):
        return len(self.data_provider)

    def __getitem__(self, idx):
        hla_name, epi_seq, hla_seq = self.data_provider[idx]
        hla_seq_a, hla_seq_b = hla_seq
        hla_seq_a = encoding.tokenize_sequence(hla_seq_a, self.tokenizer, add_special_tokens=True)
        hla_seq_b = encoding.tokenize_sequence(hla_seq_b, self.tokenizer, add_special_tokens=True)
        epi_seq = encoding.tokenize_sequence(epi_seq, self.tokenizer, add_special_tokens=True)
        return hla_seq_a, hla_seq_b, epi_seq
