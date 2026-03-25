import torch
from torch.nn.utils.rnn import pad_sequence
from esm.utils import encoding
from esm.tokenization import EsmSequenceTokenizer
tokenizer = EsmSequenceTokenizer()

def pad_and_mask_collate_fn_inf(batch):
    """
    Collate function for standard mode (pre-computed HLA embeddings).
    batch: list of (hla_emb, epi_seq)
      hla_emb: float16 tensor (hla_len, D)
      epi_seq: list of token ids (with [CLS]/[SEP])
    Returns: (padded_hla, epi_tensor, mask_hla, mask_epi)
    """
    hla_list, epi_list = zip(*batch)
    batch_size = len(epi_list)

    hla_s_lens = [len(emb) for emb in hla_list]
    max_hla_len = max(hla_s_lens) if batch_size > 0 else 0
    padded_hla = pad_sequence(hla_list, batch_first=True, padding_value=0.0).to(torch.float16)
    mask_hla = torch.ones(batch_size, max_hla_len, dtype=torch.bool)
    for i, length in enumerate(hla_s_lens):
        mask_hla[i, :length] = False

    pad_token_id = tokenizer.pad_token_id
    max_epi_len = max(len(x) for x in epi_list)
    epi_tensor = torch.full((batch_size, max_epi_len), pad_token_id, dtype=torch.long)
    mask_epi = torch.ones(batch_size, max_epi_len, dtype=torch.bool)
    for i, tks in enumerate(epi_list):
        tks_tensor = torch.as_tensor(tks, dtype=torch.long)
        epi_tensor[i, :len(tks_tensor)] = tks_tensor
        mask_epi[i, :len(tks_tensor)] = False
    mask_epi = mask_epi[:, 1:-1]  # exclude [CLS] and [SEP]

    return (
        padded_hla,   # (B, max_hla_len, D_hla)
        epi_tensor,   # (B, max_epi_len) long
        mask_hla,     # (B, max_hla_len) bool
        mask_epi      # (B, max_epi_len - 2) bool
    )

def pad_and_mask_collate_fn_inf_custom_hla(batch):
    """
    Collate function for Custom HLA mode (on-the-fly embedding).
    batch: list of (hla_seq_a, hla_seq_b, epi_seq)
      hla_seq_a/b: list of token ids for alpha/beta chain (with [CLS]/[SEP])
      epi_seq: list of token ids (with [CLS]/[SEP])
    Returns: (hla_a_tensor, hla_b_tensor, epi_tensor, mask_hla_a, mask_hla_b, mask_epi)
    """
    hla_a_list, hla_b_list, epi_list = zip(*batch)
    batch_size = len(epi_list)

    pad_token_id = tokenizer.pad_token_id

    max_epi_len = max(len(x) for x in epi_list)
    epi_tensor = torch.full((batch_size, max_epi_len), pad_token_id, dtype=torch.long)
    mask_epi = torch.ones(batch_size, max_epi_len, dtype=torch.bool)
    for i, tks in enumerate(epi_list):
        tks_tensor = torch.as_tensor(tks, dtype=torch.long)
        epi_tensor[i, :len(tks_tensor)] = tks_tensor
        mask_epi[i, :len(tks_tensor)] = False
    mask_epi = mask_epi[:, 1:-1]  # exclude [CLS] and [SEP]

    max_hla_a_len = max(len(x) for x in hla_a_list)
    hla_a_tensor = torch.full((batch_size, max_hla_a_len), pad_token_id, dtype=torch.long)
    mask_hla_a = torch.ones(batch_size, max_hla_a_len, dtype=torch.bool)
    for i, tks in enumerate(hla_a_list):
        tks_tensor = torch.as_tensor(tks, dtype=torch.long)
        hla_a_tensor[i, :len(tks_tensor)] = tks_tensor
        mask_hla_a[i, :len(tks_tensor)] = False
    mask_hla_a = mask_hla_a[:, 1:-1]  # exclude [CLS] and [SEP]

    max_hla_b_len = max(len(x) for x in hla_b_list)
    hla_b_tensor = torch.full((batch_size, max_hla_b_len), pad_token_id, dtype=torch.long)
    mask_hla_b = torch.ones(batch_size, max_hla_b_len, dtype=torch.bool)
    for i, tks in enumerate(hla_b_list):
        tks_tensor = torch.as_tensor(tks, dtype=torch.long)
        hla_b_tensor[i, :len(tks_tensor)] = tks_tensor
        mask_hla_b[i, :len(tks_tensor)] = False
    mask_hla_b = mask_hla_b[:, 1:-1]  # exclude [CLS] and [SEP]

    return (
        hla_a_tensor,  # (B, max_hla_a_len) long
        hla_b_tensor,  # (B, max_hla_b_len) long
        epi_tensor,    # (B, max_epi_len) long
        mask_hla_a,    # (B, max_hla_a_len - 2) bool
        mask_hla_b,    # (B, max_hla_b_len - 2) bool
        mask_epi       # (B, max_epi_len - 2) bool
    )
