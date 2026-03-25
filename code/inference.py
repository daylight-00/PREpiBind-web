import sys, os
sys.path.insert(0, os.getcwd())
import torch
import numpy as np
import importlib.util
import argparse
from dataprovider import DataProvider
from torch.utils.data import DataLoader
# from tqdm import tqdm
# import seaborn as sns
# import matplotlib.pyplot as plt
from model import UnifiedModel
from esm.tokenization import get_esmc_model_tokenizers
from esm.models.esmc import ESMC
import time
import streamlit as st

def load_config(config_path, batch_size=None, chkp_path=None, chkp_name=None, hla_path=None, test_path=None, num_workers=None, use_compile=None, plot=None, hla_emb_dir=None, esm_chkp_path=None, test_dataframe=None):
    """Dynamically import the config file."""
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    config = config_module.config
    if batch_size is not None:
        config["Test"]["batch_size"] = batch_size
    if num_workers is not None:
        config["Data"]["num_workers"] = num_workers
    if chkp_path is not None:
        config["chkp_path"] = chkp_path
    if chkp_name is not None:
        config["chkp_name"] = chkp_name
    if plot is not None:
        config["Test"]["plot"] = plot
    if use_compile is not None:
        config["Test"]["use_compile"] = use_compile
    if hla_path is not None:
        config["Data"]["hla_path"] = hla_path
    if test_path is not None:
        config["Data"]["test_path"] = test_path
    if test_dataframe is not None:
        config["Data"]["test_dataframe"] = test_dataframe
    if hla_emb_dir is not None:
        config["encoder_args"]["hla_emb_dir"] = hla_emb_dir
    if esm_chkp_path is not None:
        config['Test']['esm_chkp_path'] = esm_chkp_path
    return config

def load_unified_model(config, device, use_compile=False):
    model_esm = ESMC(
        d_model=960,
        n_heads=15,
        n_layers=30,
        tokenizer=get_esmc_model_tokenizers(),
        use_flash_attn=True,
    )
    # if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    #     dtype = torch.bfloat16
    dtype = torch.float16
    model_esm.load_state_dict(torch.load(config['Test']['esm_chkp_path'], map_location=device, weights_only=False))
    model_esm.to(device, dtype=dtype).eval()
    model = config["model"](**config["model_args"])
    model.load_state_dict(torch.load(config['Test']['chkp_path'], map_location=device, weights_only=False)['model_state_dict'])
    model.to(device, dtype=dtype).eval()
    unified_model = UnifiedModel(model_esm, model).to(device).eval()
    if use_compile:
        unified_model = torch.compile(unified_model)
    return unified_model

def test_model(model, dataloader, device):
    all_preds = []
    torch.backends.cudnn.benchmark = True

    # Streamlit 진행률 바와 텍스트 준비
    progress_bar = st.progress(0)
    percent_text = st.empty()
    total = len(dataloader)

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            batch = [item.to(device) for item in batch]
            y_pred = model(*batch)
            all_preds.append(y_pred.cpu())

            # 진행률 바 및 퍼센트 표시
            progress = int((idx + 1) / total * 100)
            progress_bar.progress(progress)
            percent_text.text(f"{progress}%")

    all_preds = torch.cat(all_preds, dim=0).numpy()
    # 작업이 끝나면 진행률 바와 텍스트 모두 비우기
    progress_bar.empty()
    percent_text.empty()
    return all_preds



def main(config, model):
    use_compile = config['Test'].get("use_compile", False)

    DATA_PROVIDER_ARGS = {
        "epi_path": config['Data'].get('test_path'),
        "epi_dataframe": config['Data'].get('test_dataframe'),  # Support in-memory DataFrame
        "epi_args": config['Data']['test_args'],
        "hla_path": config['Data']['hla_path'],
        "hla_args": config['Data']['hla_args'],
    }

    data_provider = DataProvider(**DATA_PROVIDER_ARGS)

    dataset = config["encoder"](data_provider, **config["encoder_args"])
    batch_size = config["Test"]["batch_size"] if "batch_size" in config["Test"] else len(dataset)
    num_workers = config["Data"]["num_workers"]
    collate_fn = config.get("collate_fn", None)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = load_unified_model(config, device, use_compile=use_compile)
    y_pred = test_model(model, dataloader, device)

    ############ Plotting ############
    df_epi = data_provider.df_epi.copy()
    df_epi['Logits'] = y_pred
    df_epi['Score'] = df_epi['Logits'].apply(lambda x: 1 / (1 + np.exp(-x)))  # Sigmoid function
    # df_epi.to_csv(os.path.join(out_path, f'prediction.csv'), index=False)
    return df_epi

def cli_main():
    parser = argparse.ArgumentParser(description="Train model with specified config.")
    parser.add_argument("config_path", type=str, help="Path to the config.py file.")
    parser.add_argument("--batch_size", type=int, help="Batch size.")
    parser.add_argument("--chkp_path", type=str, help="Checkpoint path.")
    parser.add_argument("--chkp_name", type=str, help="Checkpoint name.")
    parser.add_argument("--hla_path", type=str, help="Path to HLA data.")
    parser.add_argument("--test_path", type=str, help="Path to test data.")
    parser.add_argument("--num_workers", type=int, help="Number of workers for DataLoader.")
    parser.add_argument("--use_compile", action='store_true', help="Use torch.compile for the model.")
    parser.add_argument("--plot", action='store_true', help="Enable plotting of results.")
    parser.add_argument("--hla_emb_dir", type=str, help="Path to HLA embedding data.")
    parser.add_argument("--esm_chkp_path", type=str, help="Path to ESM checkpoint.")
    
    args = parser.parse_args()

    config = load_config(
        config_path=args.config_path,
        batch_size=args.batch_size,
        chkp_path=args.chkp_path,
        chkp_name=args.chkp_name,
        hla_path=args.hla_path,
        test_path=args.test_path,
        num_workers=args.num_workers,
        use_compile=args.use_compile,
        plot=args.plot,
        hla_emb_dir=args.hla_emb_dir,
        esm_chkp_path=args.esm_chkp_path
    )

    main(config)

if __name__ == "__main__":
    cli_main()
