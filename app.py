import streamlit as st
from security_config import apply_security
apply_security()
import torch
from esm.tokenization import get_esmc_model_tokenizers
from esm.models.esmc import ESMC
import pandas as pd
import re

import os, sys
utils_path = os.path.abspath('code')
sys.path.insert(0, utils_path)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from model import UnifiedModel, plm_cat_mean_inf

model_args = {
    "hla_dim": 960,
    "epi_dim": 960,
    "head_div": 64,
}
@st.cache_resource(show_spinner="Loading models...")
def get_models():
    esm_chkp_path = "models/esmc_300m_2024_12_v0_fp16.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def load_unified_model(esm_chkp_path, chkp_path, device, use_compile=False):
        model_esm = ESMC(
            d_model=960,
            n_heads=15,
            n_layers=30,
            tokenizer=get_esmc_model_tokenizers(),
            use_flash_attn=True,
        )
        dtype = torch.float16
        model_esm.load_state_dict(torch.load(esm_chkp_path, map_location=device, weights_only=False))
        model_esm.to(device, dtype=dtype).eval()
        model = plm_cat_mean_inf(**model_args)
        model.load_state_dict(torch.load(chkp_path, map_location=device, weights_only=False)['model_state_dict'])
        model.to(device, dtype=dtype).eval()
        unified_model = UnifiedModel(model_esm, model).to(device).eval()
        if use_compile:
            unified_model = torch.compile(unified_model)
        return unified_model
    use_compile = True
    models = {
        "Qualitative": load_unified_model(esm_chkp_path, "models/prepi_esmc_small_e5_s128_f4_fp16.pth", device, use_compile=use_compile),
        "Mass Spectrometry": load_unified_model(esm_chkp_path, "models/prepi_esmc_small_ms_e5_s100_f0_fp16.pth", device, use_compile=use_compile),
        "IC50 (<500nM)": load_unified_model(esm_chkp_path, "models/prepi_esmc_small_ic50_500_e5_s128_f4_fp16.pth", device, use_compile=use_compile),
        "IC50 (<1000nM)": load_unified_model(esm_chkp_path, "models/prepi_esmc_small_ic50_1000_e5_s128_f1_fp16.pth", device, use_compile=use_compile),
    }
    return models

def natural_key(s):
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]
def get_mhc_choices():
    mhc_map = 'data/mhc_mapping.csv'
    df_map = pd.read_csv(mhc_map)
    df_map_a = df_map[~df_map['HLA_Name'].str.contains('B')]
    df_map_b = df_map[df_map['HLA_Name'].str.contains('B')]
    hla_list_a = sorted(df_map_a['HLA_Name'].unique(), key=natural_key)
    hla_list_b = sorted(df_map_b['HLA_Name'].unique(), key=natural_key)
    return {"alpha": hla_list_a, "beta": hla_list_b}
mhc_dict = get_mhc_choices()
if 'alpha_list' not in st.session_state:
    st.session_state['alpha_list'] = mhc_dict['alpha']
if 'beta_list' not in st.session_state:
    st.session_state['beta_list'] = mhc_dict['beta']

st.markdown("""
<style>
	[data-testid="stDecoration"] {
		display: none;
	}
</style>""", unsafe_allow_html=True)

models = get_models()

@st.cache_resource(show_spinner="Loading custom model...")
def get_single_custom_model(model_name):
    """Load only the selected model to conserve GPU memory."""
    esm_chkp_path = "models/esmc_300m_2024_12_v0_fp16.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_paths = {
        "Qualitative": "models/prepi_esmc_small_e5_s128_f4_fp16.pth",
        "Mass Spectrometry": "models/prepi_esmc_small_ms_e5_s100_f0_fp16.pth",
        "IC50 (<500nM)": "models/prepi_esmc_small_ic50_500_e5_s128_f4_fp16.pth",
        "IC50 (<1000nM)": "models/prepi_esmc_small_ic50_1000_e5_s128_f1_fp16.pth",
    }
    
    from model import UnifiedModel_custom_hla, plm_cat_mean_inf
    model_args = {
        "hla_dim": 960,
        "epi_dim": 960,
        "head_div": 64,
    }
    
    model_esm = ESMC(
        d_model=960,
        n_heads=15,
        n_layers=30,
        tokenizer=get_esmc_model_tokenizers(),
        use_flash_attn=True,
    )
    dtype = torch.float16
    model_esm.load_state_dict(torch.load(esm_chkp_path, map_location=device, weights_only=False))
    model_esm.to(device, dtype=dtype).eval()
    
    model = plm_cat_mean_inf(**model_args)
    model.load_state_dict(torch.load(model_paths[model_name], map_location=device)['model_state_dict'])
    model.to(device, dtype=dtype).eval()
    
    unified_model = UnifiedModel_custom_hla(model_esm, model).to(device).eval()
    # use_compile=False for custom models to avoid compatibility issues
    return unified_model

if "data_paths" not in st.session_state:
    st.session_state.data_paths = {
        "Qualitative": "data/test.csv",
        "Mass Spectrometry": "data/test_ms.csv",
        "IC50 (<500nM)": "data/test_ic50_500.csv",
        "IC50 (<1000nM)": "data/test_ic50_1000.csv",
    }

def write_st_end():
    st.markdown("---")
    st.markdown(
        "Developed and Maintained by David Hyunyoo Jang, Dongwoo Kim and Juyong Lee"
    )
    st.markdown(
        "[Lab of Computational Drug Discovery](https://sites.google.com/view/lcbc) | [College of Pharmacy, Seoul National University](https://snupharm.snu.ac.kr/en/)"
    )
    st.markdown("© 2025 David Hyunyoo Jang | Built with ESM")

pg = st.navigation([
    st.Page("pages/0_home.py", title="Home"),
    st.Page("pages/1_prediction.py", title="Prediction"),
    st.Page("pages/5_custom.py", title="Custom Mode"),
    st.Page("pages/2_evaluation.py", title="Evaluation"),
    st.Page("pages/3_instructions.py", title="Instructions"),
    st.Page("pages/4_about.py", title="About"),
], position="top",)
pg.run()
