import streamlit as st
import pandas as pd
import re

import os, sys
from pathlib import Path
utils_path = os.path.abspath('code')
sys.path.insert(0, utils_path)
from engine import shared

@st.cache_resource(show_spinner="Loading models...")
def get_engine():
    engine = shared(Path(__file__).resolve().parent)
    engine.hla.preload()
    return engine

def predict(df, head, custom=False, batch_size=128, windows=None):
    """Run the engine on `df` with a progress bar; `custom` takes the HLA chains as sequences instead of allele names, `windows` ({chain: (start, end)}) the domain windows the user chose for them."""
    bar, text = st.progress(0), st.empty()
    def show(done, total):
        percent = int(done / total * 100)
        bar.progress(percent)
        text.text(f"{percent}%")
    engine = get_engine()
    if custom:
        out = engine.run_custom(df, head, batch_size=batch_size, progress=show, windows=windows)
    else:
        out = engine.run(df, head, batch_size=batch_size, progress=show)
    bar.empty()
    text.empty()
    return out

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

engine = get_engine()

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
    st.Page("pages/6_scan.py", title="Scan"),
    st.Page("pages/5_custom.py", title="Custom Mode"),
    st.Page("pages/2_evaluation.py", title="Evaluation"),
    st.Page("pages/3_instructions.py", title="Instructions"),
    st.Page("pages/4_about.py", title="About"),
], position="top",)
pg.run()
