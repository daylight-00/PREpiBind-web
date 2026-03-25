import streamlit as st
from app import write_st_end

st.set_page_config(page_title="PREpiBind: MHC-II Epitope Prediction", page_icon=":dna:", layout="centered")

st.title("PREpiBind Web Server")
st.markdown("**P**rotein **R**epresentation-integrated **Epi**tope-MHC Class II **Bind**ing Prediction")
st.markdown("""
### Overview
**PREpiBind** is a web server for predicting peptide binding to human and mouse MHC class II molecules.
It implements a transformer-based model using ESMC 300M protein language model embeddings, trained on
qualitative, IC50, and mass spectrometry ligandomics data from IEDB (15-mer peptides; 151 unique HLA dimers).
Pre-computed embeddings covering 7,282 alleles (838 alpha, 6,444 beta chains) from IPD-IMGT/HLA enable
prediction across alleles beyond those represented in the training data via PLM-based zero-shot generalization.

""")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
### Quick Start

1. Go to the **[Prediction](/prepibind/prediction)** page
2. Enter peptide sequences (15-mers recommended) and select MHC alleles
3. Choose a prediction model (Qualitative / IC50 / MS)
4. Click **Run Prediction** and download results as CSV

For novel or non-catalogued alleles, use **[Custom Mode](/prepibind/custom)**.

""")
with col2:
    st.markdown("""
### Capabilities

- 7,282 MHC-II alleles (human HLA-DP/DQ/DR and murine H2)
- Four prediction models: Qualitative, IC50 <500 nM, IC50 <1000 nM, MS elution
- Standard Mode (pre-computed embeddings) and Custom HLA Mode (novel allele sequences)
- Model benchmarking against user-supplied labeled datasets
""")

st.markdown("""
See [Instructions](/prepibind/instructions) for input format and usage details.
See [About](/prepibind/about) for citation, licensing, and contact information.

""")

write_st_end()
