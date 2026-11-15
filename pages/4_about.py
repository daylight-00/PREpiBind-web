import streamlit as st
from app import write_st_end

st.set_page_config(
    page_title="About | PREpiBind",
    # page_icon="assets/favicon.png",
    layout="centered",
    # initial_sidebar_state="collapsed"
)
st.title("About PREpiBind")
st.markdown("**P**rotein **R**epresentation-integrated **Epi**tope-MHC Class II **Bind**ing Prediction")

# Authors section
st.markdown("""
### Development Team
For inquiries regarding the web server, please contact the developers.

- **David Hyunyoo Jang** *(Primary Developer)*
  [hwjang00@snu.ac.kr](mailto:hwjang00@snu.ac.kr) | [GitHub](https://github.com/daylight-00)

- **Dongwoo Kim** *(Co-developer)*
  [dingoh@snu.ac.kr](mailto:dingoh@snu.ac.kr)

- **Juyong Lee** *(Corresponding Author, PI)*
  [nicole23@snu.ac.kr](mailto:nicole23@snu.ac.kr)

[Lab of Computational Drug Discovery (LCBC)](https://sites.google.com/view/lcbc) | [College of Pharmacy, Seoul National University](https://snupharm.snu.ac.kr/en/)

""")

st.markdown("---")
st.subheader("System Information")

st.markdown("""
- **Version:** beta
- **Last Updated:** November 2026
- **Platform:** Web server
- **Access:** Freely available; no login required
- **Implementation:** Python/Streamlit
- **Compatibility:** Modern browsers (including mobile and dark mode support)
- **User Privacy:** All input data and prediction results are processed entirely in memory and are never written to disk. No user data is retained on the server.
""")
st.markdown("---")

# Resources & Support (2 columns)
st.subheader("Resources & Licensing")

col1, col2 = st.columns(2)

with col1:
    st.markdown("##### **Academic & Technical Resources**")
    st.markdown("""
- PREpiBind methodology: [bioRxiv preprint](https://doi.org/10.64898/2026.09.15.751749)
- PREpiBind training & evaluation code and datasets: [GitHub](https://github.com/daylight-00/PREpiBind), archived at [Zenodo](https://doi.org/10.5281/zenodo.22934281)
- Raw model predictions and the IEDB export behind the datasets: [Zenodo](https://doi.org/10.5281/zenodo.22857373)
- PREpiBind web server code: ~[GitHub](https://github.com/daylight-00/PREpiBind-web)~
- PREpiBind checkpoints released with the paper (the server runs the float16 copies): [HuggingFace (float32)](https://huggingface.co/daylight-00/prepibind), [HuggingFace (float16 demo)](https://huggingface.co/daylight-00/prepibind-demo)
- Pre-computed MHC-II embeddings: [HuggingFace](https://huggingface.co/datasets/daylight-00/prepibind-embeddings)
- ESMC 300M weights: [HuggingFace](https://huggingface.co/biohub/esmc-300m-2024-12)
    """)


with col2:
    st.markdown("##### **Licensing Information**")
    st.markdown("""
- PREpiBind source code, web server code, and checkpoints: **MIT License**
- ESMC 300M weights (Chan Zuckerberg Biohub): **MIT License**
- Training data derive from IEDB (CC BY 4.0); allele sequences from IPD-IMGT/HLA (CC BY-NoDerivs) and UniProt (CC BY 4.0). Terms and attribution: the [third-party notices](https://github.com/daylight-00/PREpiBind/blob/main/THIRD_PARTY_NOTICES.md) of the PREpiBind repository.
    """)

st.markdown("""
##### Research Use Only
This server is for research purposes only. Not for clinical use.
            """)

st.markdown("---")

# Citation & Licensing
st.subheader("Citation")
st.markdown("""
If you use PREpiBind, please cite:
```bibtex
@article{jang2026prepibind,
  title   = {PREpiBind: Protein Representation-integrated Epitope-MHC Class II Binding Prediction},
  author  = {Jang, David Hyunyoo and Kim, Dongwoo and Hwang, Untaek and Park, Byungho and Choi, Yoonjoo and Lee, Juyong},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.09.15.751749},
  url     = {https://www.biorxiv.org/content/10.64898/2026.09.15.751749v1}
}
```
**PREpiBind: Protein Representation-integrated Epitope-MHC Class II Binding Prediction**
*bioRxiv* (2026) [doi:10.64898/2026.09.15.751749](https://doi.org/10.64898/2026.09.15.751749)

""")
with st.expander("Show full abstract"):
    st.markdown("""
**Motivation:** Accurate peptide–MHC class II (pMHC-II) binding prediction is complicated by MHC polymorphism and context-dependent peptide recognition. Protein representations are usually evaluated in different pipelines, obscuring their contribution to predictive performance.

**Results:** We present PREpiBind, a joint-attention framework comparing ten protein representations under a fixed architecture and identical data splits across qualitative binding, mass spectrometry, and thresholded IC50 datasets. Protein language models (PLMs) generally achieved the strongest pooled performance; ESM3 Large reached a ROC-AUC of 0.927 ± 0.002 on the qualitative dataset. PREpiBind PLM variants exceeded evaluated reference methods on qualitative and mass-spectrometry benchmarks, whereas NetMHCIIpan-4.3 was higher on thresholded IC50 using its binding-affinity head. PLM advantages narrowed in allele-wise and held-out MHC-molecule evaluations, and H2-out rankings depended on aggregation, indicating scenario-dependent representation performance.
    """)

write_st_end()