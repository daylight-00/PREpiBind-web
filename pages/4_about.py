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
- **Last Updated:** June 2025
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
- PREpiBind methodology: ~[Publication](https://doi.org/10.1093/tbd/xxx)~
- PREpiBind training & evaluation code: ~[GitHub](https://github.com/daylight-00/PREpiBind)~
- PREpiBind web server code: ~[GitHub](https://github.com/daylight-00/PREpiBind-web)~
- MHC-II allele datasets: ~[Zenodo](https://zenodo.org/communities/prepibind-mhc-alleles)~
- PREpiBind checkpoints: [HuggingFace](https://huggingface.co/daylight00/prepibind-esmc-300m)
- ESMC 300M checkpoints (float16): [HuggingFace](https://huggingface.co/daylight00/esmc-300m-2024-12)
- Cached embeddings (ESMC 300M): [HuggingFace](https://huggingface.co/daylight00/esmc-300m-2024-12)
    """)


with col2:
    st.markdown("##### **Licensing Information**")
    st.markdown("""
- PREpiBind source code, web server, and datasets: **MIT License**
- PREpiBind checkpoints: **MIT License**
- ESMC 300M checkpoints & generated embeddings: 
  **[Cambrian Open License](https://www.evolutionaryscale.ai/policies/cambrian-open-license-agreement)** (EvolutionaryScale)
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
@article{Jang2025PREpiBind,
  author = {Jang, David Hyunyoo and Kim, Dongwoo and Park, Byungho and Hwang, Untaek and Choi, Yoonjoo and Lee, Juyong},
  title = {PREpiBind: Protein Representation-integrated Epitope-MHC Class II Binding Prediction},
  year = {2025},
  note = {Unpublished manuscript},
}
```
**PREpiBind: Protein Representation-integrated Epitope-MHC Class II Binding Prediction**  
*TBD* (2025) [doi:10.1093/tbd/xxx]
                        
""")
with st.expander("Show full abstract"):
    st.markdown("""
Accurate prediction of peptide-major histocompatibility complex class II (pMHC-II) binding remains challenging due to polymorphism, variable peptide length, and complex structural interactions. Here, we present PREpiBind, a modular framework evaluating diverse protein representations, including classical alignment-based (BLOSUM62), structure-informed (AlphaFold 3, Chai-1, Boltz-1), and transformer-based protein language models (PLMs; ESMC, ESM3).

We systematically benchmarked PREpiBind against established methods (NetMHCIIpan, MixMHC2pred, DeepNeo) using comprehensive qualitative, quantitative (IC50), and mass spectrometry-derived ligandomics datasets. Transformer-based PLMs significantly outperformed classical and structure-informed embeddings across metrics (ROC AUC, PR AUC, F1, MCC). 

PREpiBind showed robust generalization to unseen and rare alleles, demonstrated by stringent leave-one-molecule-out and cross-species validations. Our results highlight the critical importance of representation choice, emphasizing the advantages of context-aware, sequence-based embeddings over traditional methods.

PREpiBind provides a reproducible, scalable computational tool, enhancing predictive performance and generalizability for immunotherapy design, vaccine development, and fundamental antigen presentation studies.
    """)

write_st_end()