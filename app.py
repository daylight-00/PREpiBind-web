"""PREpiBind img2 web server — entry point.

The img2 server is a reboot of the img1-era one, not an update of it
(`IMG/docs/decisions/img2-is-a-reboot-and-the-five-workstreams.md` §1). Three things differ from
the page a returning user remembers, and each is a decision rather than a preference:

  * it serves the img2 **final** artifacts, one fit on the whole development pool at seed 42, not a
    cross-validation fold picked per task;
  * it reports **%Rank**, not a sigmoid probability, and its binary call is a %Rank cut;
  * the **Qualitative** head is not on the main path. It remains reachable on the clearly labelled
    img1 legacy page.
"""
import os
import sys

import streamlit as st

from security_config import apply_security

apply_security()

sys.path.insert(0, os.path.abspath("code"))

st.set_page_config(page_title="PREpiBind", page_icon=":dna:", layout="centered")

# Which GPU to serve from. Hardcoding this cost the img1 server a silent dependency on one card
# being free; it is configuration, not code.
if os.environ.get("PREPIBIND_GPU"):
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["PREPIBIND_GPU"]

import artifacts  # noqa: E402  (after the path insert)
import chains  # noqa: E402


@st.cache_resource(show_spinner="Loading the language model and the three heads…")
def get_engine():
    import engine

    return engine.load()


@st.cache_resource(show_spinner=False)
def get_chain_lists():
    return chains.chain_lists()


@st.cache_resource(show_spinner=False)
def get_bands():
    return artifacts.bands()


@st.cache_resource(show_spinner=False)
def get_integrity():
    """Served weights and vendored code, re-hashed once per process."""
    return {"served": artifacts.verify(), "vendored": artifacts.check_vendored()}


def write_st_end():
    st.markdown("---")
    st.caption(
        "Developed and maintained by David Hyunyoo Jang, Dongwoo Kim and Juyong Lee  \n"
        "[Lab of Computational Drug Discovery](https://sites.google.com/view/lcbc) | "
        "[College of Pharmacy, Seoul National University](https://snupharm.snu.ac.kr/en/)  \n"
        "Research use only. Not for clinical or diagnostic use."
    )


st.markdown(
    """<style>[data-testid="stDecoration"] {display: none;}</style>""",
    unsafe_allow_html=True,
)

pg = st.navigation(
    [
        st.Page("pages/0_home.py", title="Home", default=True),
        st.Page("pages/1_prediction.py", title="Prediction"),
        st.Page("pages/4_scan.py", title="Antigen scan"),
        st.Page("pages/6_custom.py", title="Custom HLA"),
        st.Page("pages/2_evaluation.py", title="Evaluation"),
        st.Page("pages/5_legacy.py", title="img1 legacy"),
        st.Page("pages/3_instructions.py", title="Instructions"),
        st.Page("pages/4_about.py", title="About"),
    ],
    position="top",
)
pg.run()
