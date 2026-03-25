import streamlit as st
import pandas as pd
import os
import sys
import plotly.figure_factory as ff
import plotly.graph_objects as go
from app import get_models
from app import write_st_end

models = get_models()

utils_path = os.path.abspath('code')
sys.path.insert(0, utils_path)
from inference import main, load_config


def run_prepibind(
    df,
    num_workers=4,
    batch_size=128,
    use_compile=False,
    plot_kde=True,
    show_top_binders=5,
    model=None
):
    mhc_map = 'data/mhc_mapping.csv'
    hla_emb_dir = 'data/emb_hla_esmc_small_0601_fp16'

    config_path = 'config_demo.py'
    config = load_config(
        config_path,
        num_workers=num_workers,
        batch_size=batch_size,
        use_compile=use_compile,
        test_dataframe=df,
        hla_path=mhc_map,
        plot=plot_kde,
        hla_emb_dir=hla_emb_dir,
    )

    df_org = main(config, model)[['MHC_alpha', 'MHC_beta', 'Epitope']+['Score', 'Logits']]
    cols = ['Score', 'Logits']
    df_org[cols] = df_org[cols].apply(pd.to_numeric, errors='coerce')
    df_out = df_org.copy()
    if show_top_binders:
        df_out = df_out.nlargest(int(show_top_binders), 'Score')
    for col in cols:
        df_out[col] = df_out[col].map(lambda x: f"{x:.5f}" if pd.notnull(x) else "")
    return df_org, df_out

def plot_plot_kde(df_epi):
    scores = df_epi['Score'].astype(float).values
    hist_data = [scores]
    fig = ff.create_distplot(
        hist_data,
        group_labels=['Prediction'],
        show_hist=False,
        colors=['#29BDFD'],
        curve_type='kde',
    )
    fig['data'][0]['fill'] = 'tozeroy'
    fig.update_layout(
        xaxis_title="Predictions",
        yaxis_title="Density",
        margin=dict(l=40, r=30, t=80, b=40),
        showlegend=False,
    )
    fig.update_yaxes(showgrid=True, gridwidth=1)
    fig.add_vline(
        x=0.5,
        line_width=2,
        line_dash="dash",
        line_color="#F53255",
        annotation_text="Threshold (0.5)",
        annotation_position="top left",
    )
    st.plotly_chart(fig, width="content")

#%%
# Streamlit App

st.set_page_config(
    page_title="Prediction | PREpiBind",
    page_icon=":dna:",
    layout="centered",
    # initial_sidebar_state="collapsed"
)
st.title("Prediction")

st.subheader("Input Data")
st.markdown(
    f"""
    - **You can enter up to 20 samples below.**
    - For larger datasets, please upload a CSV file.
    - If you upload a CSV, only the first 20 rows are shown as a preview. The total sample count is also displayed.
    """
)

MHC_alpha_list = st.session_state['alpha_list']
MHC_beta_list = st.session_state['beta_list']

def find_error_df(df, hard_check=True):
    invalid_chars = df[~df['Epitope'].apply(lambda x: all(aa in "ACDEFGHIKLMNPQRSTVWYX" for aa in x.upper()))]
    invalid_length = df[~df['Epitope'].apply(lambda x: len(x) == 15)]
    invalid_alpha = df[~df['MHC_alpha'].isin(MHC_alpha_list)]
    invalid_beta = df[~df['MHC_beta'].isin(MHC_beta_list)]
    has_error = False
    if not invalid_chars.empty:
        st.error("Invalid characters found in Epitope column.")
        st.dataframe(invalid_chars[['MHC_alpha', 'MHC_beta', 'Epitope']])
        has_error = True
    if not invalid_length.empty and hard_check:
        st.warning("Peptides of 15 residues are preferred, although other lengths are accepted.")
    if not invalid_alpha.empty or not invalid_beta.empty:
        st.error("Invalid MHC alleles found.")
        invalid_allele = pd.concat([invalid_alpha, invalid_beta], ignore_index=True)
        st.dataframe(invalid_allele[['MHC_alpha', 'MHC_beta']])
        has_error = True
    return has_error

if 'input_df' not in st.session_state:
    st.session_state['input_df'] = pd.DataFrame(columns=["MHC_alpha", "MHC_beta", "Epitope"])

csv_file = st.file_uploader("Upload CSV", type="csv")

uploaded = False
if csv_file:
    df_csv = pd.read_csv(csv_file)
    if all(col in df_csv.columns for col in ["MHC_alpha", "MHC_beta", "Epitope"]):
        if not find_error_df(df_csv):
            st.session_state['input_df'] = df_csv
            uploaded = True
            st.success(f"Loaded CSV with {len(df_csv)} samples.")
    else:
        st.error("CSV must contain 'MHC_alpha', 'MHC_beta', and 'Epitope' columns.")

import re
def get_mhc_prefix(mhc):
    for prefix in ['HLA-DP', 'HLA-DQ', 'HLA-DR', 'H2']:
        if mhc.startswith(prefix):
            return prefix
def get_mhc_prefixes(mhc_list):
    return sorted(set(get_mhc_prefix(mhc)for mhc in mhc_list if get_mhc_prefix(mhc) is not None))
all_prefixes = sorted(set(get_mhc_prefixes(MHC_alpha_list) + get_mhc_prefixes(MHC_beta_list)))

def filter_by_prefix(mhc_list, prefixes):
    return [mhc for mhc in mhc_list if any(mhc.startswith(prefix) for prefix in prefixes)]

if not uploaded:
    default_selected = all_prefixes
    selected_prefixes = st.multiselect(
        "Select MHC Prefixes",
        options=all_prefixes, 
        default=default_selected, 
        help="Select prefixes to filter MHC alleles. If no prefix is selected, all MHC alleles will be shown."
    )
    filtered_alpha = filter_by_prefix(MHC_alpha_list, selected_prefixes)
    filtered_beta = filter_by_prefix(MHC_beta_list, selected_prefixes)
    with st.form("manual_input"):

        cols = st.columns(3)
        mhc_alpha = cols[0].selectbox("MHC alpha", filtered_alpha, index=0 if filtered_alpha else None)
        mhc_beta = cols[1].selectbox("MHC beta", filtered_beta, index=0 if filtered_beta else None)
        epitope = cols[2].text_input("Epitope", max_chars=25)
        submitted = st.form_submit_button("Add Entry")
        if epitope and not all(aa in "ACDEFGHIKLMNPQRSTVWY" for aa in epitope.upper()):
            st.error("Epitope must contain only valid amino acids.")
        elif submitted and mhc_alpha and mhc_beta and epitope:
            if len(epitope) != 15:
                st.warning("Peptides of 15 residues are preferred, although other lengths are accepted.")

            new_entry = pd.DataFrame({"MHC_alpha": [mhc_alpha], "MHC_beta": [mhc_beta], "Epitope": [epitope.upper()]})
            st.session_state['input_df'] = pd.concat([st.session_state['input_df'], new_entry], ignore_index=True)
            # st.success("Entry added successfully. ")

if not st.session_state['input_df'].empty:
    if not uploaded:
        st.session_state['input_df']['Delete'] = False
        edited_df = st.data_editor(
            st.session_state['input_df'],
            column_config={"Delete": st.column_config.CheckboxColumn("Select")},
            width="stretch"
        )
        button_cols = st.columns(2)
        delete_selected = button_cols[0].button("Delete Selected", key="delete_selected_btn")
        delete_all = button_cols[1].button("Delete All", key="delete_all_btn")
        # 선택된 행 삭제
        if delete_selected:
            st.session_state['input_df'] = edited_df[~edited_df['Delete']].drop(columns=['Delete']).reset_index(drop=True)
            st.rerun()
        if delete_all:
            st.session_state['input_df'] = pd.DataFrame(columns=["MHC_alpha", "MHC_beta", "Epitope"])
            st.rerun()
    else:
        st.dataframe(st.session_state['input_df'], width="stretch")
else:
    st.info("No data available.")

with st.expander("Options", expanded=False):
    batch_size = 128
    selected_model = st.selectbox("Select measurement type", ['Qualitative', 'Mass Spectrometry', 'IC50 (<500nM)', 'IC50 (<1000nM)'], index=0)
    show_top_binders = st.selectbox("Show top binders", ['All', 5, 10, 20, 50], index=2)
    model = models.get(selected_model, None)
    if show_top_binders == 'All': show_top_binders = None
    plot_kde = st.checkbox("Plot KDE", value=True)
    use_compile = False

if st.button("Run Prediction"):
    if st.session_state['input_df'].empty:
        st.error("No input data provided!")
    elif not find_error_df(st.session_state['input_df'], hard_check=False):
        with st.spinner("Running prediction..."):
            df = st.session_state['input_df'].copy()
            df['MHC'] = df['MHC_alpha'] + '_' + df['MHC_beta']
            original_df, result_df = run_prepibind(
                df,
                batch_size=batch_size,
                show_top_binders=show_top_binders or 0,
                plot_kde=plot_kde,
                use_compile=use_compile,
                model=model
            )
            st.session_state['original_df'] = original_df
            st.session_state['result_df'] = result_df
            st.session_state['prediction_done'] = True
        st.success("Prediction complete!")

if st.session_state.get('prediction_done', False):
    original_df = st.session_state['original_df']
    result_df = st.session_state['result_df']
    st.markdown("# Results")
    if show_top_binders is None or len(original_df) <= show_top_binders:
        pass
    else:
        st.markdown(f"### Top {show_top_binders} Binders")
    st.dataframe(result_df)
    if plot_kde:
        score_count = original_df['Score'].unique().size
        if score_count < 2:
            st.warning("Not enough data to plot KDE. Passing plot.")
        else:
            st.markdown("### Prediction Plot")
            plot_plot_kde(original_df)
    st.markdown("### Download Results")
    csv = original_df.to_csv(index=False).encode()
    st.download_button("Download result", csv, file_name="prediction.csv", mime="text/csv")

write_st_end()