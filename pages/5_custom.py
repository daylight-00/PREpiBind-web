import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from app import get_engine
from app import predict
from app import write_st_end
from engine import MAX_CHAIN_LENGTH, MAX_EPITOPE_LENGTH
from plots import kde_figure


def run_prepibind(
    df,
    batch_size=128,
    show_top_binders=5,
    head=None,
    windows=None
):
    df_org = predict(df, head, custom=True, batch_size=batch_size, windows=windows)[['MHC_alpha', 'MHC_beta', 'Epitope', 'HLA chains']+['Score', 'Logits']]
    cols = ['Score', 'Logits']
    df_org[cols] = df_org[cols].apply(pd.to_numeric, errors='coerce')
    df_out = df_org.copy()
    if show_top_binders:
        df_out = df_out.nlargest(int(show_top_binders), 'Score')
    for col in cols:
        df_out[col] = df_out[col].map(lambda x: f"{x:.5f}" if pd.notnull(x) else "")
    return df_org, df_out

def plot_plot_kde(df_epi):
    st.plotly_chart(kde_figure(df_epi['Score'].astype(float).values), width="content")

#%%
# Streamlit App

st.set_page_config(
    page_title="Custom Mode | PREpiBind",
    page_icon=":dna:",
    layout="centered",
    # initial_sidebar_state="collapsed"
)
st.title("Custom HLA Mode")

st.subheader("Input Data")
st.markdown(
    f"""
    This mode allows you to **directly input custom HLA α and β chain sequences** instead of selecting from predefined alleles.
    
    - Enter MHC alpha and beta sequences (full amino acid sequences, up to 300 residues each) along with epitope sequences (up to 20 residues).
    - You can manually add entries or upload a CSV file with `MHC_alpha`, `MHC_beta`, and `Epitope` columns.
    - All sequences must contain only valid amino acids (A-Z standard amino acid codes).
    """
)

def find_error_df(df, hard_check=True):
    invalid_chars = df[~df['Epitope'].apply(lambda x: all(aa in "ACDEFGHIKLMNPQRSTVWYX" for aa in x.upper()))]
    invalid_length = df[~df['Epitope'].apply(lambda x: len(x) == 15)]
    too_long = df[df['Epitope'].apply(lambda x: len(x) > MAX_EPITOPE_LENGTH)]
    long_chain = df[(df['MHC_alpha'].apply(len) > MAX_CHAIN_LENGTH) | (df['MHC_beta'].apply(len) > MAX_CHAIN_LENGTH)]
    invalid_alpha = df[~df['MHC_alpha'].apply(lambda x: all(aa in "ACDEFGHIKLMNPQRSTVWYX" for aa in x.upper()))]
    invalid_beta = df[~df['MHC_beta'].apply(lambda x: all(aa in "ACDEFGHIKLMNPQRSTVWYX" for aa in x.upper()))]
    has_error = False
    if not invalid_chars.empty:
        st.error("Invalid characters found in Epitope column.")
        st.dataframe(invalid_chars[['MHC_alpha', 'MHC_beta', 'Epitope']])
        has_error = True
    if not too_long.empty:
        st.error(f"Epitopes longer than {MAX_EPITOPE_LENGTH} residues are not supported ({len(too_long)} row(s); the longest has {too_long['Epitope'].str.len().max()}).")
        st.dataframe(too_long[['MHC_alpha', 'MHC_beta', 'Epitope']])
        has_error = True
    if not long_chain.empty:
        st.error(f"HLA chains longer than {MAX_CHAIN_LENGTH} residues are not supported ({len(long_chain)} row(s)); the longest chain of the allele catalogue has 285.")
        st.dataframe(long_chain[['MHC_alpha', 'MHC_beta', 'Epitope']])
        has_error = True
    if not invalid_length.empty and hard_check:
        st.warning(f"Peptides of 15 residues are preferred; other lengths up to {MAX_EPITOPE_LENGTH} are accepted, with lower accuracy.")
    if not invalid_alpha.empty or not invalid_beta.empty:
        st.error("Invalid characters found in MHC allele columns.")
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

if not uploaded:
    with st.form("manual_input"):
        cols = st.columns(3)
        mhc_alpha = cols[0].text_input("MHC α sequence", max_chars=MAX_CHAIN_LENGTH)
        mhc_beta = cols[1].text_input("MHC β sequence", max_chars=MAX_CHAIN_LENGTH)
        epitope = cols[2].text_input("Epitope", max_chars=MAX_EPITOPE_LENGTH)
        submitted = st.form_submit_button("Add Entry")
        if epitope and not all(aa in "ACDEFGHIKLMNPQRSTVWY" for aa in epitope.upper()):
            st.error("Epitope must contain only valid amino acids.")
        elif submitted and mhc_alpha and mhc_beta and epitope:
            if len(epitope) != 15:
                st.warning(f"Peptides of 15 residues are preferred; other lengths up to {MAX_EPITOPE_LENGTH} are accepted, with lower accuracy.")

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

@st.cache_data(show_spinner=False, max_entries=2048)
def chain_report(chain):
    """The locator's verdict on one chain (cached: every change of a widget reruns the page)."""
    report = get_engine().chain_report(chain)
    report.pop("hit")
    return report


MAX_LISTED_CHAINS = 50  # more distinct chains outside the catalogue than this are located when the prediction runs, without a table to edit
windows = {}
if not st.session_state['input_df'].empty:
    frame = st.session_state['input_df']
    chains = [c for c in dict.fromkeys(c.strip().upper() for col in ('MHC_alpha', 'MHC_beta') for c in frame[col]) if c not in get_engine().chain_names]
    if chains and len(chains) <= MAX_LISTED_CHAINS:
        st.markdown("#### Domains of chains outside the catalogue")
        st.caption("The model reads the domain of an HLA chain (58-82 residues). It is located by alignment to the allele catalogue; Start and End (residue numbers, from 1) can be corrected. "
                   "Where no domain is found, enter Start and End, or the full chain is used and the result is not validated.")
        reports = [chain_report(chain) for chain in chains]
        table = pd.DataFrame({
            "Chain": [f"{chain[:10]}... ({len(chain)} aa)" for chain in chains],
            "Nearest allele": [r['nearest'] or '-' for r in reports],
            "Identity": [round(r['identity'], 2) for r in reports],
            "Status": [{'located': 'located', 'low confidence': 'located, low confidence', 'no domain': 'no domain found'}[r['tier']] for r in reports],
            "Start": [None if r['start'] is None else r['start'] + 1 for r in reports],
            "End": [r['end'] for r in reports],
        })
        edited = st.data_editor(
            table, hide_index=True, width="stretch", key=f"windows_{hash(tuple(chains))}", disabled=["Chain", "Nearest allele", "Identity", "Status"],
            column_config={"Start": st.column_config.NumberColumn(min_value=1, step=1), "End": st.column_config.NumberColumn(min_value=1, step=1)},
        )
        for chain, report, (_, row) in zip(chains, reports, edited.iterrows()):
            start, end = row["Start"], row["End"]
            if pd.notna(start) and pd.notna(end) and (int(start) - 1, int(end)) != (report['start'], report['end']):
                windows[chain] = (int(start) - 1, int(end))
    elif chains:
        st.info(f"{len(chains)} distinct chains are not in the allele catalogue: their domains are located by alignment when you run the prediction.")

with st.expander("Options", expanded=False):
    batch_size = 128
    selected_model = st.selectbox("Select measurement type", ['Qualitative', 'Mass Spectrometry', 'IC50 (<500nM)', 'IC50 (<1000nM)'], index=0)
    show_top_binders = st.selectbox("Show top binders", ['All', 5, 10, 20, 50], index=2)
    if show_top_binders == 'All': show_top_binders = None
    plot_kde = st.checkbox("Plot KDE", value=True)

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
                head=selected_model,
                windows=windows
            )
            st.session_state['original_df'] = original_df
            st.session_state['result_df'] = result_df
            st.session_state['prediction_done'] = True
        st.success("Prediction complete!")

if st.session_state.get('prediction_done', False):
    original_df = st.session_state['original_df']
    result_df = st.session_state['result_df']
    st.markdown("# Results")
    counts, n_rows = original_df['HLA chains'].value_counts(), len(original_df)
    notes = {
        'catalogue': (st.info, "both HLA chains are in the allele catalogue and were scored exactly as Standard Mode scores these alleles"),
        'located': (st.info, "the domain of a chain outside the catalogue was located by alignment (identity 0.5 or more) and cut out of its embedding, as the model was trained"),
        'manual window': (st.info, "the domain window of a chain was the one you gave"),
        'low confidence': (st.warning, "the domain of a chain was located with an identity of only 0.4 to 0.5 to the nearest catalogue chain: check its window"),
        'full chain': (st.warning, "a chain has no domain found and no window was given: it was run on the full chain, which is not what the model was trained on (a domain of 58-82 residues), and is not validated"),
    }
    for label, (show, text) in notes.items():
        if counts.get(label, 0):
            show(f"{counts[label]} of {n_rows} rows: {text} (column 'HLA chains': {label}).")
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