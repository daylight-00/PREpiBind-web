import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import scan
from app import get_engine
from app import predict
from app import write_st_end

st.set_page_config(
    page_title="Scan | PREpiBind",
    page_icon=":dna:",
    layout="centered",
)
st.title("Protein Scan")
st.markdown(
    f"""
    Score **every window of a protein** against a panel of HLA class II alleles, to find the regions that bind.

    - Paste a protein sequence or a FASTA text, or upload a FASTA file (up to {scan.MAX_PROTEIN_LENGTH:,} residues).
    - Choose the alleles: a preset panel or your own list (up to {scan.MAX_ALLELES}).
    - A scan may make at most {scan.MAX_SCAN_ROWS:,} predictions (windows x alleles); a protein of 500 residues against the 27-allele panel makes about 13,000.
    """
)

alpha_names = set(st.session_state["alpha_list"])
beta_names = set(st.session_state["beta_list"])
all_names = alpha_names | beta_names

st.subheader("Protein")
fasta_file = st.file_uploader("FASTA file (optional)", type=["fa", "fasta", "faa", "txt"])
pasted = st.text_area("Protein sequence or FASTA text", height=150, placeholder=">name\nMKTAYIAKQR...")
source = fasta_file.getvalue().decode("utf-8", errors="replace") if fasta_file else pasted

protein, protein_name, input_error = "", "", None
try:
    records = scan.read_proteins(source)
except scan.ScanRefused as error:
    records, input_error = [], str(error)
if input_error:
    st.error(input_error)
elif records:
    protein_name, protein = records[0]
    if len(records) > 1:
        protein_name = st.selectbox("The input has several proteins; scan", [name for name, _ in records])
        protein = dict(records)[protein_name]
    st.caption(f"{protein_name}: {len(protein):,} residues")

st.subheader("Alleles")
choice = st.selectbox("Panel", ["Reference panel (27 alleles: DR, DQ and DP)", "DRB1 panel (11 alleles)", "My own list"])
typed = ""
if choice == "My own list":
    typed = st.text_area(
        "One allele pair per line",
        height=120,
        placeholder="HLA-DRB1*15:01\nHLA-DQA1*05:01, HLA-DQB1*02:01",
        help="For HLA-DR a beta chain alone is enough (its alpha chain is HLA-DRA*01:01). Names must be in the allele catalogue.",
    )

st.subheader("Window")
lengths = st.multiselect(
    "Window lengths",
    options=list(scan.WINDOW_LENGTHS),
    default=[scan.DEFAULT_WINDOW],
    help="The model was trained on 15-mers. Other lengths are possible but less reliable (built-in Qualitative test set, random flanks: ROC AUC 0.93 at 15 residues, 0.87 at 20).",
)
if sorted(lengths) != [scan.DEFAULT_WINDOW]:
    st.info("Windows other than 15 residues are scored less reliably than 15-mers, and non-binders more often as binders; see Instructions, Limitations.")
selected_model = st.selectbox("Select measurement type", ["Qualitative", "Mass Spectrometry", "IC50 (<500nM)", "IC50 (<1000nM)"], index=0)


def run_scan():
    if not protein:
        st.error("No protein sequence provided.")
        return
    try:
        if choice == "My own list":
            pairs = scan.parse_panel(typed, alpha_names, beta_names)
        elif choice.startswith("Reference"):
            pairs = scan.reference_panel(all_names)
        else:
            pairs = scan.drb1_panel(all_names)
        windows = scan.check_scan(len(protein), lengths, len(pairs), fast=get_engine().plans is not None)
    except scan.ScanRefused as error:
        st.error(str(error))
        return
    frame = scan.scan_frame(protein, lengths, pairs)
    labels = {f"{a}_{b}": scan.short_label(a, b) for a, b in pairs}
    started = time.perf_counter()
    with st.spinner(f"Scoring {len(frame):,} windows x alleles..."):
        result = predict(frame, selected_model)
    result["Allele"] = result["MHC"].map(labels)
    result["End"] = result["Start"] + result["Length"] - 1
    st.session_state["scan"] = {"result": result, "name": protein_name, "protein": protein, "windows": windows, "alleles": len(pairs), "seconds": time.perf_counter() - started, "model": selected_model}


if st.button("Run Scan"):
    run_scan()

if "scan" in st.session_state:
    done = st.session_state["scan"]
    result, labels_in_order = done["result"], list(dict.fromkeys(done["result"]["Allele"]))
    st.markdown("# Results")
    st.caption(f"{done['name']}: {len(result):,} predictions ({done['windows']:,} windows x {done['alleles']} alleles, {done['model']}) in {done['seconds']:.1f} s")

    st.markdown("### Top binders")
    top = result.nlargest(20, "Score")[["Start", "End", "Epitope", "Allele", "Score", "Logits"]].rename(columns={"Epitope": "Peptide"})
    top["Score"] = top["Score"].map(lambda x: f"{x:.5f}")
    top["Logits"] = top["Logits"].astype(float).map(lambda x: f"{x:.5f}")
    st.dataframe(top, hide_index=True, width="stretch")

    st.markdown("### Windows bound by many alleles")
    per_window = result.assign(Bound=result["Score"] > 0.5).groupby(["Start", "End", "Epitope"], sort=False).agg(Alleles_above_0_5=("Bound", "sum"), Mean_score=("Score", "mean")).reset_index()
    best = per_window.sort_values(["Alleles_above_0_5", "Mean_score"], ascending=False).head(10).rename(columns={"Epitope": "Peptide", "Alleles_above_0_5": "Alleles above 0.5", "Mean_score": "Mean score"})
    best["Mean score"] = best["Mean score"].map(lambda x: f"{x:.3f}")
    st.dataframe(best, hide_index=True, width="stretch")

    length = st.selectbox("Window length shown in the plots", sorted(result["Length"].unique()), index=0) if result["Length"].nunique() > 1 else int(result["Length"].iloc[0])
    shown = result[result["Length"] == length]
    matrix = shown.pivot(index="Allele", columns="Start", values="Score").loc[labels_in_order]
    st.markdown("### Score along the protein")
    heat = go.Figure(go.Heatmap(z=matrix.to_numpy(), x=matrix.columns, y=matrix.index, zmin=0, zmax=1, colorscale="YlOrRd", colorbar={"title": "Score"}, hovertemplate="start %{x}<br>%{y}<br>score %{z:.3f}<extra></extra>"))
    heat.update_layout(xaxis_title="Window start (residue)", yaxis={"autorange": "reversed"}, height=max(260, 24 * len(matrix) + 120))
    st.plotly_chart(heat)
    count = (matrix > 0.5).sum(axis=0)
    bars = go.Figure(go.Bar(x=count.index, y=count.to_numpy(), marker_color="#F46920"))
    bars.update_layout(xaxis_title="Window start (residue)", yaxis_title="Alleles with a score above 0.5", height=260)
    st.plotly_chart(bars)

    st.markdown("### Download Results")
    alleles = result["MHC"].str.split("_", n=1, expand=True)
    table = pd.DataFrame({"Start": result["Start"], "End": result["End"], "Peptide": result["Epitope"], "MHC_alpha": alleles[0], "MHC_beta": alleles[1], "Score": result["Score"], "Logits": result["Logits"]})
    st.download_button("Download result", table.to_csv(index=False).encode(), file_name="scan.csv", mime="text/csv")

write_st_end()
