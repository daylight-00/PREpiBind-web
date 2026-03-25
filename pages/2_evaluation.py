import streamlit as st
import pandas as pd
import os
import sys
import plotly.figure_factory as ff
import plotly.graph_objects as go
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, balanced_accuracy_score, precision_score, precision_recall_curve, auc, matthews_corrcoef

from app import get_models
from app import write_st_end

models = get_models()

utils_path = os.path.abspath('code')
sys.path.insert(0, utils_path)
from inference import main, load_config

st.set_page_config(
    page_title="Evaluation | PREpiBind",
    page_icon=":dna:",
    layout="centered",
    # initial_sidebar_state="collapsed"
)
st.title("Evaluation")
st.markdown("""
For detailed instruction, see [Instructions](/prepibind/instructions#evaluation-and-benchmarking) page.
""")
            
selected_model = st.selectbox("Select measurement type", ['Qualitative', 'Mass Spectrometry', 'IC50 (<500nM)', 'IC50 (<1000nM)'], index=0)
model = models.get(selected_model, None)
df_path = st.session_state.data_paths.get(selected_model, None)

csv_file_eval = st.file_uploader("Upload CSV", type="csv", key="csv_upload")

# CSV upload handling
if csv_file_eval is not None:
    df_csv = pd.read_csv(csv_file_eval)
    if all(col in df_csv.columns for col in ["MHC_alpha", "MHC_beta", "Epitope", "Target"]):
        total_rows = len(df_csv)
        records = df_csv[["MHC_alpha", "MHC_beta", "Epitope", "Target"]]
        st.success(f"CSV file loaded. Total samples: {total_rows}")
        st.dataframe(records, width="stretch")
    else:
        st.error("CSV must contain columns: 'MHC_alpha', 'MHC_beta', 'Epitope', 'Target'.")
else:
    df_csv = pd.read_csv(df_path)[["MHC_alpha", "MHC_beta", "Epitope", "Target"]]
    total_rows = len(df_csv)
    st.info(f"Use the test set from the original paper. Total samples: {total_rows}")
    csv = df_csv.to_csv(index=False).encode()
    file_name = df_path.split("/")[-1]
    st.download_button("Download original test set", csv, file_name, "text/csv", key="download_test_set")
    st.dataframe(df_csv, width="stretch")
df_csv['MHC'] = df_csv['MHC_alpha'] + '_' + df_csv['MHC_beta']

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

    df_org = main(config, model)
    cols = ['Score', 'Logits']
    return df_org

custom_palette = ["#FFAF00", "#F46920", "#F53255", "#F857C1", "#29BDFD", "#00CBBF", "#01C159", "#9DCA1C"]

def plot_multibar(scores, targets):
    st.markdown("### Performance Metrics")
    bin_scores = (scores > 0.5).astype(int)
    roc_auc  = roc_auc_score(targets, scores)
    precision_vals, recall_vals, _ = precision_recall_curve(targets, scores)
    pr_auc = auc(recall_vals, precision_vals)
    f1 = f1_score(targets, bin_scores)
    precision = precision_score(targets, bin_scores)
    mcc = matthews_corrcoef(targets, bin_scores)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=['ROC AUC', 'PR AUC', 'F1 Score', 'Precision', 'MCC'],
        y=[roc_auc, pr_auc, f1, precision, mcc],
        marker_color=custom_palette,
        text=[f"{roc_auc:.2f}", f"{pr_auc:.2f}",
                f"{f1:.2f}", f"{precision:.2f}", f"{mcc:.2f}"],
        # textposition='auto',
        hoverinfo='text',
        hovertext=[
            f"ROC AUC: {roc_auc:.2f}",
            f"PR AUC: {pr_auc:.2f}",
            f"F1 Score: {f1:.2f}",
            f"Precision: {precision:.2f}",
            f"MCC: {mcc:.2f}"
        ]
    ))
    st.plotly_chart(fig)

def plot_roc_curve(scores, targets):
    st.markdown("### Receiver Operating Characteristic (ROC) Curve")
    fpr, tpr, _ = roc_curve(targets, scores)
    roc_auc = auc(fpr, tpr)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode='lines',
        name=f'AUC = {roc_auc:.2f}',
        line=dict(color='#29BDFD'),
        fill='tozeroy'   # 곡선 아래 채우기
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode='lines',
        name='Random', line=dict(dash='dash', width=1),
        showlegend=False
    ))
    fig.update_layout(
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        showlegend=True,
        # yaxis=dict(scaleanchor="x", scaleratio=1),
        # xaxis_range=[0, 1],
    )
    st.plotly_chart(fig)

def plot_pr_curve(scores, targets):
    st.markdown("### Precision-Recall Curve")
    precision, recall, _ = precision_recall_curve(targets, scores)
    pr_auc = auc(recall, precision)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recall, 
        y=precision, 
        mode='lines', 
        name=f'AUC = {pr_auc:.2f}',
        fill='tozeroy',
        line=dict(color='#F46920'),
    ))
    fig.update_layout(
        xaxis_title='Recall',
        yaxis_title='Precision',
        showlegend=True,
        # yaxis=dict(scaleanchor="x", scaleratio=1),
        # xaxis_range=[0, 1],
    )
    st.plotly_chart(fig)

def plot_plot(df_epi):
    scores = df_epi['Score'].astype(float).values
    targets = df_epi['Target'].astype(float).values
    scores = scores.flatten()
    targets = targets.flatten()
    plot_multibar(scores, targets)
    plot_roc_curve(scores, targets)
    plot_pr_curve(scores, targets)


if st.button("Run Prediction"):
    with st.spinner("Running prediction..."):
        original_df = run_prepibind(
            df=df_csv,
            model=model
        )
    st.success("Prediction complete!")
    st.markdown("# Results")
    st.dataframe(original_df, width="stretch")
    st.markdown("## Benchmark")
    plot_plot(original_df)

    st.markdown("### Download Results")
    csv = original_df.to_csv(index=False).encode()
    st.download_button("Download result", csv, file_name="prediction.csv", mime="text/csv")

write_st_end()