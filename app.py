"""DPP-4 pIC50 Inference Web App — Gradio UI, models loaded from Hugging Face.
Deploy: Hugging Face Space (Gradio) AND GitHub Pages link to the Space.
This app is prepared now; it is TESTED only after training finishes,
once the HF repos below actually exist.

Models (HF repos, loaded on demand by model name):
  - extra-trees      Delower/dpp4-pic50-extra-trees
  - random-forest    Delower/dpp4-pic50-random-forest
  - xgboost          Delower/dpp4-pic50-xgboost
  - dnn              Delower/dpp4-pic50-dnn
  - ltn              Delower/dpp4-pic50-ltn        (NeSy / LTN)
  - gat              Delower/dpp4-pic50-gat        (graph)
  - transformer      Delower/dpp4-pic50-transformer (SMILES Transformer)
  - chemberta        Delower/dpp4-pic50-chemberta  (LLM / ChemBERTa)

Features: single SMILES input, or batch CSV upload (smiles column,
optional compound_id). Preview table, predicted pIC50 / IC50 / class,
download CSV, and two plots (pIC50 distribution, predicted vs wet-lab
for the FDA-12) downloadable as PNG.
"""
import io, os
import numpy as np
import pandas as pd
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_REPOS = {
    "ExtraTrees (best)": "Delower/dpp4-pic50-extra-trees",
    "RandomForest": "Delower/dpp4-pic50-random-forest",
    "XGBoost (tuned)": "Delower/dpp4-pic50-xgboost",
    "DNN": "Delower/dpp4-pic50-dnn",
    "LTN (NeSy)": "Delower/dpp4-pic50-ltn",
    "GAT (graph)": "Delower/dpp4-pic50-gat",
    "Transformer": "Delower/dpp4-pic50-transformer",
    "ChemBERTa (LLM)": "Delower/dpp4-pic50-chemberta",
}
_cache = {}

def load_model(name):
    """Load a model from its HF repo on first use, cache after."""
    if name in _cache:
        return _cache[name]
    from huggingface_hub import hf_hub_download
    import pickle
    repo = MODEL_REPOS[name]
    # model file name is <model>_<features>.pkl, recorded in config.json
    from huggingface_hub import list_repo_files
    files = list_repo_files(repo, repo_type="model")
    pkl = [f for f in files if f.endswith(".pkl")][0]
    path = hf_hub_download(repo_id=repo, filename=pkl, repo_type="model")
    with open(path, "rb") as f:
        model = pickle.load(f)
    _cache[name] = model
    return model

def predict_df(df, model_name):
    model = load_model(model_name)
    # featurise with the model's own feature code (Morgan+physchem
    # for tree models; the model pipeline carries its preprocessor)
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    X = []
    for s in df["smiles"]:
        mol = Chem.MolFromSmiles(s)
        fp = list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024))
        phys = [Descriptors.MolWt(mol), Descriptors.MolLogP(mol),
                Descriptors.TPSA(mol), Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.NumRotatableBonds(mol)]
        X.append(fp + phys)
    df = df.copy()
    Xdf = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(1030)])
    df["predicted_pIC50"] = model.predict(Xdf)
    df["predicted_IC50_nM"] = 10 ** (9 - df["predicted_pIC50"])
    df["predicted_class"] = np.where(df["predicted_pIC50"] >= 7, "potent",
        np.where(df["predicted_pIC50"] >= 6, "moderate", "weak"))
    return df

def plot_dist(df):
    fig, ax = plt.subplots()
    ax.hist(df["predicted_pIC50"], bins=20)
    ax.set_xlabel("Predicted pIC50"); ax.set_ylabel("Count")
    ax.set_title("Predicted pIC50 distribution")
    return fig

def plot_fda(df):
    # predicted vs wet-lab, only for rows that have a wet-lab value
    if "wetlab_pIC50" not in df.columns or df["wetlab_pIC50"].dropna().empty:
        return None
    sub = df.dropna(subset=["wetlab_pIC50"])
    fig, ax = plt.subplots()
    ax.scatter(sub["wetlab_pIC50"], sub["predicted_pIC50"])
    ax.plot([4, 10], [4, 10], "--")
    ax.set_xlabel("Wet-lab pIC50"); ax.set_ylabel("Predicted pIC50")
    ax.set_title("Predicted vs wet-lab pIC50 (FDA-12)")
    return fig

def single(smiles, model_name):
    df = pd.DataFrame({"compound_id": ["input"], "smiles": [smiles]})
    out = predict_df(df, model_name)
    return out, plot_dist(out), plot_fda(out)

def batch(file, model_name):
    df = pd.read_csv(file.name)
    out = predict_df(df, model_name)
    csv = out.to_csv(index=False)
    return out, plot_dist(out), plot_fda(out), csv

with gr.Blocks(title="Computational Chemistry and Drug Discovery — DPP-4 pIC50 Predictor") as demo:
    gr.Markdown("# Computational Chemistry and Drug Discovery\n## DPP-4 pIC50 Predictor\nSMILES in → predicted pIC50 / IC50. Models are loaded from Hugging Face on demand. Not for clinical use.")
    gr.Markdown("---")
    gr.Markdown("© 2026 Delower Hossain — Computational Chemistry and Drug Discovery. All rights reserved. | Models are in-silico predictions only, not for clinical use.")
    model = gr.Dropdown(list(MODEL_REPOS), value="ExtraTrees (best)", label="Model")
    with gr.Tab("Single SMILES"):
        smi = gr.Textbox(label="SMILES", placeholder="CC(=O)Oc1ccccc1C(=O)O")
        btn = gr.Button("Predict")
        out1 = gr.Dataframe(label="Prediction preview")
        p1 = gr.Plot(label="pIC50 distribution")
        p2 = gr.Plot(label="Predicted vs wet-lab (FDA-12, if applicable)")
        btn.click(single, [smi, model], [out1, p1, p2])
    with gr.Tab("Batch CSV"):
        f = gr.File(label="CSV with a 'smiles' column (optional 'compound_id')")
        btn2 = gr.Button("Predict batch")
        out2 = gr.Dataframe(label="Prediction preview")
        q1 = gr.Plot(label="pIC50 distribution")
        q2 = gr.Plot(label="Predicted vs wet-lab (FDA-12, if applicable)")
        dl = gr.DownloadButton(label="Download predictions (CSV)")
        btn2.click(batch, [f, model], [out2, q1, q2, dl])
    gr.Markdown("---\n© 2026 Delower Hossain — Computational Chemistry and Drug Discovery. All rights reserved.")

if __name__ == "__main__":
    demo.launch()
