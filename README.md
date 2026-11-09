# DPP-4 pIC50 Inference Web App

Gradio app. Models are loaded from Hugging Face on demand.
Deploy as a Hugging Face Space; link it from GitHub Pages.

## Use
- Single SMILES tab: paste one SMILES, get predicted pIC50 / IC50 / class.
- Batch CSV tab: upload a CSV with a `smiles` column (optional
  `compound_id`), get a preview table, two plots (pIC50 distribution,
  predicted vs wet-lab), and download the predictions as CSV.
- Model dropdown: ExtraTrees, RandomForest, XGBoost, DNN, LTN (NeSy),
  GAT (graph), Transformer, ChemBERTa (LLM).

## Status
PREPARED 2026-11-08. NOT yet tested — testing waits until training
finishes and the HF model repos exist. The FDA-12 dataset will also
be uploaded to Hugging Face as a dataset repo for the comparison plot.
