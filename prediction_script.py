# -*- coding: utf-8 -*-
"""
Pks13 Inhibitor Prediction:
    Input: Csv file with SMILES column (input.csv)
    Output: Folder (result) containing prediction file and image folders
"""

import os
import io
import torch
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader as PyGDataLoader
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image, ImageDraw, ImageFont

from training_advanced import EdgeAwareGCNNet, compute_grad_cam_for_batch
from Smiles_to_graph import smiles_to_graph
from torch_geometric.explain import Explainer, GNNExplainer

# ---------------- CONFIG ---------------- #
INPUT_CSV = "input.csv"
MODEL_PATH = "gnn_model.pt"
OUT_DIR = "result"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 16   # safer for stability
TOP_K = 10
THRESHOLD = 0.5

CLASS_MAP = {0: "Non-Inhibitor", 1: "Inhibitor"}

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- DETERMINISTIC ---------------- #
torch.manual_seed(42)
np.random.seed(42)
torch.use_deterministic_algorithms(True)

# ---------------- DRAW FUNCTION ---------------- #
def draw_molecule(smiles, top_atoms, save_path, pred_label, prob):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return

    if mol.GetNumConformers() == 0:
        Chem.rdDepictor.Compute2DCoords(mol)

    drawer = rdMolDraw2D.MolDraw2DCairo(900, 450)
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        highlightAtoms=top_atoms,
        highlightAtomColors={i: (1, 0.4, 0.4) for i in top_atoms}
    )
    drawer.FinishDrawing()

    img = Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()

    text = f"Pred: {pred_label}\nProb: {prob:.3f}"
    draw.rectangle([5, 5, 260, 60], fill=(255, 255, 255))
    draw.multiline_text((10, 10), text, fill=(0, 0, 0), font=font)

    img.save(save_path)


class WrappedModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, edge_index, edge_attr, batch):
        out, _ = self.model(x, edge_index, edge_attr, batch)
        return out
# ---------------- LOAD MODEL ---------------- #
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

base_model = EdgeAwareGCNNet(
    in_channels=checkpoint["in_channels"],
    edge_dim=checkpoint["edge_dim"],
    hidden_dim=checkpoint["hidden_dim"]
).to(DEVICE)

base_model.load_state_dict(checkpoint["model_state_dict"])
base_model.eval()

model = WrappedModel(base_model).to(DEVICE)  # for GNNExplainer
model.eval()


explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=100),  # keep moderate for speed
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(
        mode='binary_classification',
        task_level='graph',
        return_type='raw',
    ),
)

feature_mean = checkpoint["feature_mean"]
feature_std = checkpoint["feature_std"]

print("Model loaded successfully")


# ---------------- LOAD DATA ---------------- #
df = pd.read_csv(INPUT_CSV)

if "SMILES" not in df.columns:
    raise ValueError("CSV must contain 'SMILES' column")

df = df.reset_index(drop=True)
smiles_list = df["SMILES"].tolist()

graphs = []
valid_smiles = []

for s in smiles_list:
    g = smiles_to_graph(s, 0.0)
    if g is not None:
        g.smiles = s  # IMPORTANT
        graphs.append(g)
        valid_smiles.append(s)

# Normalize
for g in graphs:
    g.x = (g.x - feature_mean) / feature_std

print(f"Valid molecules: {len(graphs)}")


# ---------------- MAIN INFERENCE ---------------- #
def run_inference(model, dataset):
    loader = PyGDataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    results = []
    img_dir = os.path.join(OUT_DIR, "gradcam_images")
    os.makedirs(img_dir, exist_ok=True)
    gnn_img_dir = os.path.join(OUT_DIR, "gnnexplainer_images")
    os.makedirs(gnn_img_dir, exist_ok=True)

    idx_global = 0

    for batch in loader:
        batch = batch.to(DEVICE)

        # ---- Prediction ----
        with torch.no_grad():
            out, _ = base_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            probs = torch.sigmoid(out).view(-1).cpu().numpy()
            preds = (probs > THRESHOLD).astype(int)

        scores = compute_grad_cam_for_batch(base_model, batch, target_class=1)
        data_list = batch.to_data_list()

        for i in range(len(data_list)):
            data = data_list[i]
            smiles = data.smiles
            
            ##--------------------- Grad-CAM ----------------#
            gcam = scores[i]

            if gcam is None or len(gcam) == 0:
                gcam_top_atoms, gcam_atom_scores, atom_labels = [], [], []
            else:
                k = min(TOP_K, len(gcam))
                idxs = np.argsort(gcam)[-k:][::-1]

                gcam_top_atoms = idxs.tolist()
                gcam_atom_scores = [float(gcam[x]) for x in gcam_top_atoms]

##-------------- GNNExplainer----------------#
            explanation = explainer(
                x=data.x,
                edge_index=data.edge_index,
                edge_attr=data.edge_attr,
                batch=torch.zeros(data.x.size(0), dtype=torch.long).to(data.x.device)
                )
            node_mask = explanation.node_mask
            
            if node_mask is None:
                gnn_top_atoms, gnn_scores = [], []
            else:
                node_mask = node_mask.detach().cpu().numpy()
                # ---- FIX: reduce feature dimension ---- #
                if node_mask.ndim == 2:
                    #node_importance = node_mask.mean(axis=1)   # or sum(axis=1)
                    node_importance = np.linalg.norm(node_mask, axis=1)
                else:
                    node_importance = node_mask
                
                k = min(TOP_K, len(node_importance))
                idxs = np.argsort(node_importance)[-k:][::-1]
                
                gnn_top_atoms = idxs.tolist()
                gnn_scores = [float(node_importance[x]) for x in gnn_top_atoms]
                
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                gcam_atom_labels = [mol.GetAtomWithIdx(x).GetSymbol() for x in gcam_top_atoms]
                gnn_atom_labels = [mol.GetAtomWithIdx(x).GetSymbol() for x in gnn_top_atoms]
            else:
                gcam_atom_labels = []
                gnn_atom_labels = []

            pred = int(preds[i])
            prob = float(probs[i])
            pred_label = CLASS_MAP[pred]

            img_path = os.path.join(img_dir, f"mol_{idx_global}.png")
            gnnexp_img_path = os.path.join(gnn_img_dir, f"mol_{idx_global}.png")

            draw_molecule(smiles, gcam_top_atoms, img_path, pred_label, prob)
            draw_molecule(smiles, gnn_top_atoms, gnnexp_img_path, pred_label, prob)
            
            results.append({
                "SMILES": smiles,
                "Predicted": pred,
                "Predicted_Label": pred_label,
                "Probability": prob,
                "GCAM_Top_Atoms": ";".join(map(str, gcam_top_atoms)),
                "GCAM_Atom_Scores": ";".join([f"{x:.4f}" for x in gcam_atom_scores]),
                "GCAM_Atom_Labels": ";".join(gcam_atom_labels),

                "GCAM_Image_Path": img_path,
                
                # GNNExplainer
                "GNN_Top_Atoms": ";".join(map(str, gnn_top_atoms)),
                "GNN_Scores": ";".join([f"{x:.4f}" for x in gnn_scores]),
                "GNN_Atom_Labels": ";".join(gnn_atom_labels),
                "GNN_Image_Path": gnnexp_img_path,
                "Overlap_Atoms": ";".join(map(str, set(gcam_top_atoms) & set(gnn_top_atoms)))
            })

            idx_global += 1

    return pd.DataFrame(results)


# ---------------- RUN ---------------- #
print("Running inference...")
results_df = run_inference(model, graphs)

# ---------------- SAVE ---------------- #
output_csv = os.path.join(OUT_DIR, "predictions_with_gradcam.csv")
results_df.to_csv(output_csv, index=False)

print("DONE")
print(f"Results saved at: {output_csv}")
