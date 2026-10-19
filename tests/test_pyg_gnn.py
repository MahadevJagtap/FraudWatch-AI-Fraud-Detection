"""
Automated unit tests for PyTorch Geometric (PyG) Graph Neural Network.
"""

from __future__ import annotations

import json
import os
import torch
import pytest

import importlib
pyg_module = importlib.import_module("src.09_train_pyg_gnn")
FraudGraphSAGE = pyg_module.FraudGraphSAGE

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "pyg_fraud_gnn_model.pt")
REPORT_PATH = os.path.join(PROJECT_ROOT, "reports", "pyg_gnn_evaluation_report.json")


def test_pyg_artifacts_exist():
    assert os.path.isfile(MODEL_PATH), f"PyG model weights missing: {MODEL_PATH}"
    assert os.path.getsize(MODEL_PATH) > 0, "PyG model file is empty"
    assert os.path.isfile(REPORT_PATH), f"Report missing: {REPORT_PATH}"

    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        rep = json.load(f)

    assert rep["model_name"] == "pyg_fraud_graphsage"
    assert "blending_analysis" in rep
    assert rep["num_nodes"] > 1000
    assert rep["test_roc_auc"] > 0.80, f"Expected ROC-AUC > 0.80, got {rep['test_roc_auc']}"


def test_pyg_forward_inference():
    model = FraudGraphSAGE(in_channels=3, hidden_channels=16, edge_channels=4)
    model.eval()

    # Create dummy 4-node, 3-edge bipartite graph
    x = torch.randn((4, 3), dtype=torch.float32)
    edge_index = torch.tensor([[0, 1, 2], [3, 3, 3]], dtype=torch.long)
    edge_attr = torch.randn((3, 4), dtype=torch.float32)

    with torch.no_grad():
        logits = model(x, edge_index, edge_attr)

    assert logits.shape == (3,)
    probs = torch.sigmoid(logits)
    assert torch.all((probs >= 0.0) & (probs <= 1.0))
