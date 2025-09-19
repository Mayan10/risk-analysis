# pyg_scaffold.py
# ---
# This file is a ready-to-run scaffold showing:
# 1) how to convert a CSV of node features + an edge list into a torch_geometric.data.Data object
# 2) a simple GNN model (GCN -> MLP) that outputs a risk score per node.
#
# Requirements:
# pip install torch torch_geometric pandas scikit-learn
#
# Usage (example):
# python pyg_scaffold.py --nodes_csv supplychain_node_features_example.csv --edges_csv supplychain_edges_example.csv
#
import argparse
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from typing import Tuple
import math

class SimpleGNNRisk(nn.Module):
    def __init__(self, in_channels, hidden=64):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, 1),
            nn.Sigmoid(),  # risk score in [0,1]
        )
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        out = self.mlp(x).squeeze(-1)
        return out

def build_graph_from_csv(nodes_csv, edges_csv):
    nodes = pd.read_csv(nodes_csv)
    edges = pd.read_csv(edges_csv)  # columns: src,dst (node_id strings)
    # Build node feature matrix (select numeric columns)
    feature_cols = [
        "news_count_1d","news_count_7d","neg_tone_frac_3d","weather_anomaly_7d",
        "strike_flag_7d","avg_lead_time_days","inventory_days","single_sourced",
        "past_delay_days","news_velocity"
    ]
    X = nodes[feature_cols].fillna(0).values.astype(float)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    x = torch.tensor(X, dtype=torch.float)
    # Create mapping from node_id -> index
    node_id_to_idx = {nid: i for i,nid in enumerate(nodes['node_id'].tolist())}
    # Convert edges
    src_idx = [node_id_to_idx[s] for s in edges['src'].tolist()]
    dst_idx = [node_id_to_idx[d] for d in edges['dst'].tolist()]
    edge_index = torch.tensor([src_idx + dst_idx, dst_idx + src_idx], dtype=torch.long)  # undirected
    y = torch.tensor(nodes['disruption_within_7d'].fillna(0).values, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, y=y)
    return data

def create_train_val_masks(num_nodes: int, val_fraction: float, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_nodes, generator=generator)
    num_val = max(1, int(math.ceil(val_fraction * num_nodes)))
    val_idx = perm[:num_val]
    train_idx = perm[num_val:]
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    return train_mask, val_mask

def train_and_eval(model: nn.Module, data: Data, epochs: int, lr: float, weight_decay: float, train_mask: torch.Tensor, val_mask: torch.Tensor, device: str = "cpu") -> None:
    model.to(device)
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        out = model(x, edge_index)
        loss = criterion(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred = (out >= 0.5).float()
            # Train metrics
            train_correct = (pred[train_mask] == y[train_mask]).float().mean().item() if train_mask.any() else float("nan")
            # Val metrics
            val_loss = criterion(out[val_mask], y[val_mask]).item() if val_mask.any() else float("nan")
            val_acc = (pred[val_mask] == y[val_mask]).float().mean().item() if val_mask.any() else float("nan")

        if epoch % max(1, epochs // 10) == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch {epoch:03d} | train_loss={loss.item():.4f} train_acc={train_correct:.3f} val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes_csv", required=True)
    parser.add_argument("--edges_csv", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--val_frac", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    data = build_graph_from_csv(args.nodes_csv, args.edges_csv)
    model = SimpleGNNRisk(in_channels=data.num_node_features)
    print("Graph built. Number of nodes:", data.num_nodes, "Num edges:", data.num_edges)
    train_mask, val_mask = create_train_val_masks(data.num_nodes, args.val_frac, args.seed)
    train_and_eval(model, data, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, train_mask=train_mask, val_mask=val_mask, device=args.device)