"""Parameter-matched beamforming-only SecGNN baseline (Song et al., TVT 2024)."""

from __future__ import annotations
import math
from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F

class SecGNNGraphPartitionLayer(nn.Module):
    """Construct UU, UE and EE relation masks, including self connections."""
    def __init__(self):
        super().__init__()

    def forward(self, K: int, J: int, device: torch.device) -> torch.Tensor:
        M = K + J
        masks = torch.zeros(3, M, M, dtype=torch.bool, device=device)
        if K > 0:
            masks[0, :K, :K] = True
        if K > 0 and J > 0:
            masks[1, :K, K:] = True
            masks[1, K:, :K] = True
        if J > 0:
            masks[2, K:, K:] = True
        eye = torch.eye(M, dtype=torch.bool, device=device)
        return masks | eye.unsqueeze(0)

class SecGNNRelationGAL(nn.Module):
    """Multi-head graph attention for one relation-specific subgraph."""
    def __init__(self, input_dim: int, head_dim: int, n_heads: int, dropout: float = 0.1, negative_slope: float = 0.2):
        super().__init__()
        self.input_dim = input_dim
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.negative_slope = negative_slope
        self.dropout = nn.Dropout(dropout)

        self.W_dir = nn.ModuleList([nn.Linear(input_dim, head_dim, bias=False) for _ in range(n_heads)])
        self.W_ner = nn.ModuleList([nn.Linear(input_dim, head_dim, bias=False) for _ in range(n_heads)])
        self.att = nn.Parameter(torch.empty(n_heads, head_dim))
        self.reset_parameters()

    def reset_parameters(self):
        for h in range(self.n_heads):
            nn.init.xavier_uniform_(self.W_dir[h].weight)
            nn.init.xavier_uniform_(self.W_ner[h].weight)
        nn.init.xavier_uniform_(self.att)

    def forward(self, X: torch.Tensor, mask: torch.Tensor, return_attention: bool = False):
        B, M, D = X.shape
        if D != self.input_dim:
            raise ValueError(f"GAL input dim={D}, expected={self.input_dim}")
        outputs, attentions = [], []

        for h in range(self.n_heads):
            X_dir = self.W_dir[h](X)
            X_ner = self.W_ner[h](X)

            pair = X_dir.unsqueeze(2) + X_ner.unsqueeze(1)
            pair = F.leaky_relu(pair, negative_slope=self.negative_slope)
            score = torch.einsum("bijd,d->bij", pair, self.att[h])
            score = score.masked_fill(~mask.unsqueeze(0), torch.finfo(score.dtype).min)

            alpha = F.softmax(score, dim=-1)
            alpha_used = self.dropout(alpha)

            message = torch.bmm(alpha_used, X_ner)
            alpha_self = torch.diagonal(alpha_used, dim1=-2, dim2=-1).unsqueeze(-1)
            message = message - alpha_self * X_ner + alpha_self * X_dir

            outputs.append(message)
            if return_attention:
                attentions.append(alpha)

        X_out = torch.cat(outputs, dim=-1)
        if return_attention:
            return X_out, torch.stack(attentions, dim=1)
        return X_out

class SecGNNGraphAttentionLayer(nn.Module):
    """Apply separate graph attention parameters to UU, UE and EE relations."""
    def __init__(self, input_dim: int, head_dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.relations = nn.ModuleList([
            SecGNNRelationGAL(input_dim, head_dim, n_heads, dropout=dropout),
            SecGNNRelationGAL(input_dim, head_dim, n_heads, dropout=dropout),
            SecGNNRelationGAL(input_dim, head_dim, n_heads, dropout=dropout),
        ])

    def forward(self, X: torch.Tensor, masks: torch.Tensor, return_attention: bool = False):
        features, attention = [], {}
        for g in range(3):
            if return_attention:
                X_g, A_g = self.relations[g](X, masks[g], return_attention=True)
                attention[g] = A_g
            else:
                X_g = self.relations[g](X, masks[g], return_attention=False)
            features.append(X_g)
        if return_attention:
            return features, attention
        return features

class SecGNNSemanticAttentionLayer(nn.Module):
    """Compute sample-wise semantic weights across relation subgraphs."""
    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.Theta = nn.Linear(feature_dim, hidden_dim, bias=True)
        self.q = nn.Parameter(torch.empty(hidden_dim))
        nn.init.xavier_uniform_(self.Theta.weight)
        nn.init.zeros_(self.Theta.bias)
        nn.init.normal_(self.q, mean=0.0, std=1.0 / math.sqrt(hidden_dim))

    def forward(self, subgraph_features: List[torch.Tensor], return_weight: bool = False):
        scores = []
        for X_g in subgraph_features:
            Z = torch.tanh(self.Theta(X_g))
            node_score = torch.einsum("bmu,u->bm", Z, self.q)
            scores.append(node_score.mean(dim=1))
        gamma = F.softmax(torch.stack(scores, dim=-1), dim=-1)
        X_stack = torch.stack(subgraph_features, dim=1)
        X_out = torch.sum(gamma[:, :, None, None] * X_stack, dim=1)
        if return_weight:
            return X_out, gamma
        return X_out

class SecGNNGraphModule(nn.Module):
    """Graph module: FCL, relation-specific GAL, SAL and activation."""
    def __init__(self, input_dim: int, fcl_dim: int, head_dim: int, n_heads: int, semantic_hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.output_dim = head_dim * n_heads
        self.FCL = nn.Linear(input_dim, fcl_dim)
        self.GPL = SecGNNGraphPartitionLayer()
        self.GAL = SecGNNGraphAttentionLayer(fcl_dim, head_dim, n_heads, dropout=dropout)
        self.SAL = SecGNNSemanticAttentionLayer(self.output_dim, semantic_hidden_dim)
        self.activation = nn.LeakyReLU(0.2)

        nn.init.xavier_uniform_(self.FCL.weight)
        nn.init.zeros_(self.FCL.bias)

    def forward(self, X: torch.Tensor, K: int, J: int, return_aux: bool = False):
        X = self.activation(self.FCL(X))
        masks = self.GPL(K, J, X.device)

        if return_aux:
            relation_features, attention = self.GAL(X, masks, return_attention=True)
            X, semantic_weight = self.SAL(relation_features, return_weight=True)
            X = self.activation(X)
            return X, {"attention": attention, "semantic_weight": semantic_weight, "graph_masks": masks}

        X = self.GAL(X, masks, return_attention=False)
        X = self.SAL(X, return_weight=False)
        return self.activation(X)

class SecGNN_matched_128(nn.Module):
    """Parameter-matched beamforming-only SecGNN baseline."""
    def __init__(self, input_dim, hidden_layers, output_dim, n_heads, NT: int, Pmax: float = 1.0, power_mode: str = "gformer", dropout: float = 0.1):
        super().__init__()

        in_ch = int(input_dim[0] if isinstance(input_dim, (list, tuple)) else input_dim)
        out_ch = int(output_dim[-1] if isinstance(output_dim, (list, tuple)) else output_dim)
        if in_ch != 2 or out_ch != 2:
            raise ValueError("This implementation requires input_dim=[2], output_dim=[2]")
        if NT <= 0:
            raise ValueError("NT must be positive")
        if power_mode not in ("gformer", "paper"):
            raise ValueError("power_mode must be 'gformer' or 'paper'")

        self.NT = int(NT)
        self.Pmax = float(Pmax)
        self.power_mode = power_mode

        self.fcl_dim = 112
        self.little_n_heads = 8
        self.head_dim = 20
        self.graph_dim = self.little_n_heads * self.head_dim
        self.semantic_hidden_dim = 128
        self.output_hidden_dim = 160

        self.GM1 = SecGNNGraphModule(
            input_dim=2 * self.NT,
            fcl_dim=self.fcl_dim,
            head_dim=self.head_dim,
            n_heads=self.little_n_heads,
            semantic_hidden_dim=self.semantic_hidden_dim,
            dropout=dropout,
        )
        self.GM2 = SecGNNGraphModule(
            input_dim=self.graph_dim,
            fcl_dim=self.fcl_dim,
            head_dim=self.head_dim,
            n_heads=self.little_n_heads,
            semantic_hidden_dim=self.semantic_hidden_dim,
            dropout=dropout,
        )
        self.output_fcn1 = nn.Linear(self.graph_dim, self.output_hidden_dim)
        self.output_fcn2 = nn.Linear(self.output_hidden_dim, 2 * self.NT)

        self.output_activation = nn.LeakyReLU(0.2)

        nn.init.xavier_uniform_(self.output_fcn1.weight)
        nn.init.zeros_(self.output_fcn1.bias)
        nn.init.xavier_uniform_(self.output_fcn2.weight)
        nn.init.zeros_(self.output_fcn2.bias)

    def _channel_to_node_feature(self, H: torch.Tensor) -> torch.Tensor:
        if H.ndim != 4 or H.shape[1] != 2 or H.shape[-1] != self.NT:
            raise ValueError(f"H expected [B,2,M,{self.NT}], got {tuple(H.shape)}")
        return torch.cat([H[:, 0], H[:, 1]], dim=-1)

    def _power_process(self, V: torch.Tensor) -> torch.Tensor:
        power = torch.sum(V**2, dim=(1,2,3), keepdim=True)
        if self.power_mode == "gformer":
            return math.sqrt(self.Pmax) * V / torch.sqrt(power + 1e-12)

        scale = torch.sqrt(self.Pmax / torch.clamp(power, min=self.Pmax))
        return V * scale

    def forward(self, H_LU: torch.Tensor, H_Eve: torch.Tensor, return_aux: bool = False):
        if H_LU.ndim != 4 or H_LU.shape[1] != 2:
            raise ValueError(f"H_LU expected [B,2,K,NT], got {tuple(H_LU.shape)}")
        if H_Eve.ndim != 4 or H_Eve.shape[1] != 2:
            raise ValueError(f"H_Eve expected [B,2,J,NT], got {tuple(H_Eve.shape)}")

        B, _, K, NT = H_LU.shape
        B2, _, J, NT2 = H_Eve.shape
        if B2 != B or NT != self.NT or NT2 != self.NT:
            raise ValueError("H_LU / H_Eve batch size or NT mismatch")

        X_LU = self._channel_to_node_feature(H_LU)
        X_Eve = self._channel_to_node_feature(H_Eve)
        X = torch.cat([X_LU, X_Eve], dim=1)

        if return_aux:
            X, aux1 = self.GM1(X, K, J, return_aux=True)
            X, aux2 = self.GM2(X, K, J, return_aux=True)
        else:
            X = self.GM1(X, K, J, return_aux=False)
            X = self.GM2(X, K, J, return_aux=False)

        X_LU = X[:, :K]
        W = self.output_fcn2(self.output_activation(self.output_fcn1(X_LU)))
        W = W.reshape(B, K, 2, self.NT)
        V = W.permute(0, 2, 3, 1).contiguous()
        V = self._power_process(V)

        if return_aux:
            return V, {"GM1": aux1, "GM2": aux2, "node_feature": X, "K": K, "J": J}
        return V

@torch.no_grad()
def test_secgnn_basic(model: nn.Module, K=4, J=4, NT=2, batch_size=2):
    model.eval()
    device = next(model.parameters()).device

    H_LU = torch.randn(batch_size, 2, K, NT, device=device)
    H_Eve = torch.randn(batch_size, 2, J, NT, device=device)
    V = model(H_LU, H_Eve)

    shape_pass = V.shape == (batch_size, 2, NT, K)
    finite_pass = torch.isfinite(V).all().item()
    power = torch.sum(V**2, dim=(1,2,3))

    perm_K = torch.randperm(K, device=device)
    V_user = model(H_LU[:, :, perm_K], H_Eve)
    user_diff = torch.max(torch.abs(V_user - V[:, :, :, perm_K])).item()
    user_pass = user_diff < 1e-5

    perm_J = torch.randperm(J, device=device)
    V_eve = model(H_LU, H_Eve[:, :, perm_J])
    eve_diff = torch.max(torch.abs(V_eve - V)).item()
    eve_pass = eve_diff < 1e-5

    perm_NT = torch.randperm(NT, device=device)
    V_ant = model(H_LU[:, :, :, perm_NT], H_Eve[:, :, :, perm_NT])
    ant_diff = torch.max(torch.abs(V_ant - V[:, :, perm_NT])).item()

    structural_pass = shape_pass and finite_pass and user_pass and eve_pass

    print("=" * 72)
    print("SecGNN BF-only revised test")
    print("=" * 72)
    print("V shape:", tuple(V.shape))
    print("power:", power.cpu())
    print(f"user permutation equivariance: {user_diff:.3e} {'PASS' if user_pass else 'FAIL'}")
    print(f"Eve permutation invariance    : {eve_diff:.3e} {'PASS' if eve_pass else 'FAIL'}")
    print(f"antenna permutation diagnostic: {ant_diff:.3e} (not required to pass)")
    print(f"structural tests: {'PASS' if structural_pass else 'FAIL'}")
    print("=" * 72)

    return {"power": power.cpu(), "user_perm_diff": user_diff, "eve_perm_diff": eve_diff, "ant_perm_diff": ant_diff, "all_pass": structural_pass}

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    K, J, NT = 4, 4, 8

    model = SecGNN_matched_128(
        input_dim=[2],
        hidden_layers=[128,128,128],
        output_dim=[2],
        n_heads=1,
        NT=NT,
        Pmax=1.0,
        power_mode="gformer",
        dropout=0.1,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    target_params = 305_744 if NT == 8 else None

    print("=" * 72)
    print("Matched SecGNN-128 parameter report")
    print("=" * 72)
    print(f"trainable parameters: {n_params:,} ({n_params / 1e6:.6f} M)")
    if target_params is not None:
        diff = n_params - target_params
        print(f"target parameters   : {target_params:,}")
        print(f"difference          : {diff:+,}")
        assert n_params == target_params, (
            f"Parameter mismatch: got {n_params:,}, expected {target_params:,}"
        )
    print("=" * 72)

    test_secgnn_basic(model, K=K, J=J, NT=NT, batch_size=2)
