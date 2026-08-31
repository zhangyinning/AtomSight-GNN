from __future__ import print_function, division

"""
AtomSight-GNN: CGCNN + Global Feature-Dimension Attention
==========================================================
Implements the architecture defined in:
    "Depth Exam Proposal Spring 2026 v2"

Architecture summary (Section: Architecture Design)
----------------------------------------------------
AtomSight-GNN learns ONE global attention vector:

    w in R^F                  -- learnable parameter vector (logits)
    a = softmax(w) in R^F     -- normalized feature importance weights

The same a is broadcast and applied to EVERY atom in the crystal:

    x_tilde_i = a o x_i      -- element-wise reweighting (Eq. 5)

x_tilde_i then feeds into the original CGCNN message-passing pipeline unchanged.

Key property: a is GLOBAL -- it does not vary across atoms or structures.
This makes a a direct, interpretable estimate of feature importance that
requires no post-hoc aggregation.

Classes
-------
GlobalFeatureAttention   -- the attention module (Eqs. 2-5)
ConvLayer                -- original CGCNN conv layer (unchanged)
AtomSightGNN             -- full model
FeatureImportanceRanker  -- extracts and reports feature importance from a
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional


# =============================================================================
# 1.  Global Feature-Dimension Attention  (Eqs. 2-5 in proposal)
# =============================================================================

class GlobalFeatureAttention(nn.Module):
    """
    Global feature-dimension attention.

    Learns a SINGLE shared attention vector a in R^F applied identically
    to every atom in the crystal graph.  Attention weights do not vary
    per atom or per structure.

    Parameters
    ----------
    n_features : F -- number of atomic feature channels (orig_atom_fea_len)

    Attributes exposed for interpretability
    ----------------------------------------
    attention_weights()  ->  a = softmax(w)   shape [F]  normalized
    attention_logits()   ->  w                shape [F]  unnormalized
    """

    def __init__(self, n_features: int):
        super(GlobalFeatureAttention, self).__init__()
        self.n_features = n_features

        # w in R^F -- the sole learnable parameter of this module (Eq. 3)
        # Initialized to zeros -> softmax gives uniform 1/F weights at start,
        # so early training behaves identically to vanilla CGCNN.
        self.w = nn.Parameter(torch.zeros(n_features))

    def forward(self, atom_fea: torch.Tensor) -> torch.Tensor:
        """
        Apply global feature reweighting to all atoms.

        Parameters
        ----------
        atom_fea : [N, F]  -- raw atomic features for all atoms in the batch

        Returns
        -------
        atom_fea_weighted : [N, F]  -- reweighted features x_tilde_i = a o x_i
        """
        # a = softmax(w)  shape [F] -- same vector for every atom
        # a = torch.softmax(self.w, dim=0)   # [F]
        a = torch.softmax(self.w, dim=0) * self.n_features

        # Broadcast multiply: every row of atom_fea is scaled by a
        # [N, F] * [F] -> [N, F]
        return atom_fea * a                # x_tilde_i = a o x_i

    def attention_weights(self) -> torch.Tensor:
        """Return normalized attention vector a = softmax(w).  Shape [F]."""
        return torch.softmax(self.w, dim=0).detach().cpu()

    def attention_logits(self) -> torch.Tensor:
        """Return unnormalized logits w.  Shape [F]."""
        return self.w.detach().cpu()


# =============================================================================
# 2.  ConvLayer  (original CGCNN -- completely unchanged)
# =============================================================================

class ConvLayer(nn.Module):
    """
    Convolutional operation on graphs.
    Original CGCNN ConvLayer -- not modified.
    """
    def __init__(self, atom_fea_len, nbr_fea_len):
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len  = nbr_fea_len
        self.fc_full      = nn.Linear(
            2 * self.atom_fea_len + self.nbr_fea_len,
            2 * self.atom_fea_len
        )
        self.sigmoid   = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1       = nn.BatchNorm1d(2 * self.atom_fea_len)
        self.bn2       = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        N, M          = nbr_fea_idx.shape
        atom_nbr_fea  = atom_in_fea[nbr_fea_idx, :]
        total_nbr_fea = torch.cat(
            [atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
             atom_nbr_fea, nbr_fea], dim=2
        )
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(
            total_gated_fea.view(-1, self.atom_fea_len * 2)
        ).view(N, M, self.atom_fea_len * 2)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core   = self.softplus1(nbr_core)
        nbr_sumed  = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed  = self.bn2(nbr_sumed)
        out        = self.softplus2(atom_in_fea + nbr_sumed)
        return out


# =============================================================================
# 3.  AtomSight-GNN  (Eqs. 1-7 + CGCNN backbone)
# =============================================================================

class AtomSightGNN(nn.Module):
    """
    AtomSight-GNN.

    CGCNN augmented with a global feature-attention layer inserted before
    the embedding projection.  The attention vector a = softmax(w) is a
    direct, trainable representation of feature importance -- no post-hoc
    aggregation needed after training.

    Constructor parameters are identical to CrystalGraphConvNet, making
    this a drop-in replacement.

    Architecture flow
    -----------------
    raw x_i  ->  [GlobalFeatureAttention]  ->  x_tilde_i = a o x_i
             ->  [Linear embedding]        ->  h_i^(0)
             ->  [ConvLayer x n_conv]      ->  h_i^(L)
             ->  [Mean pooling]            ->  h_crystal
             ->  [FC head]                 ->  y_hat
    """

    def __init__(
        self,
        orig_atom_fea_len: int,
        nbr_fea_len:       int,
        atom_fea_len:      int  = 64,
        n_conv:            int  = 3,
        h_fea_len:         int  = 128,
        n_h:               int  = 1,
        classification:    bool = False,
    ):
        super(AtomSightGNN, self).__init__()
        self.classification = classification

        # -- Global feature-dimension attention (Eqs. 2-5) ------------------
        # One learnable vector w in R^F; shared across ALL atoms and crystals.
        # Applied to raw features BEFORE embedding so that a maps directly
        # onto interpretable atomic property dimensions.
        self.attention = GlobalFeatureAttention(
            n_features = orig_atom_fea_len,
        )

        # -- CGCNN backbone (unchanged) -------------------------------------
        # self.embedding           = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs               = nn.ModuleList([
            ConvLayer(atom_fea_len=orig_atom_fea_len, nbr_fea_len=nbr_fea_len)
            for _ in range(n_conv)
        ])
        self.conv_to_fc          = nn.Linear(orig_atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()

        if n_h > 1:
            self.fcs        = nn.ModuleList(
                [nn.Linear(h_fea_len, h_fea_len) for _ in range(n_h - 1)]
            )
            self.softpluses = nn.ModuleList(
                [nn.Softplus() for _ in range(n_h - 1)]
            )

        if self.classification:
            self.fc_out     = nn.Linear(h_fea_len, 2)
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout    = nn.Dropout()
        else:
            self.fc_out     = nn.Linear(h_fea_len, 1)

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        """
        Identical signature to CrystalGraphConvNet.forward().

        Parameters
        ----------
        atom_fea         : [N, F]
        nbr_fea          : [N, M, nbr_fea_len]
        nbr_fea_idx      : [N, M]  LongTensor
        crystal_atom_idx : list of N0 LongTensors

        Returns
        -------
        out : [N0, 1]  regression  |  [N0, 2]  classification log-probs
        """
        # -- Eq. 5: x_tilde_i = a o x_i  (global reweighting) --------------
        atom_fea = self.attention(atom_fea)         # [N, F]

        # -- CGCNN pipeline -------------------------------------------------
        # atom_fea = self.embedding(atom_fea)         # [N, atom_fea_len]
        for conv in self.convs:
            atom_fea = conv(atom_fea, nbr_fea, nbr_fea_idx)

        crys_fea = self.pooling(atom_fea, crystal_atom_idx)
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)

        if self.classification:
            crys_fea = self.dropout(crys_fea)
        if hasattr(self, "fcs") and hasattr(self, "softpluses"):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))

        out = self.fc_out(crys_fea)
        if self.classification:
            out = self.logsoftmax(out)
        return out

    def pooling(self, atom_fea, crystal_atom_idx):
        assert sum(len(idx) for idx in crystal_atom_idx) == atom_fea.shape[0]
        return torch.cat(
            [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
             for idx_map in crystal_atom_idx], dim=0
        )

    def get_attention_weights(self) -> torch.Tensor:
        """
        Return the trained global attention vector a = softmax(w).
        Shape: [F].  This IS the feature importance ranking -- no aggregation
        or additional inference pass needed after training.
        """
        return self.attention.attention_weights()

    def get_attention_logits(self) -> torch.Tensor:
        """
        Return unnormalized logits w.  Shape: [F].
        Report alongside softmax weights per the proposal to address
        concerns about softmax compressing absolute importance differences.
        """
        return self.attention.attention_logits()


# =============================================================================
# 4.  Feature Importance Ranker
# =============================================================================

class FeatureImportanceRanker:
    """
    Converts the trained global attention vector a = softmax(w) into a
    ranked feature-importance report.

    Because AtomSight-GNN uses a global attention mechanism, feature
    importance is read directly from the trained model parameters -- no
    inference pass or dataset aggregation required.

    Both the normalized weights a and the unnormalized logits w are
    reported, as recommended in the proposal.

    Usage
    -----
        ranker = FeatureImportanceRanker(
            model         = model,
            feature_names = FEATURE_NAMES,   # list of F strings
        )
        report = ranker.rank()
        ranker.plot(report, top_k=20, save_path="importance.png")
        ranker.save_report("importance.csv")
    """

    def __init__(
        self,
        model:         nn.Module,
        feature_names: Optional[List[str]] = None,
    ):
        self.model         = model
        self.feature_names = feature_names

    def _resolve_names(self, n_features: int) -> List[str]:
        if self.feature_names is not None:
            assert len(self.feature_names) == n_features, \
                f"feature_names length {len(self.feature_names)} != F {n_features}"
            return self.feature_names
        return [f"feat_{k}" for k in range(n_features)]

    def rank(self) -> "pd.DataFrame":
        """
        Build ranked importance DataFrame from the model's learned a.

        Columns
        -------
        rank          : 1 = most important feature
        feature       : feature name
        attn_weight   : a[f] = softmax(w)[f]   normalized importance
        attn_logit    : w[f]                    unnormalized logit
        pct_importance: attn_weight as a percentage
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pip install pandas")

        weights = self.model.get_attention_weights().numpy()
        logits  = self.model.get_attention_logits().numpy()
        names   = self._resolve_names(len(weights))

        df = pd.DataFrame({
            "feature"       : names,
            "attn_weight"   : weights,
            "attn_logit"    : logits,
            "pct_importance": weights * 100,
        })
        df = df.sort_values("attn_weight", ascending=False).reset_index(drop=True)
        df.insert(0, "rank", range(1, len(df) + 1))
        return df

    def plot(
        self,
        report:    "pd.DataFrame" = None,
        top_k:     int            = 20,
        title:     str            = "AtomSight-GNN Feature Importance",
        save_path: Optional[str]  = None,
    ):
        """
        Horizontal bar chart of top-k features.
        Bar length = normalized weight; annotation shows logit alongside.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("pip install matplotlib")

        if report is None:
            report = self.rank()

        top = report.head(top_k)
        y   = np.arange(len(top))

        fig, ax = plt.subplots(figsize=(10, max(4, top_k * 0.38)))
        ax.barh(
            y, top["attn_weight"][::-1].values,
            color="steelblue", edgecolor="white", linewidth=0.5,
        )
        ax.set_yticks(y)
        ax.set_yticklabels(top["feature"][::-1].values, fontsize=9)
        ax.set_xlabel("Attention Weight  a = softmax(w)", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

        for i, (w, logit) in enumerate(
            zip(top["attn_weight"][::-1].values,
                top["attn_logit"][::-1].values)
        ):
            ax.text(w + 2e-5, i,
                    f"weight={w:.4f}  logit={logit:.3f}",
                    va="center", ha="left", fontsize=7.5)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved to {save_path}")
        plt.show()
        return fig, ax

    def save_report(self, path: str):
        """Save ranked importance report to CSV."""
        report = self.rank()
        report.to_csv(path, index=False)
        print(f"Saved feature importance report -> {path}")