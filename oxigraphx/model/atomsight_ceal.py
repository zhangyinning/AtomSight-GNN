import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.nn import BatchNorm, global_add_pool, global_mean_pool
from tqdm import tqdm
import os.path as osp
from sklearn.metrics import r2_score
import numpy as np
from sklearn.metrics import pairwise_distances
from typing import List, Optional

from .ceal import *
from utils import *
from config import *


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




class MyCEALNetwork(torch.nn.Module):
   """
   in_channels,
   out_channels,
   aggregators,

   scalers,
   deg,
   edge_dim = None,
   towers = 1,
   pre_layers = 1,
   post_layers = 1,
   divide_input = False,
   
   """   
   def __init__(self, 
                in_channels, 
                out_channels, 
                aggregators,
                scalers,
                deg,
                numLayers = 2,
                edge_dim = None,
                towers = 1,
                pre_layers = 1,
                post_layers = 1,
                divide_input = False
               ):

      super().__init__()

      self.out_channels = out_channels
      self.numLayers = numLayers
      
      self.attention = GlobalFeatureAttention(
            n_features = in_channels
        ) 

        # The first CEAL layer
      self.ceal_1 = CEALConv( in_channels = in_channels,
                                out_channels = out_channels,
                                aggregators = aggregators,
                                scalers = scalers,
                                deg = deg,
                                edge_dim = edge_dim,
                                towers = 1,
                                pre_layers = 1,
                                post_layers = 1,
                                divide_input = False
                            )
      self.batch_norm_1 = BatchNorm(self.out_channels)

      # This container holds 2nd and more ceal layers
      self.cealConvs = ModuleList()
      self.batch_norms = ModuleList()
      for _ in range(1, self.numLayers):
        cealconv = CEALConv( in_channels = out_channels,
                                out_channels = out_channels,
                                aggregators = aggregators,
                                scalers = scalers,
                                deg = deg,
                                edge_dim = edge_dim,
                                towers = 1,
                                pre_layers = 1,
                                post_layers = 1,
                                divide_input = False
                            )
        batch_norm = BatchNorm(self.out_channels)
        self.cealConvs.append(cealconv)
        self.batch_norms.append(batch_norm)

      self.pre_mlp = Sequential(Linear(in_channels, out_channels//2), ReLU(), Linear(out_channels//2, out_channels))
      self.post_mlp = Sequential(Linear(self.out_channels, 40), ReLU(), Linear(40, 1))
   
   
   def forward(self, batch_data):
      # Note: batch_data is provided by Dataloader
      x, edge_index, edge_attr = batch_data.x, batch_data.edge_index, batch_data.edge_attr
      
      x = self.attention(x)

      # x = self.pre_mlp(x)
      x = self.ceal_1(x, edge_index, batch_data.batch, edge_attr)
      x = self.batch_norm_1(x)
      x = F.relu(x)
      # x = F.dropout(x, p=0.2, training=self.training)

      for cealconv, batch_norm in zip(self.cealConvs, self.batch_norms):
          x = batch_norm(cealconv(x, edge_index, batch_data.batch, edge_attr))
          x = F.relu(x)
          # x = F.dropout(x, p=0.1,  training=self.training)
      
      x = global_mean_pool(x, batch_data.batch)
      out = self.post_mlp(x)
      return out


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




















