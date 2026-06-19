"""GRU4Rec (Hidasi et al. 2016 style) — session/sequence next-item recommender, CPU-safe.
Embedding -> GRU -> tied-weight next-item logits. Trained full-position (every step predicts the
next item) with sampled-softmax BCE. Same eval interface (score_last) as the SASRec module."""
from __future__ import annotations
import torch, torch.nn as nn


class GRU4Rec(nn.Module):
    def __init__(self, n_items, d=64, layers=1, dropout=0.2, maxlen=30):
        super().__init__()
        self.maxlen = maxlen
        self.item_emb = nn.Embedding(n_items + 1, d, padding_idx=0)
        self.drop = nn.Dropout(dropout)
        self.gru = nn.GRU(d, d, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)

    def seq_repr(self, seq):                 # (B,L) -> (B,L,d)
        x = self.drop(self.item_emb(seq))
        out, _ = self.gru(x)
        return out

    def forward_last(self, seq):             # (B,V) WITH grad at last position
        h = self.seq_repr(seq)[:, -1, :]
        return h @ self.item_emb.weight.T

    @torch.no_grad()
    def score_last(self, seq):               # (B,V) eval scoring at last position
        h = self.seq_repr(seq)[:, -1, :]
        return h @ self.item_emb.weight.T
