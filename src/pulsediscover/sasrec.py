"""
PulseDiscover — SASRec (Phase 3A), small CPU-safe build.

Self-attentive sequential recommender (Kang & McAuley 2018 style):
  item + positional embeddings -> stacked CAUSAL self-attention blocks -> next-item logits.
Trained with full-softmax next-item cross-entropy (n_items ~3.7k on MovieLens-1M, cheap).
Causal mask = left-to-right (no future leakage), matching the serving objective.

CPU-safe defaults: d=48, 2 blocks, 1 head, maxlen=30. Checkpoint/resume friendly.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SASBlock(nn.Module):
    def __init__(self, d, nheads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, nheads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d), nn.Dropout(dropout))

    def forward(self, x, attn_mask, key_padding_mask):
        a, _ = self.attn(x, x, x, attn_mask=attn_mask,
                         key_padding_mask=key_padding_mask, need_weights=False)
        x = self.ln1(x + a)
        x = self.ln2(x + self.ff(x))
        return x


class SASRec(nn.Module):
    def __init__(self, n_items, d=48, maxlen=30, nblocks=2, nheads=1, dropout=0.2):
        super().__init__()
        self.maxlen = maxlen
        self.item_emb = nn.Embedding(n_items + 1, d, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen, d)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([SASBlock(d, nheads, dropout) for _ in range(nblocks)])
        self.ln = nn.LayerNorm(d)

    def seq_repr(self, seq):
        B, L = seq.shape
        pos = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)
        x = self.drop(self.item_emb(seq) + self.pos_emb(pos))
        causal = torch.triu(torch.ones(L, L, device=seq.device), diagonal=1).bool()
        pad = (seq == 0)
        for blk in self.blocks:
            x = blk(x, causal, pad)
        return self.ln(x)

    def forward(self, seq):
        h = self.seq_repr(seq)                      # (B,L,d)
        return h @ self.item_emb.weight.T           # (B,L,V)

    def forward_last(self, seq):
        """Logits at the final position only (B,V) — WITH grad. CPU-cheap: avoids the
        (B,L,V) matmul. Trains next-item-given-full-history (the served objective)."""
        h = self.seq_repr(seq)[:, -1, :]            # (B,d)
        return h @ self.item_emb.weight.T           # (B,V)

    @torch.no_grad()
    def score_last(self, seq):
        h = self.seq_repr(seq)[:, -1, :]            # (B,d) last position
        return h @ self.item_emb.weight.T           # (B,V)


def build_user_sequences(train_df, item2idx):
    """Return {user_id: [item_idx in chronological order]} from the (already time-sorted) train df."""
    seqs = {}
    for u, it in zip(train_df["user_id"].to_numpy(), train_df["item_id"].to_numpy()):
        seqs.setdefault(u, []).append(item2idx[it])
    return seqs


def _left_pad(seq_idx, maxlen):
    seq = seq_idx[-maxlen:]
    return [0] * (maxlen - len(seq)) + seq


def make_training_tensors(user_seqs, maxlen):
    """input = items[:-1], target = items[1:], left-padded. Skips users with <2 items."""
    X, Y = [], []
    for u, items in user_seqs.items():
        if len(items) < 2:
            continue
        inp, tgt = items[:-1], items[1:]
        X.append(_left_pad(inp, maxlen)); Y.append(_left_pad(tgt, maxlen))
    return torch.tensor(X, dtype=torch.long), torch.tensor(Y, dtype=torch.long)


def train_one_epoch(model, X, Y, opt, batch=256, seed=0):
    """Next-item-at-sequence-end training (CPU-budget choice): predict items[-1] from the
    full left-padded history items[:-1]. Full per-position training is a documented future option."""
    model.train()
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(X.shape[0], generator=g)
    total, nb = 0.0, 0
    for i in range(0, X.shape[0], batch):
        idx = perm[i:i + batch]
        logits_last = model.forward_last(X[idx])     # (b,V)
        loss = F.cross_entropy(logits_last, Y[idx][:, -1], ignore_index=0)
        opt.zero_grad(); loss.backward(); opt.step()
        total += float(loss.detach()); nb += 1
    return total / max(1, nb)


@torch.no_grad()
def evaluate(model, user_seqs, test_pairs, item2idx, maxlen, KS=(5, 10, 20)):
    """test_pairs: list of (user_id, gold_item_id). Excludes the user's train history.
    Returns per-user {K: {'r':.., 'n':..}} lists for CI computation."""
    model.eval()
    out = {m: {k: [] for k in KS} for m in ["sasrec"]}
    for u, gold in test_pairs:
        if u not in user_seqs or gold not in item2idx:
            # cold gold / no-history: count as a miss so N matches the baselines exactly
            for k in KS:
                out["sasrec"][k].append({"r": 0.0, "n": 0.0})
            continue
        seq = torch.tensor([_left_pad(user_seqs[u], maxlen)], dtype=torch.long)
        scores = model.score_last(seq)[0].numpy()    # (V,)
        scores[0] = -1e9                              # never recommend padding
        for it in user_seqs[u]:                       # exclude seen
            scores[it] = -1e9
        gi = item2idx[gold]
        order = np.argpartition(-scores, max(KS))[:max(KS)]
        order = order[np.argsort(-scores[order])]
        ranked = list(order)
        for k in KS:
            hit = gi in ranked[:k]
            out["sasrec"][k].append({"r": 1.0 if hit else 0.0,
                                      "n": (1.0 / np.log2(ranked.index(gi) + 2)) if hit else 0.0})
    return out
