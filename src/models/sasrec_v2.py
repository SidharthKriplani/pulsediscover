"""SASRec v2 — same architecture as pulsediscover/sasrec.SASRec, but trained FULL-POSITION
(every step supervised, the canonical Kang & McAuley objective) instead of last-position only.
Full-position was the #1 diagnosed weakness of the D3B CPU run. CPU-feasible config here; a
GPU-ready canonical config is provided for an external run (not executed in this sandbox)."""
from pulsediscover.sasrec import SASRec  # reuse the attention stack (seq_repr gives all positions)

# CPU-feasible v2 config used in this sandbox (resumable, time-budgeted)
CPU_CONFIG = dict(d=64, maxlen=30, nblocks=2, nheads=2, dropout=0.2,
                  batch=512, nneg=100, lr=1e-3, training="full-position sampled-softmax BCE")

# GPU-ready canonical config (NOT run here — for an external canonical training run)
GPU_CANONICAL_CONFIG = dict(d=64, maxlen=50, nblocks=2, nheads=2, dropout=0.2,
                            batch=128, full_softmax=True, lr=1e-3, epochs=200,
                            note="run on GPU; full softmax over vocab; popularity-corrected negatives")

__all__ = ["SASRec", "CPU_CONFIG", "GPU_CANONICAL_CONFIG"]
