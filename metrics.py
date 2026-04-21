"""
Metrics for FSQ token-level next-token prediction.

All helpers are rank-local. `ddp_sum_reduce` aggregates across ranks when
DDP is initialized.
"""

from __future__ import annotations

import math
from typing import Dict

import torch as T
import torch.nn.functional as F


def ddp_sum_reduce(x: T.Tensor) -> T.Tensor:
    if T.distributed.is_available() and T.distributed.is_initialized():
        T.distributed.all_reduce(x, op=T.distributed.ReduceOp.SUM)
    return x


@T.no_grad()
def token_topk_accuracy(logits: T.Tensor, targets: T.Tensor, ks=(1, 5)) -> Dict[int, T.Tensor]:
    """
    logits:  (N, V) float
    targets: (N,)  long
    Returns a dict mapping k -> scalar tensor with (#correct, #total) count tuple
    represented as two-element tensor for easy DDP summation.
    """
    out: Dict[int, T.Tensor] = {}
    maxk = max(ks)
    _, pred = logits.topk(maxk, dim=-1, largest=True, sorted=True)   # (N, maxk)
    correct = pred.eq(targets.unsqueeze(-1))                          # (N, maxk)
    for k in ks:
        n_correct = correct[:, :k].any(dim=-1).sum()
        out[k] = T.stack([n_correct.to(T.float64),
                          T.tensor(targets.numel(), device=logits.device, dtype=T.float64)])
    return out


@T.no_grad()
def token_entropy(logits: T.Tensor) -> T.Tensor:
    """Mean Shannon entropy (nats) of softmax(logits) over the vocab dim.

    Returns a two-element tensor [sum_entropy, count] for DDP summation.
    """
    logp = F.log_softmax(logits, dim=-1)
    p = logp.exp()
    h = -(p * logp).sum(dim=-1)  # (N,)
    return T.stack([h.sum().to(T.float64),
                    T.tensor(h.numel(), device=logits.device, dtype=T.float64)])


@T.no_grad()
def code_utilization(argmax_ids: T.Tensor, vocab_size: int) -> T.Tensor:
    """
    argmax_ids: (N,) long — the argmax token id at each position.
    Returns a one-hot-like histogram tensor of shape (vocab_size,) with counts
    per id. Caller can sum across DDP ranks and compute utilization as
    (hist > 0).sum() / vocab_size.
    """
    hist = T.zeros(vocab_size, dtype=T.float64, device=argmax_ids.device)
    hist.scatter_add_(0, argmax_ids.long(),
                      T.ones_like(argmax_ids, dtype=T.float64))
    return hist


def summarize_val(
    loss_sum: T.Tensor, loss_count: T.Tensor,
    top1: T.Tensor, top5: T.Tensor,
    entropy: T.Tensor,
    code_hist: T.Tensor,
    vocab_size: int,
) -> Dict[str, float]:
    """Given DDP-summed accumulators, produce a dict of scalar metrics."""
    loss = (loss_sum / loss_count.clamp_min(1.0)).item()
    top1_acc = (top1[0] / top1[1].clamp_min(1.0)).item()
    top5_acc = (top5[0] / top5[1].clamp_min(1.0)).item()
    pred_entropy = (entropy[0] / entropy[1].clamp_min(1.0)).item()
    code_util = ((code_hist > 0).sum().item()) / float(vocab_size)
    return {
        "loss": loss,
        "perplexity": math.exp(min(loss, 20.0)),  # cap to avoid overflow pre-warmup
        "top1_acc": top1_acc,
        "top5_acc": top5_acc,
        "pred_entropy": pred_entropy,
        "code_util": code_util,
    }
