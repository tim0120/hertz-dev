#!/usr/bin/env python3
"""
1-GPU smoke test for the DDP training stack.

Checks:
  1. Model loads with pretrained checkpoints (is_split=False).
  2. Frozen modules are frozen; top model is trainable.
  3. Plex ablation produces identical shape/dtype with different values.
  4. One R2 batch fetch succeeds.
  5. One forward + backward pass on real data.
  6. One validation pass on a tiny test-split slice.
  7. Prints measured step/s, tok/s, and first-batch R2 throughput.

Usage:
  RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29501 \\
  python3 smoke_test.py --manifest_prefix <prefix> --max_manifests 2
"""

from __future__ import annotations

import os
import time
import argparse

import torch as T
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import get_hertz_dev_config, HertzDevModel
from utils.dist import init_dist, print0, rank0
from r2_dataset import R2DatasetConfig, R2ManifestDataset, collate_waveforms, list_manifests
from ablation import apply_plex_mode
from metrics import (
    token_topk_accuracy, token_entropy, code_utilization, summarize_val,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", default=os.environ.get("R2_BUCKET", "segmentation-datasets"))
    p.add_argument("--manifest_prefix", required=True)
    p.add_argument("--max_manifests", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--train_steps", type=int, default=5)
    p.add_argument("--val_batches", type=int, default=4)
    return p.parse_args()


def _train_forward(core: HertzDevModel, lat: T.Tensor) -> T.Tensor:
    x = core.input(lat)
    for layer in core.layers:
        x = layer(x)
    return core.output(x)


@T.no_grad()
def _tokenize(core: HertzDevModel, wav: T.Tensor, device):
    wav = wav.to(device=device, dtype=T.bfloat16, non_blocking=True).unsqueeze(1)
    lat = core.audio_tokenizer.latent_from_data(wav)
    _, tok = core.resynthesizer.quantizer(lat, return_latent=True)
    return lat, tok.long()


def main():
    args = parse_args()
    init_dist()
    device = "cuda:0"
    T.cuda.set_device(device)

    print0("=== [smoke] loading model ===")
    t0 = time.time()
    cfg = get_hertz_dev_config(is_split=False)
    model: HertzDevModel = cfg()
    model = model.to(device).bfloat16()
    print0(f"[smoke] model loaded in {time.time() - t0:.1f}s")

    # ---- freeze non-top ----
    for p in model.audio_tokenizer.parameters():
        p.requires_grad = False
    for p in model.resynthesizer.parameters():
        p.requires_grad = False
    model.audio_tokenizer.train(False)
    model.resynthesizer.train(False)

    # ---- frozen/trainable audit ----
    def count(module):
        return (sum(p.numel() for p in module.parameters() if p.requires_grad),
                sum(p.numel() for p in module.parameters()))
    tok_t, tok_a = count(model.audio_tokenizer)
    res_t, res_a = count(model.resynthesizer)
    all_t = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_a = sum(p.numel() for p in model.parameters())
    print0(f"[smoke] audio_tokenizer trainable={tok_t:,} / {tok_a:,}  (expect 0 / >0)")
    print0(f"[smoke] resynthesizer    trainable={res_t:,} / {res_a:,}  (expect 0 / >0)")
    print0(f"[smoke] top model        trainable={all_t - tok_t - res_t:,}")
    print0(f"[smoke] total            trainable={all_t:,} / {all_a:,}")
    assert tok_t == 0 and res_t == 0, "frozen modules must have 0 trainable params"
    assert all_t > 0, "top model must be trainable"

    # ---- ablation parity on the resynthesizer ----
    # In non-split mode the resynthesizer returns a 3-tuple (locs, scales, weights)
    # from the GMM output head. Shapes must match between modes; values must differ.
    print0("\n=== [smoke] ablation parity ===")
    wav_rand = T.randn(1, 16000 * 4, device=device, dtype=T.bfloat16).unsqueeze(1)
    with T.no_grad():
        lat_real = model.audio_tokenizer.latent_from_data(wav_rand)

    def _as_tuple(x):
        return x if isinstance(x, tuple) else (x,)

    apply_plex_mode(model, disable=False)
    with T.no_grad():
        out_normal = _as_tuple(model.resynthesizer(lat_real))
    apply_plex_mode(model, disable=True)
    with T.no_grad():
        out_skip = _as_tuple(model.resynthesizer(lat_real))

    assert len(out_normal) == len(out_skip), "output tuple arity differs between modes"
    max_diff = 0.0
    for i, (a, b) in enumerate(zip(out_normal, out_skip)):
        assert a.shape == b.shape and a.dtype == b.dtype, (
            f"element {i} differs in shape/dtype: {a.shape}/{a.dtype} vs {b.shape}/{b.dtype}")
        d = (a.float() - b.float()).abs().mean().item()
        print0(f"[smoke] elem {i}: shape={tuple(a.shape)} dtype={a.dtype}  mean |Δ|={d:.4e}")
        max_diff = max(max_diff, d)
    assert max_diff > 0, "plex ablation produced bit-identical output — ablation didn't take effect"
    # Leave plex disabled for the remaining smoke checks.
    apply_plex_mode(model, disable=True)

    # ---- dataset ----
    print0("\n=== [smoke] R2 dataset ===")
    keys = list_manifests(args.bucket, args.manifest_prefix)
    keys = keys[: args.max_manifests]
    print0(f"[smoke] using {len(keys)} manifests")

    train_ds = R2ManifestDataset(
        R2DatasetConfig(args.bucket, args.manifest_prefix, split="train"),
        rank=0, world_size=1, manifest_keys=keys, seed=0,
    )
    val_ds = R2ManifestDataset(
        R2DatasetConfig(args.bucket, args.manifest_prefix, split="test"),
        rank=0, world_size=1, manifest_keys=keys, seed=0,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=0,
                              collate_fn=collate_waveforms, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0,
                            collate_fn=collate_waveforms, drop_last=True)

    # First-batch fetch timing + R2 throughput
    t_fetch = time.time()
    batch = next(iter(train_loader))
    fetch_time = time.time() - t_fetch
    bytes_fetched = batch["wav"].numel() * 2  # int16 original
    print0(f"[smoke] first batch: shape={tuple(batch['wav'].shape)} "
           f"fetch={fetch_time:.2f}s ({bytes_fetched / fetch_time / 1024**2:.1f} MiB/s incl. manifest)")

    # ---- training step(s) ----
    print0(f"\n=== [smoke] {args.train_steps} training steps ===")
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = T.optim.AdamW(trainable, lr=3e-5, weight_decay=0.1, betas=(0.9, 0.95))
    model.train()
    vocab_size = model.c.vocab_size

    tokens_seen = 0
    t_start = time.time()
    data_iter = iter(train_loader)
    for i in range(args.train_steps):
        try:
            b = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            b = next(data_iter)
        lat, tok = _tokenize(model, b["wav"], device)
        with T.autocast(device_type="cuda", dtype=T.bfloat16):
            logits = _train_forward(model, lat)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, vocab_size).float(),
                tok[:, 1:].reshape(-1),
            )
        loss.backward()
        # Check gradient flow
        if i == 0:
            g_train = sum((p.grad.float().norm() ** 2).item()
                          for p in trainable if p.grad is not None) ** 0.5
            g_frozen = 0.0
            for p in model.audio_tokenizer.parameters():
                if p.grad is not None:
                    g_frozen += (p.grad.float().norm() ** 2).item()
            g_frozen = g_frozen ** 0.5
            print0(f"[smoke] first step grad norms: trainable={g_train:.4f} "
                   f"frozen={g_frozen:.4f}  (frozen must be 0)")
            assert g_frozen == 0.0, "frozen parameters have gradients!"
            assert g_train > 0, "trainable parameters have no gradients"
        T.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        tokens_seen += tok[:, 1:].numel()
        print0(f"[smoke] step {i + 1}: loss={loss.item():.4f}")
    t_total = time.time() - t_start
    print0(f"[smoke] trained {args.train_steps} steps in {t_total:.1f}s "
           f"({args.train_steps / t_total:.2f} step/s, "
           f"{tokens_seen / t_total:.0f} tok/s)")
    print0(f"[smoke] peak memory: {T.cuda.max_memory_allocated(device) / 1024**3:.1f} GiB")

    # ---- one validation pass ----
    print0(f"\n=== [smoke] validation pass ({args.val_batches} batches) ===")
    model.train(False)
    loss_sum = T.zeros((), dtype=T.float64, device=device)
    loss_count = T.zeros((), dtype=T.float64, device=device)
    top1 = T.zeros(2, dtype=T.float64, device=device)
    top5 = T.zeros(2, dtype=T.float64, device=device)
    ent = T.zeros(2, dtype=T.float64, device=device)
    hist = T.zeros(vocab_size, dtype=T.float64, device=device)
    with T.no_grad():
        vit = iter(val_loader)
        for i in range(args.val_batches):
            try:
                b = next(vit)
            except StopIteration:
                break
            lat, tok = _tokenize(model, b["wav"], device)
            with T.autocast(device_type="cuda", dtype=T.bfloat16):
                logits = _train_forward(model, lat)
            pred = logits[:, :-1].reshape(-1, vocab_size).float()
            tgt = tok[:, 1:].reshape(-1)
            loss_sum += F.cross_entropy(pred, tgt, reduction="sum").to(T.float64)
            loss_count += float(tgt.numel())
            tk = token_topk_accuracy(pred, tgt, ks=(1, 5))
            top1 += tk[1]; top5 += tk[5]
            ent += token_entropy(pred)
            hist += code_utilization(pred.argmax(dim=-1), vocab_size)
    metrics = summarize_val(loss_sum, loss_count, top1, top5, ent, hist, vocab_size)
    print0(f"[smoke] val: loss={metrics['loss']:.4f} ppl={metrics['perplexity']:.1f} "
           f"top1={metrics['top1_acc']:.4f} top5={metrics['top5_acc']:.4f} "
           f"code_util={metrics['code_util']:.4f} ent={metrics['pred_entropy']:.3f}")

    print0("\n[smoke] ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
