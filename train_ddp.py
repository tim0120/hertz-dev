#!/usr/bin/env python3
"""
DDP fine-tuning for the hertz-dev top model on R2 LibriLight target clips.

- Frozen: audio_tokenizer, resynthesizer (and its FSQ quantizer).
- Trainable: HertzDevModel.input, layers, output, shape_rotator.
- Non-split mono model (is_split=False).
- Optional plex ablation: --disable_plex skips the resynthesizer plex addition.
- Resumable: saves {model_trainable, optimizer, step, rng, val_history, args}
  every --save_every steps, with a `latest.pt` symlink.
- Periodic validation on the test split with loss, perplexity, top-1/5 acc,
  prediction entropy, and FSQ code utilization.

Launch:
  torchrun --standalone --nproc_per_node=8 train_ddp.py \\
      --manifest_prefix training_outputs/<run>/manifests/ \\
      [--disable_plex] --total_steps 1000 --save_dir /root/ckpts/<name>
"""

from __future__ import annotations

import os
import sys
import math
import time
import json
import random
import argparse
from pathlib import Path

import numpy as np
import torch as T
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from model import get_hertz_dev_config, HertzDevModel
from utils.dist import init_dist, print0, rank0
from r2_dataset import R2DatasetConfig, R2ManifestDataset, collate_waveforms, list_manifests
from ablation import apply_plex_mode
from metrics import (
    ddp_sum_reduce, token_topk_accuracy, token_entropy,
    code_utilization, summarize_val,
)


# ---------------------------------------------------------------------------
# Training forward (bypass the @no_grad on HertzDevModel.forward)
# ---------------------------------------------------------------------------

def train_forward(core: HertzDevModel, lat: T.Tensor) -> T.Tensor:
    """Non-split forward that preserves grads to the top transformer."""
    x = core.input(lat)
    for layer in core.layers:
        x = layer(x)
    return core.output(x)


# ---------------------------------------------------------------------------
# On-the-fly frozen tokenization
# ---------------------------------------------------------------------------

@T.no_grad()
def tokenize_waveforms(core: HertzDevModel, wav: T.Tensor, device: str):
    """
    wav: (B, samples) float32 on CPU.
    Returns:
      lat: (B, T, latent_size) bf16
      tok: (B, T) long  -- FSQ code indices
    """
    wav = wav.to(device=device, dtype=T.bfloat16, non_blocking=True).unsqueeze(1)
    lat = core.audio_tokenizer.latent_from_data(wav)
    _, tok = core.resynthesizer.quantizer(lat, return_latent=True)
    return lat, tok.long()


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def get_lr(step, total, warmup, max_lr, min_lr):
    if step < warmup:
        return max_lr * step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@T.no_grad()
def run_validation(core: HertzDevModel, loader, device, vocab_size,
                   max_batches: int, step: int) -> dict:
    """Single validation pass; returns aggregated metrics dict."""
    core.train(False)  # inference mode
    loss_sum = T.zeros((), dtype=T.float64, device=device)
    loss_count = T.zeros((), dtype=T.float64, device=device)
    top1 = T.zeros(2, dtype=T.float64, device=device)
    top5 = T.zeros(2, dtype=T.float64, device=device)
    ent = T.zeros(2, dtype=T.float64, device=device)
    code_hist = T.zeros(vocab_size, dtype=T.float64, device=device)

    it = iter(loader)
    t0 = time.time()
    seen = 0
    for i in range(max_batches):
        try:
            batch = next(it)
        except StopIteration:
            break
        lat, tok = tokenize_waveforms(core, batch["wav"], device)

        with T.autocast(device_type="cuda", dtype=T.bfloat16):
            logits = train_forward(core, lat)

        pred_logits = logits[:, :-1].reshape(-1, vocab_size).float()
        tgt = tok[:, 1:].reshape(-1)

        loss = F.cross_entropy(pred_logits, tgt, reduction="sum")
        loss_sum += loss.to(T.float64)
        loss_count += float(tgt.numel())

        tk = token_topk_accuracy(pred_logits, tgt, ks=(1, 5))
        top1 += tk[1]
        top5 += tk[5]

        ent += token_entropy(pred_logits)
        argmax_ids = pred_logits.argmax(dim=-1)
        code_hist += code_utilization(argmax_ids, vocab_size)
        seen += 1

    # DDP reduce
    ddp_sum_reduce(loss_sum)
    ddp_sum_reduce(loss_count)
    ddp_sum_reduce(top1)
    ddp_sum_reduce(top5)
    ddp_sum_reduce(ent)
    ddp_sum_reduce(code_hist)

    core.train(True)
    metrics = summarize_val(loss_sum, loss_count, top1, top5, ent, code_hist, vocab_size)
    metrics["step"] = step
    metrics["val_batches"] = seen
    metrics["val_seconds"] = time.time() - t0
    return metrics


# ---------------------------------------------------------------------------
# Checkpoint IO
# ---------------------------------------------------------------------------

def save_ckpt(core, optimizer, step, args, val_history, save_dir: str, final: bool = False):
    name = "final.pt" if final else f"step_{step:07d}.pt"
    path = os.path.join(save_dir, name)
    payload = {
        "__version__": 1,
        "__step__": step,
        "__args__": vars(args),
        "__rng__": {
            "torch_cpu": T.get_rng_state(),
            "torch_cuda_all": T.cuda.get_rng_state_all() if T.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
        "model_trainable": {
            n: p.data.detach().cpu()
            for n, p in core.named_parameters() if p.requires_grad
        },
        "optimizer": optimizer.state_dict(),
        "val_history": val_history,
    }
    T.save(payload, path)
    latest = os.path.join(save_dir, "latest.pt")
    try:
        if os.path.islink(latest) or os.path.exists(latest):
            os.remove(latest)
        os.symlink(os.path.basename(path), latest)
    except OSError:
        pass
    print(f"[ckpt] saved {path}")


def load_ckpt_into(core, optimizer, path: str, device: str) -> int:
    print(f"[ckpt] resuming from {path}")
    payload = T.load(path, map_location="cpu", weights_only=False)
    # Restore trainable params
    loaded = 0
    trainable = {n: p for n, p in core.named_parameters() if p.requires_grad}
    for n, t in payload.get("model_trainable", {}).items():
        if n in trainable:
            trainable[n].data.copy_(t.to(device=trainable[n].device, dtype=trainable[n].dtype))
            loaded += 1
    # Optimizer
    if "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    # RNG
    rng = payload.get("__rng__", {})
    if rng.get("torch_cpu") is not None:
        T.set_rng_state(rng["torch_cpu"])
    if rng.get("torch_cuda_all") is not None and T.cuda.is_available():
        T.cuda.set_rng_state_all(rng["torch_cuda_all"])
    if rng.get("numpy") is not None:
        np.random.set_state(rng["numpy"])
    if rng.get("python") is not None:
        random.setstate(rng["python"])
    step = int(payload.get("__step__", 0))
    print(f"[ckpt] restored {loaded} trainable tensors, step={step}")
    return step


# ---------------------------------------------------------------------------
# Freezing
# ---------------------------------------------------------------------------

def freeze_non_top(model: HertzDevModel):
    for p in model.audio_tokenizer.parameters():
        p.requires_grad = False
    for p in model.resynthesizer.parameters():
        p.requires_grad = False
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    return n_train, n_total


def enable_grad_ckpt(model: HertzDevModel):
    from torch.utils.checkpoint import checkpoint as ckpt_fn
    for layer in model.layers:
        orig = layer.forward

        def make(fn):
            def wrapped(x, kv=None):
                return ckpt_fn(fn, x, kv, use_reentrant=False)
            return wrapped
        layer.forward = make(orig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="hertz-dev plex-ablation fine-tune")

    # Data
    p.add_argument("--bucket", default=os.environ.get("R2_BUCKET", "segmentation-datasets"))
    p.add_argument("--manifest_prefix", required=True)
    p.add_argument("--max_manifests", type=int, default=0,
                   help="If >0, cap discovered manifest files")

    # Model / ablation
    p.add_argument("--disable_plex", action="store_true",
                   help="Skip the resynthesizer plex addition in its forward pass")

    # Training
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--total_steps", type=int, default=1000)
    p.add_argument("--warmup_steps", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--grad_ckpt", action="store_true")
    p.add_argument("--num_workers", type=int, default=2)

    # Validation
    p.add_argument("--val_every", type=int, default=50)
    p.add_argument("--val_batches", type=int, default=32)

    # Checkpointing
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--save_every", type=int, default=50)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--resume", type=str, default=None)

    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


def build_loader(manifest_keys, bucket, prefix, split, rank, world_size,
                 batch_size, num_workers, seed):
    cfg = R2DatasetConfig(
        bucket=bucket,
        manifest_prefix=prefix,
        split=split,
    )
    ds = R2ManifestDataset(cfg, rank=rank, world_size=world_size,
                           manifest_keys=manifest_keys, seed=seed)
    return ds, DataLoader(
        ds, batch_size=batch_size, num_workers=num_workers,
        collate_fn=collate_waveforms, pin_memory=True, drop_last=True,
    )


def main():
    args = parse_args()

    rank, local_rank, world_size = init_dist()
    device = f"cuda:{local_rank}"
    T.cuda.set_device(device)
    T.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    random.seed(args.seed + rank)

    if rank0():
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Manifests discovered on rank 0, broadcast.
    # ------------------------------------------------------------------
    if rank0():
        keys = list_manifests(args.bucket, args.manifest_prefix)
        if args.max_manifests > 0:
            keys = keys[: args.max_manifests]
        print0(f"[data] discovered {len(keys)} manifests")
    else:
        keys = None
    if T.distributed.is_initialized():
        obj = [keys]
        T.distributed.broadcast_object_list(obj, src=0)
        keys = obj[0]

    train_ds, train_loader = build_loader(
        keys, args.bucket, args.manifest_prefix, "train",
        rank, world_size, args.batch_size, args.num_workers, args.seed,
    )
    val_ds, val_loader = build_loader(
        keys, args.bucket, args.manifest_prefix, "test",
        rank, world_size, args.batch_size, max(1, args.num_workers // 2),
        args.seed + 10001,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    print0("[model] building hertz-dev (is_split=False)")
    model_cfg = get_hertz_dev_config(is_split=False)
    model: HertzDevModel = model_cfg()
    model = model.to(device).bfloat16()

    apply_plex_mode(model, disable=args.disable_plex, verbose=rank0())

    n_train, n_total = freeze_non_top(model)
    print0(f"[model] trainable: {n_train:,} / {n_total:,} "
           f"({100.0 * n_train / max(n_total, 1):.2f}%)")

    # Put frozen submodules in inference mode (safety — they have no BN/Dropout,
    # but be explicit).
    model.audio_tokenizer.train(False)
    model.resynthesizer.train(False)

    if args.grad_ckpt:
        enable_grad_ckpt(model)
        print0("[model] gradient checkpointing enabled")

    if world_size > 1:
        ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                        find_unused_parameters=True, broadcast_buffers=False)
    else:
        ddp_model = model
    core: HertzDevModel = ddp_model.module if isinstance(ddp_model, DDP) else ddp_model

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    trainable_params = [p for p in ddp_model.parameters() if p.requires_grad]
    optimizer = T.optim.AdamW(trainable_params, lr=args.lr,
                              weight_decay=args.weight_decay, betas=(0.9, 0.95))

    start_step = 0
    if args.resume and os.path.exists(args.resume):
        start_step = load_ckpt_into(core, optimizer, args.resume, device)

    # ------------------------------------------------------------------
    # Training loop with periodic validation + checkpointing
    # ------------------------------------------------------------------
    vocab_size = core.c.vocab_size
    val_history = []

    # Pre-training validation anchor (step == start_step, before any updates).
    if rank0():
        print0(f"[val] running anchor validation at step {start_step} ...")
    metrics = run_validation(core, val_loader, device, vocab_size,
                             max_batches=args.val_batches, step=start_step)
    if rank0():
        val_history.append(metrics)
        print0(f"[val] step={metrics['step']:6d} "
               f"loss={metrics['loss']:.4f} ppl={metrics['perplexity']:.1f} "
               f"top1={metrics['top1_acc']:.4f} top5={metrics['top5_acc']:.4f} "
               f"code_util={metrics['code_util']:.4f} "
               f"ent={metrics['pred_entropy']:.3f} "
               f"batches={metrics['val_batches']} secs={metrics['val_seconds']:.1f}")

    ddp_model.train()
    step = start_step
    t_last = time.time()
    tokens_last = 0
    data_iter = iter(train_loader)

    while step < args.total_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            train_ds.set_seed(train_ds.seed + 1)
            data_iter = iter(train_loader)
            batch = next(data_iter)

        lat, tok = tokenize_waveforms(core, batch["wav"], device)

        lr = get_lr(step, args.total_steps, args.warmup_steps, args.lr, args.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        with T.autocast(device_type="cuda", dtype=T.bfloat16):
            logits = train_forward(core, lat)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, vocab_size).float(),
                tok[:, 1:].reshape(-1),
            )
            loss_scaled = loss / args.grad_accum

        loss_scaled.backward()

        if ((step + 1) % args.grad_accum) == 0:
            if args.grad_clip > 0:
                T.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        step += 1
        tokens_last += tok[:, 1:].numel()

        if rank0() and (step % args.log_every == 0 or step == 1):
            dt = max(time.time() - t_last, 1e-6)
            tok_per_sec = tokens_last / dt
            gpu_mem = T.cuda.max_memory_allocated(device) / (1024 ** 3)
            print(f"[train] step {step:6d}/{args.total_steps} "
                  f"loss {loss.item():.4f} lr {lr:.2e} "
                  f"{tok_per_sec:.0f} tok/s mem {gpu_mem:.1f} GiB")
            t_last = time.time()
            tokens_last = 0

        if step % args.val_every == 0:
            metrics = run_validation(core, val_loader, device, vocab_size,
                                     max_batches=args.val_batches, step=step)
            if rank0():
                val_history.append(metrics)
                print0(f"[val] step={metrics['step']:6d} "
                       f"loss={metrics['loss']:.4f} ppl={metrics['perplexity']:.1f} "
                       f"top1={metrics['top1_acc']:.4f} top5={metrics['top5_acc']:.4f} "
                       f"code_util={metrics['code_util']:.4f} "
                       f"ent={metrics['pred_entropy']:.3f}")

        if rank0() and step % args.save_every == 0:
            save_ckpt(core, optimizer, step, args, val_history, args.save_dir)

    if rank0():
        save_ckpt(core, optimizer, step, args, val_history, args.save_dir, final=True)

    if T.distributed.is_initialized():
        T.distributed.barrier()
        T.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
