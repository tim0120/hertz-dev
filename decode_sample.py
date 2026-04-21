#!/usr/bin/env python3
"""
Decode a short audio sample from hertz-dev (stock or fine-tuned) for listen-back.

- Loads the mono (is_split=False) Hertz model with pretrained checkpoints.
- Optionally applies --disable_plex (plex ablation) in the resynthesizer.
- Optionally loads a fine-tuned checkpoint (only trainable-top weights).
- Seeds generation from a local prompt wav (default prompts/bob_mono.wav).
- Writes a .wav to the given output path.

Usage:
  python3 decode_sample.py --out /root/ckpts/sample_baseline.wav
  python3 decode_sample.py --disable_plex --load /root/ckpts/<run>/latest.pt \\
                           --out /root/ckpts/<run>/sample_skip.wav
"""

from __future__ import annotations

import os
import argparse

import torch as T
import torchaudio

from model import get_hertz_dev_config, HertzDevModel
from ablation import apply_plex_mode


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt_wav", type=str, default="prompts/bob_mono.wav",
                   help="Seed audio (mono, any sample rate; resampled to 16k).")
    p.add_argument("--prompt_seconds", type=float, default=6.0,
                   help="Length of seed audio to use for priming.")
    p.add_argument("--gen_seconds", type=float, default=6.0,
                   help="Length of generated continuation to decode.")
    p.add_argument("--disable_plex", action="store_true",
                   help="Skip the resynthesizer plex addition in its forward pass.")
    p.add_argument("--load", type=str, default=None,
                   help="Optional fine-tuned checkpoint path (trainable-top only).")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--temps_tok", type=float, default=0.8)
    p.add_argument("--temps_cat", type=float, default=0.5)
    p.add_argument("--temps_gauss", type=float, default=0.1)
    return p.parse_args()


def load_prompt(path: str, seconds: float, sr: int = 16000) -> T.Tensor:
    wav, orig_sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if orig_sr != sr:
        wav = torchaudio.functional.resample(wav, orig_sr, sr)
    n = int(seconds * sr)
    if wav.shape[-1] < n:
        pad = T.zeros(1, n - wav.shape[-1])
        wav = T.cat([wav, pad], dim=-1)
    else:
        wav = wav[:, :n]
    return wav  # (1, samples)


def main():
    args = parse_args()
    device = "cuda" if T.cuda.is_available() else "cpu"

    print("[decode] loading model (is_split=False)")
    cfg = get_hertz_dev_config(is_split=False)
    model: HertzDevModel = cfg()
    model = model.to(device).bfloat16()
    model.train(False)

    apply_plex_mode(model, disable=args.disable_plex)

    if args.load:
        print(f"[decode] loading fine-tuned trainable weights from {args.load}")
        payload = T.load(args.load, map_location="cpu", weights_only=False)
        param_map = dict(model.named_parameters())
        loaded = 0
        for n, t in payload.get("model_trainable", {}).items():
            if n in param_map:
                param_map[n].data.copy_(t.to(device=param_map[n].device,
                                             dtype=param_map[n].dtype))
                loaded += 1
        print(f"[decode] restored {loaded} trainable tensors")

    # Prompt → latents
    print(f"[decode] priming from {args.prompt_wav}")
    prompt = load_prompt(args.prompt_wav, args.prompt_seconds).to(device).bfloat16()
    # model.tokenize expects (B, C, T) with B=1, C=1 for mono
    prompt_in = prompt.unsqueeze(0)  # (1, 1, samples)

    with T.no_grad():
        latents_in = model.tokenize(prompt_in)  # (1, T_lat, latent_size)

    # Completion
    print(f"[decode] generating {args.gen_seconds}s continuation")
    gen_frames = int(args.gen_seconds * 8)  # 16kHz / 2000 stride = 8 frames/s
    with T.no_grad():
        generated_latents = model.completion(
            latents_in,
            temps=(args.temps_tok, (args.temps_cat, args.temps_gauss)),
            gen_len=gen_frames,
            use_cache=True,
        )
    # Decode latents to waveform
    with T.no_grad():
        audio = model.audio_tokenizer.data_from_latent(generated_latents)
    audio = audio.squeeze(0).float().cpu()  # (1, samples)
    # Normalize to avoid clipping
    peak = audio.abs().max().item()
    if peak > 0.99:
        audio = audio * (0.99 / peak)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    torchaudio.save(args.out, audio, 16000)
    print(f"[decode] wrote {args.out}  shape={tuple(audio.shape)}  peak={peak:.3f}")


if __name__ == "__main__":
    main()
