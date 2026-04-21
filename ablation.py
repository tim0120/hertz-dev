"""
Plex-ablation helper. The only knob is:

    apply_plex_mode(model, disable=True)

which sets `model.resynthesizer._disable_plex = True`. The runtime check in
`TransformerVAE.forward` (model.py) reads this attribute via
`getattr(self, "_disable_plex", False)` and skips the plex addition at
layer `layers // 2`.

Caveat: the resynthesizer is frozen during top-model fine-tuning, so this
ablation does not alter the training CE loss directly. It is exercised
whenever the resynthesizer is called (audio generation / decode).
"""

from __future__ import annotations


def apply_plex_mode(model, disable: bool = False, verbose: bool = True) -> None:
    """Flag the resynthesizer to skip its plex addition in the forward pass.

    Safe to call multiple times and in either direction (on/off).
    """
    resynth = getattr(model, "resynthesizer", None)
    if resynth is None:
        raise AttributeError("model has no .resynthesizer; cannot apply plex mode")

    setattr(resynth, "_disable_plex", bool(disable))
    if verbose:
        state = "disabled" if disable else "enabled"
        print(f"[ablation] resynthesizer plex addition is now {state}")
