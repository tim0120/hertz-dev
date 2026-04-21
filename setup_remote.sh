#!/usr/bin/env bash
# Idempotent remote setup for hertz-dev DDP training.
# Expects to run inside /root/hertz-dev-train after an rsync push.
set -euo pipefail

echo "[setup] ensuring checkpoint + HF cache dirs"
mkdir -p /root/ckpts
mkdir -p /root/.cache/huggingface

echo "[setup] pip install required packages"
pip install --quiet boto3 einops huggingface_hub hf_transfer soundfile bitsandbytes >/dev/null

if [[ ! -f .env ]]; then
    echo "[setup] WARNING: .env not found — create it with R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET"
else
    echo "[setup] .env exists (contents not echoed)"
fi

python3 -c "import torch; print('[setup] torch', torch.__version__, 'cuda', torch.version.cuda, 'gpus', torch.cuda.device_count())"
python3 -c "import boto3; print('[setup] boto3', boto3.__version__)"
python3 -c "import torchaudio; print('[setup] torchaudio', torchaudio.__version__)"
echo "[setup] done"
