#!/bin/bash
# Rebuild the model and both drafters from their upstream sources, instead of downloading the
# finished files (download-models.sh). Produces byte-identical files to the published ones with
# the same llama.cpp build. Needs about 80 GB free: 56 GB of inputs plus 19 GB of outputs.
# The quantization takes about 8 minutes on an 8-core CPU; everything else is download time.
#
#   deploy/scripts/make-models.sh
#
# Environment:
#   MODELS     working and output directory     (default: ~/models)
#   LLAMA_DIR  llama.cpp fork checkout, built   (default: the one containing this script)
#   PYTHON     python with numpy and pyyaml     (default: ~/.venv-llama/bin/python)
#   HF_TOKEN   Hugging Face token (optional; the sources are public as of 2026-09-27)
set -euo pipefail
repo="$(cd "$(dirname "$0")/../.." && pwd)"

MODELS=${MODELS:-$HOME/models}
LLAMA_DIR=${LLAMA_DIR:-$repo}
PYTHON=${PYTHON:-$HOME/.venv-llama/bin/python}
auth=(); [ -n "${HF_TOKEN:-}" ] && auth=(-H "Authorization: Bearer $HF_TOKEN")
BIN=$LLAMA_DIR/build/bin
THREADS=$(nproc)

mkdir -p "$MODELS"
cd "$MODELS"

get() { # <repo> <path in repo> [local name]
  local out=${3:-$(basename "$2")}
  [ -f "$out" ] && { echo "have $out"; return; }
  echo "== $out"
  curl -fL -C - ${auth[@]+"${auth[@]}"} -o "$out.part" "https://huggingface.co/$1/resolve/main/$2"
  mv "$out.part" "$out"
}

# 1. inputs
for i in 1 2 3; do
  get ukisai/Swift-Qwen3.8-27B-GGUF Swift-Qwen3.8-27B-F16-0000$i-of-00003.gguf   # the fine-tune, F16
done
get unsloth/Qwen3.8-27B-GGUF imatrix_unsloth.gguf                              # Unsloth's imatrix (base Qwen3.8-27B)
get z-lab/Qwen3.8-27B-DFlash2-GGUF Qwen3.8-27B-DFlash2-BF16.gguf dflash-Qwen3.8-27B-BF16.gguf
get unsloth/Qwen3.8-27B-GGUF MTP/mtp-Qwen3.8-27B-Q4_0.gguf                     # Unsloth's MTP drafter

export ROCM_PATH=/usr HIP_PATH=/usr

# 2. the model: Unsloth's UD-Q4_K_XL per-tensor types with the IQ tensors as Q4_K, using
#    Unsloth's importance matrix for the base model. "did not find weights for output.weight"
#    is expected - the imatrix has no entry for it and it is kept at Q6_K.
if [ ! -f Swift-Qwen3.8-27B-Q4_K_XL-noIQ.gguf ]; then
  "$BIN/llama-quantize" --imatrix imatrix_unsloth.gguf \
    --tensor-type-file "$repo/deploy/recipes/swift-q4_k_xl.txt" \
    --token-embedding-type q4_k --output-tensor-type q6_k \
    Swift-Qwen3.8-27B-F16-00001-of-00003.gguf Swift-Qwen3.8-27B-Q4_K_XL-noIQ.gguf.part Q4_K_M "$THREADS"
  mv Swift-Qwen3.8-27B-Q4_K_XL-noIQ.gguf.part Swift-Qwen3.8-27B-Q4_K_XL-noIQ.gguf
fi

# 3. DFlash2 drafter at Q4_0 (the fastest of the drafter formats tested on gfx1030; the drafter's precision
#    does not change output, only acceptance)
[ -f dflash-Qwen3.8-27B-Q4_0.gguf ] ||
  "$BIN/llama-quantize" dflash-Qwen3.8-27B-BF16.gguf dflash-Qwen3.8-27B-Q4_0.gguf Q4_0 "$THREADS"

# 4. reduced draft vocabulary: each drafter's LM head becomes the 65536 most likely rows of the
#    model's own output projection plus a d2t map. Several times cheaper per step, same output.
VOCAB="$PYTHON $LLAMA_DIR/scripts/draft-vocab/build-draft-vocab.py"
[ -f dflash-Qwen3.8-27B-Q4_0-d2t64k-swiftxl.gguf ] ||
  $VOCAB dflash-Qwen3.8-27B-Q4_0.gguf Swift-Qwen3.8-27B-Q4_K_XL-noIQ.gguf dflash-Qwen3.8-27B-Q4_0-d2t64k-swiftxl.gguf
[ -f mtp-Qwen3.8-27B-d2t64k-swiftxl.gguf ] ||
  $VOCAB mtp-Qwen3.8-27B-Q4_0.gguf Swift-Qwen3.8-27B-Q4_K_XL-noIQ.gguf mtp-Qwen3.8-27B-d2t64k-swiftxl.gguf

sha256sum Swift-Qwen3.8-27B-Q4_K_XL-noIQ.gguf dflash-Qwen3.8-27B-Q4_0-d2t64k-swiftxl.gguf \
  mtp-Qwen3.8-27B-d2t64k-swiftxl.gguf
echo
echo "done. the inputs (F16 shards, BF16 drafter, imatrix) can be deleted to recover about 60 GB."
