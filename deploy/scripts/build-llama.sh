#!/bin/bash
# Build this llama.cpp fork (gfx1030 / Radeon PRO V620 tuning, speculative sampling, reduced-
# vocabulary drafters). By default it builds the checkout this script lives in.
#
#   deploy/scripts/build-llama.sh
#
# Environment:
#   LLAMA_DIR   checkout to build   (default: the one containing this script; cloned if missing)
#   LLAMA_REPO  git URL for a clone (default: https://github.com/sixvolts/llama-navi21-furnace.git)
#   LLAMA_REF   branch or commit to check out first (default: leave the checkout as it is)
#   LLVM        LLVM/clang prefix   (default: /usr/lib/llvm-21)
#   JOBS        parallel jobs       (default: nproc)
set -euo pipefail

LLAMA_DIR=${LLAMA_DIR:-"$(cd "$(dirname "$0")/../.." && pwd)"}
LLAMA_REPO=${LLAMA_REPO:-https://github.com/sixvolts/llama-navi21-furnace.git}
LLAMA_REF=${LLAMA_REF:-}
LLVM=${LLVM:-/usr/lib/llvm-21}
JOBS=${JOBS:-$(nproc)}

# cmake/ninja from install-packages.sh's venv, if present
[ -x "$HOME/.venv-llama/bin/cmake" ] && export PATH="$HOME/.venv-llama/bin:$PATH"

if [ ! -d "$LLAMA_DIR/.git" ]; then
  git clone "$LLAMA_REPO" "$LLAMA_DIR"
fi
cd "$LLAMA_DIR"
if [ -n "$LLAMA_REF" ]; then
  git fetch origin
  git checkout "$LLAMA_REF"
fi

# Ubuntu installs ROCm under /usr. CMake's HIP support does not accept the hipcc wrapper and
# Ubuntu's multiarch layout hides the HIP cmake package, hence the explicit compiler, device
# library and package paths.
export ROCM_PATH=/usr HIP_PATH=/usr
CC=$LLVM/bin/clang CXX=$LLVM/bin/clang++ \
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1030 -DGGML_HIP_ROCWMMA_FATTN=OFF \
  -DCMAKE_HIP_COMPILER=$LLVM/bin/clang++ \
  -DCMAKE_HIP_FLAGS="--rocm-path=/usr --rocm-device-lib-path=$(ls -d $LLVM/lib/clang/*/amdgcn/bitcode | head -1)" \
  -DCMAKE_HIP_COMPILER_ROCM_LIB=/usr/lib/x86_64-linux-gnu \
  -DCMAKE_HIP_LIBRARY_ARCHITECTURE=x86_64-linux-gnu \
  -DCMAKE_PREFIX_PATH=/usr/lib/x86_64-linux-gnu/cmake \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF

cmake --build build -j "$JOBS" --target llama-server llama-cli llama-bench llama-quantize llama-perplexity

echo
echo "built: $LLAMA_DIR/build/bin/llama-server ($(git rev-parse --short HEAD))"
