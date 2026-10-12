#!/usr/bin/env bash
# nano-cuda-moe 启动脚本: Qwen3.8-Flash-Next (125B MoE) 单卡 RTX3090 OpenAI 服务
# 用法: ./run.sh        (前台运行, Ctrl-C 退出)
#       PORT=18010 CTX=32768 ./run.sh
set -euo pipefail
cd "$(dirname "$0")"
exec ./build/bl-server \
  --model models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
  --host 0.0.0.0 \
  --port "${PORT:-18010}" \
  --ctx "${CTX:-32768}" \
  "$@"
