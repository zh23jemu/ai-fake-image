#!/bin/bash
#
# 服务器环境初始化脚本。
# 用途：
#   1. 在项目根目录创建本地虚拟环境 .venv。
#   2. 安装 CUDA 12.x 兼容的 PyTorch/torchvision，避免误装 CUDA 13.0 或 CPU-only wheel。
#   3. 安装 requirements.txt 中记录的通用训练依赖。
#   4. 做最小导入检查；如果当前节点可见 GPU，则同时打印 CUDA 可用性。
#
# 使用方式：
#   bash scripts/setup_server_env.sh
#
# 如果服务器默认 Python 不合适，可以显式指定解释器：
#   PYTHON_BIN=python3.11 bash scripts/setup_server_env.sh

set -euo pipefail

cd "$(dirname "$0")/.."

echo "===== 选择 Python 解释器 ====="

if [ -n "${PYTHON_BIN:-}" ]; then
  SELECTED_PYTHON="${PYTHON_BIN}"
else
  SELECTED_PYTHON=""
  for candidate in python3.11 python3.10 python3.12 python3 python; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      SELECTED_PYTHON="${candidate}"
      break
    fi
  done
fi

if [ -z "${SELECTED_PYTHON}" ]; then
  echo "未找到可用 Python，请先在服务器加载 Python 模块或安装 Python 3.10/3.11。"
  exit 1
fi

"${SELECTED_PYTHON}" - <<'PY'
import sys

version = sys.version_info
print(f"使用 Python: {sys.executable}")
print(f"Python 版本: {version.major}.{version.minor}.{version.micro}")

if version < (3, 10):
    raise SystemExit("当前 Python 版本过低，建议使用 Python 3.10 或 3.11 创建 .venv。")
PY

echo "===== 创建或复用 .venv ====="

if [ ! -x ".venv/bin/python" ]; then
  "${SELECTED_PYTHON}" -m venv .venv
else
  echo "检测到已有 .venv，将在现有虚拟环境中安装/更新依赖。"
fi

echo "===== 升级 pip 基础工具 ====="
".venv/bin/python" -m pip install --upgrade pip setuptools wheel

echo "===== 安装 CUDA 12.x 兼容 PyTorch ====="
".venv/bin/pip" install torch torchvision --index-url https://download.pytorch.org/whl/cu126

echo "===== 安装项目通用依赖 ====="
".venv/bin/pip" install -r requirements.txt

echo "===== 最小导入与 CUDA 检查 ====="
".venv/bin/python" - <<'PY'
import cv2
import matplotlib
import numpy
import PIL
import scipy
import sklearn
import torch
import torchvision
import tqdm

print(f"torch: {torch.__version__}")
print(f"torchvision: {torchvision.__version__}")
print(f"torch CUDA 编译版本: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
else:
    print("当前节点未检测到 CUDA。若这是登录节点，属于正常情况；训练作业会在 GPU Slurm 节点再次检查。")

print("服务器虚拟环境初始化完成。")
PY
