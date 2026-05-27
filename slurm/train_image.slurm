#!/bin/bash
#
# 图像真假检测训练任务提交脚本。
# 默认面向 GPU 训练环境，使用项目约定的 gpu 分区、gpo-ifv7xx 账号和 normal QOS。
# 不默认使用 aws 分区，避免产生额外费用；短时任务可在提交时覆盖到 gpuHz。
# 如需临时切换分区，可使用：
#   sbatch --partition=目标分区 slurm/train_image.slurm

#SBATCH --job-name=ai-fake-image-train
#SBATCH --partition=gpu
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail

# 切换到提交命令所在目录，确保相对路径和 Git 同步后的项目结构一致。
cd "${SLURM_SUBMIT_DIR}"

# 日志目录由 Slurm 输出配置引用，提交前确保目录存在。
mkdir -p slurm/logs

echo "===== Slurm 作业信息 ====="
echo "作业 ID: ${SLURM_JOB_ID:-unknown}"
echo "节点列表: ${SLURM_JOB_NODELIST:-unknown}"
echo "提交目录: ${SLURM_SUBMIT_DIR}"
echo "当前目录: $(pwd)"
echo "开始时间: $(date)"

# 打印 GPU 信息，便于确认作业确实运行在 GPU 节点上；如果集群未提供 nvidia-smi，
# 这里不会中断训练，后续仍会通过 PyTorch CUDA 检查决定是否继续。
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "===== GPU 信息 ====="
  nvidia-smi
else
  echo "未找到 nvidia-smi，将继续使用 PyTorch 检查 CUDA 可用性。"
fi

# 训练脚本依赖项目本地虚拟环境。远端首次运行前应先创建 .venv，
# 并安装 CUDA 12.x 兼容的 PyTorch，例如 cu126 版本，避免误装 CUDA 13.0 wheel。
if [ ! -x ".venv/bin/python" ]; then
  echo "未找到 .venv/bin/python，请先在项目根目录创建并配置本地虚拟环境。"
  echo "推荐先在服务器项目根目录运行："
  echo "  bash scripts/setup_server_env.sh"
  exit 1
fi

# 训练输出路径和采样上限都允许在 sbatch 前通过环境变量覆盖，便于在不同服务器磁盘上复用。
# 例如：
#   IMAGE_BATCH_SIZE=128 IMAGE_EVAL_BATCH_SIZE=256 IMAGE_EPOCHS=60 IMAGE_MAX_SAMPLES_PER_CLASS=60000 sbatch slurm/train_image.slurm
export IMAGE_DATA_ROOT="${IMAGE_DATA_ROOT:-data/image}"
export IMAGE_SAVE_DIR="${IMAGE_SAVE_DIR:-models/weights}"
export IMAGE_RESULT_DIR="${IMAGE_RESULT_DIR:-results}"

# L40S 显存较大，默认使用比笔记本 GPU 更高的 batch 和 worker 数来提高吞吐。
# 如果遇到显存不足，可在提交时降低 IMAGE_BATCH_SIZE；如果 GPU 利用率低，可继续提高 IMAGE_NUM_WORKERS。
export IMAGE_BATCH_SIZE="${IMAGE_BATCH_SIZE:-128}"
export IMAGE_EVAL_BATCH_SIZE="${IMAGE_EVAL_BATCH_SIZE:-256}"
export IMAGE_ACCUMULATION_STEPS="${IMAGE_ACCUMULATION_STEPS:-1}"
export IMAGE_EPOCHS="${IMAGE_EPOCHS:-60}"
export IMAGE_PATIENCE="${IMAGE_PATIENCE:-12}"
export IMAGE_WARMUP_EPOCHS="${IMAGE_WARMUP_EPOCHS:-4}"
export IMAGE_NUM_WORKERS="${IMAGE_NUM_WORKERS:-8}"
export IMAGE_PREFETCH_FACTOR="${IMAGE_PREFETCH_FACTOR:-2}"
export IMAGE_MAX_SAMPLES_PER_CLASS="${IMAGE_MAX_SAMPLES_PER_CLASS:-30000}"

mkdir -p "${IMAGE_SAVE_DIR}" "${IMAGE_RESULT_DIR}"

echo "===== Python 与 CUDA 检查 ====="
".venv/bin/python" - <<'PY'
import sys

try:
    import torch
except Exception as exc:
    raise SystemExit(
        "无法导入 torch，请先安装 CUDA 12.x 兼容的 PyTorch，例如：\n"
        "  .venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126\n"
        f"原始错误：{exc}"
    )

print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version from torch: {torch.version.cuda}")

if not torch.cuda.is_available():
    raise SystemExit("当前作业未检测到可用 CUDA，请确认 Slurm GPU 资源、驱动和 PyTorch wheel 是否匹配。")

print(f"GPU count: {torch.cuda.device_count()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")
PY

# 如果服务器刚从 GitHub 拉取，还没有生成训练脚本需要的 train/val/test 目录，
# 则自动基于 content/ 原始数据创建软链接划分，节省磁盘空间和同步时间。
DATA_DIR_READY=0
if [ -d "${IMAGE_DATA_ROOT}/train/real" ] && \
   [ -d "${IMAGE_DATA_ROOT}/train/fake" ] && \
   [ -d "${IMAGE_DATA_ROOT}/val/real" ] && \
   [ -d "${IMAGE_DATA_ROOT}/val/fake" ] && \
   [ -d "${IMAGE_DATA_ROOT}/test/real" ] && \
   [ -d "${IMAGE_DATA_ROOT}/test/fake" ]; then
  DATA_DIR_READY=1
fi

if [ ! -d "${IMAGE_DATA_ROOT}/train/real" ] || \
   [ ! -d "${IMAGE_DATA_ROOT}/train/fake" ] || \
   [ ! -d "${IMAGE_DATA_ROOT}/val/real" ] || \
   [ ! -d "${IMAGE_DATA_ROOT}/val/fake" ] || \
   [ ! -d "${IMAGE_DATA_ROOT}/test/real" ] || \
   [ ! -d "${IMAGE_DATA_ROOT}/test/fake" ] || \
   [ ! -f "${IMAGE_DATA_ROOT}/.train_manifest.json" ] || \
   [ ! -f "${IMAGE_DATA_ROOT}/.val_manifest.json" ] || \
   [ ! -f "${IMAGE_DATA_ROOT}/.test_manifest.json" ]; then
  echo "===== 准备训练数据划分 ====="
  echo "未检测到完整的数据目录或 manifest，将从 content/ 自动生成软链接划分和文件清单。"
  if [ "${DATA_DIR_READY}" -eq 1 ]; then
    echo "检测到 ${IMAGE_DATA_ROOT} 数据目录已存在，仅使用 Git 索引快速生成 manifest。"
    ".venv/bin/python" scripts/prepare_image_splits.py \
      --content-root content \
      --output-root "${IMAGE_DATA_ROOT}" \
      --mode symlink \
      --manifest-only
  else
    ".venv/bin/python" scripts/prepare_image_splits.py \
      --content-root content \
      --output-root "${IMAGE_DATA_ROOT}" \
      --mode symlink
  fi
fi

echo "===== 训练配置 ====="
echo "IMAGE_DATA_ROOT=${IMAGE_DATA_ROOT}"
echo "IMAGE_SAVE_DIR=${IMAGE_SAVE_DIR}"
echo "IMAGE_RESULT_DIR=${IMAGE_RESULT_DIR}"
echo "IMAGE_BATCH_SIZE=${IMAGE_BATCH_SIZE}"
echo "IMAGE_EVAL_BATCH_SIZE=${IMAGE_EVAL_BATCH_SIZE}"
echo "IMAGE_ACCUMULATION_STEPS=${IMAGE_ACCUMULATION_STEPS}"
echo "IMAGE_EPOCHS=${IMAGE_EPOCHS}"
echo "IMAGE_PATIENCE=${IMAGE_PATIENCE}"
echo "IMAGE_WARMUP_EPOCHS=${IMAGE_WARMUP_EPOCHS}"
echo "IMAGE_NUM_WORKERS=${IMAGE_NUM_WORKERS}"
echo "IMAGE_PREFETCH_FACTOR=${IMAGE_PREFETCH_FACTOR}"
echo "IMAGE_MAX_SAMPLES_PER_CLASS=${IMAGE_MAX_SAMPLES_PER_CLASS}"

# 使用 -u 关闭 Python 标准输出缓冲，确保 tail -f Slurm 日志时能实时看到数据集加载和 epoch 进度。
".venv/bin/python" -u train_image.py

echo "训练结束时间: $(date)"
