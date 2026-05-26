#!/bin/bash
#
# 图像真假检测训练任务提交脚本。
# 默认面向 GPU 训练环境，使用项目约定的 aws 分区、gpo-ifv7xx 账号和 normal QOS。
# 如需临时切换分区，可使用：
#   sbatch --partition=目标分区 slurm/train_image.slurm

#SBATCH --job-name=ai-fake-image-train
#SBATCH --partition=aws
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail

# 切换到提交命令所在目录，确保相对路径和 Git 同步后的项目结构一致。
cd "${SLURM_SUBMIT_DIR}"

# 训练脚本依赖项目本地虚拟环境。远端首次运行前应先创建 .venv，
# 并安装 CUDA 12.x 兼容的 PyTorch，例如 cu126 版本，避免误装 CUDA 13.0 wheel。
if [ ! -x ".venv/bin/python" ]; then
  echo "未找到 .venv/bin/python，请先在项目根目录创建并配置本地虚拟环境。"
  exit 1
fi

# 日志目录由 Slurm 输出配置引用，提交前确保目录存在。
mkdir -p slurm/logs

# 使用项目本地解释器启动训练，避免依赖系统 Python 或 shell 激活状态。
.venv/bin/python train_image.py
