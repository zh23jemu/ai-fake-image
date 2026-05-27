# 项目长期维护记录

## 项目目标

本项目用于训练 AI 生成图像检测模型，目标是基于真实图像与多类生成/篡改图像数据，训练一个二分类模型来判断输入图像是否为 fake。

## 技术栈

- Python
- PyTorch / TorchVision
- NumPy / SciPy / OpenCV / Pillow
- scikit-learn
- matplotlib / tqdm

## 当前架构

- `image_cnn.py`：定义图像二分类模型 `DoubleBranchCNN`。模型接收 6 通道输入，其中前 3 通道为原图，后 3 通道为噪声/高频特征；内部使用双分支残差网络、SE 注意力、CBAM 融合与全连接分类头。
- `image_dataset.py`：定义 `ImageFakeDataset` 数据集类。训练集会按 real/fake 做类别平衡，并在读取图像后额外提取噪声特征，与原图特征拼接为 6 通道张量。
- `image_preprocess.py`：提供图像缩放、归一化、噪声特征提取和兼容旧流程的预处理函数。
- `train_image.py`：包含训练配置、过拟合监控、评估、阈值优化、训练循环、最佳模型保存和测试评估逻辑。
- `scripts/setup_server_env.sh`：用于服务器首次运行时创建项目本地 `.venv`，安装 CUDA 12.x 兼容 PyTorch/torchvision 与通用训练依赖，并做最小导入检查。
- `content/`：当前保存原始数据快照，包括 `fake_images/`、`real_images/` 和 `progress.json`。

## 开发规范

- 默认使用项目本地 `.venv`，不要使用系统 Python。
- 修改文件前先读取现有内容，保持最小修改，不做无关重构。
- 新增复杂逻辑时使用较详细的中文注释，说明用途、关键逻辑、重要分支、参数、返回值和不明显实现细节。
- 数据集小文件默认允许入库；训练生成的大模型权重、检查点、本地虚拟环境、缓存、密钥和日志不入库。
- 本项目是深度学习训练项目，长时间训练优先准备 Slurm 运行方式；GPU 任务默认使用 `gpu` 分区、`gpo-ifv7xx` 账号和 `normal` QOS。不要默认使用 `aws` 分区，避免产生额外费用；短时任务可按需使用 `gpuHz` 分区。

## TODO

- 进一步整理项目包结构；当前 `train_image.py` 已增加根目录 fallback 导入，后续可再正式重构为 `datasets/`、`models/` 包目录。
- 使用 `scripts/prepare_image_splits.py` 生成训练代码期望的 `DATA_ROOT/train|val|test/real|fake` 数据结构。
- 继续清理兼容性细节；`train_image.py` 已改为项目内默认路径并支持环境变量覆盖，`image_preprocess.py` 的自测示例路径后续仍可整理。
- 维护 `requirements.txt` 中除 PyTorch/torchvision 之外的通用训练依赖；GPU 版 PyTorch 继续按服务器 CUDA 版本单独安装。
- 使用 `slurm/train_image.slurm` 在 Slurm GPU 节点训练，并根据服务器实际 GPU、内存和数据路径调整资源参数。

## Current Status

仓库当前已完成 Git 初版快照提交，并已推送到 GitHub public 仓库 `https://github.com/zh23jemu/ai-fake-image`。远端 `master` 使用分批提交历史承载完整文件树，包含模型、数据集、预处理、训练脚本、大量图片数据、项目维护记录、忽略规则和 Slurm 训练入口。当前正在不改变既定模型结构的前提下提升 AI 生成图像检测准确率，已补强数据增强、噪声特征、训练配置、数据划分流程，并为服务器 GPU 训练完善 Slurm 提交脚本。

## Recent Changes

- 创建项目级 `AGENTS.md`，记录项目目标、架构、规范、TODO、当前进度和风险。
- 创建 `.gitignore`，保留 `content/` 数据集进入 Git，同步时排除虚拟环境、缓存、日志、本地配置和常见大模型权重/检查点。
- 创建 `slurm/train_image.slurm`，为远端 GPU 集群训练准备默认提交入口。
- 初始化 Git 仓库并完成初版提交，提交哈希为 `1e910bff8d`。
- 创建 GitHub public 仓库 `zh23jemu/ai-fake-image`，并通过“代码配置先推、数据目录分批推”的方式完成完整文件树上传；远端最新提交为 `bc24e2f9`。
- 修复双分支输入增强错位问题：原图和噪声图现在使用同一组随机几何增强参数，避免 6 通道特征空间不对齐。
- 调整噪声特征提取方式，保留高频残差幅值信息，避免负残差被直接清零。
- 调整训练配置：默认使用项目内 `data/image`、`models/weights`、`results` 路径，降低学习率、减弱标签平滑、取消 fake 类额外权重、默认关闭加噪 TTA，并支持通过环境变量覆盖路径。
- 新增 `scripts/prepare_image_splits.py`，可将 `content/` 原始图像整理为训练脚本需要的 `train/val/test/real/fake` 结构。
- 增强 `slurm/train_image.slurm`：保留 `gpu` GPU 分区、`gpo-ifv7xx` 账号和 `normal` QOS，增加 GPU/nvidia-smi 诊断、PyTorch CUDA 可用性检查、训练路径环境变量覆盖、输出目录创建，以及缺少 `data/image` 时自动生成软链接数据划分。
- 新增 `requirements.txt`，记录 `numpy`、`scipy`、`opencv-python`、`pillow`、`scikit-learn`、`matplotlib`、`tqdm` 等通用依赖，并在注释中说明 GPU 版 PyTorch 需要使用 CUDA 12.x 兼容 wheel 单独安装。
- 新增 `scripts/setup_server_env.sh`，便于服务器尚未创建 `.venv` 时一键初始化虚拟环境和训练依赖；`slurm/train_image.slurm` 在缺少 `.venv` 时会提示先运行该脚本。
- 为服务器提速和排障补充训练环境变量：`IMAGE_BATCH_SIZE`、`IMAGE_EPOCHS`、`IMAGE_ACCUMULATION_STEPS`、`IMAGE_NUM_WORKERS`、`IMAGE_PREFETCH_FACTOR`、`IMAGE_PATIENCE`、`IMAGE_WARMUP_EPOCHS`；Slurm 默认改为 L40S 更合适的 batch/workers/epoch，并使用 `python -u` 实时刷新日志。
- 为 `ImageFakeDataset` 增加 split manifest 缓存和扫描进度日志，减少服务器重复扫描 `data/image` 下大量软链接导致的启动等待，并让 Slurm 日志能看到数据集加载进度。
- 更新 `scripts/prepare_image_splits.py`，在创建/复用训练软链接时直接写出 split manifest；Slurm 在 manifest 缺失时会自动重新运行数据准备脚本，避免训练阶段递归扫描 `data/image` 导致 I/O 等待。
- `scripts/prepare_image_splits.py` 优先用 `git ls-files` 从 Git 索引收集 `content/` 图片；当 `data/image` 已存在但 manifest 缺失时，Slurm 使用 `--manifest-only` 只生成清单，不再逐个检查软链接。
- 移除 `ReduceLROnPlateau(verbose=True)` 过时参数，兼容服务器当前 PyTorch 2.12 调度器签名。
- 将 `ImageFakeDataset` 在线噪声特征提取从 SciPy `ndimage` 实现切换为 OpenCV `GaussianBlur/filter2D`，保留高频残差幅值归一化语义，减少 DataLoader CPU 预处理瓶颈。
- 同步全局 Slurm 分区策略：GPU 默认分区从 `aws` 改为 `gpu`，仅在短时任务或用户明确指定时使用 `gpuHz`，避免 `aws` 额外费用。
- 修复 PyTorch 2.6+ checkpoint 加载默认 `weights_only=True` 导致的最终测试失败；训练脚本加载本项目可信 checkpoint 时显式设置 `weights_only=False`，并新增 `scripts/evaluate_image_checkpoint.py` 复用已保存最佳模型单独评估。
- 为学校要求的 Val_Acc 指标补充准确率导向流程：训练时额外保存 `models/weights/image_best_acc.pth`，评估脚本默认加载该 checkpoint，并用验证集 accuracy 搜索分类阈值。
- 为 L40S 提升吞吐并规避 DataLoader OOM：Slurm 默认 CPU 为 16、内存 96G，训练 batch 默认 128，验证/测试 batch 默认 256，DataLoader workers 默认 8、prefetch 默认 2；仍可用环境变量覆盖以平衡速度和内存。

## Next TODO

- 进一步整理项目运行入口，优先修复导入路径和数据目录结构不一致问题。
- 服务器首次拉取后先运行 `bash scripts/setup_server_env.sh` 创建 `.venv` 并安装 PyTorch CUDA 12.x 兼容依赖。
- 验证远端服务器上的最小导入、CUDA 可用性和 Slurm 训练启动流程。
- 远端服务器可从 GitHub public 仓库拉取当前完整快照，但首次 clone/pull 仍会因为图片数据量较大而耗时较长。
- 使用新的 Slurm 脚本重新训练，重点观察验证集 accuracy/F1 是否从 60% 段提升到 80% 目标附近，并保存 `results/` 下曲线和指标用于报告。
- 如果 `nvidia-smi dmon` 显示 GPU 利用率长期为 0%，优先检查 Slurm 日志是否进入 `Epoch`；若停在训练配置后，重点排查 DataLoader 首批样本加载、软链接数据路径和 CPU 预处理瓶颈。
- 首次生成 manifest 后再次提交作业应明显缩短数据集加载时间；如果 manifest 过期或损坏，代码会自动重新扫描并覆盖。
- 若 Python 进程处于 `D` 状态且 CPU 很低，通常是慢速文件系统 I/O 等待；优先取消作业、拉取最新版，通过数据准备脚本生成 manifest 后再提交训练。
- 若 `data/image` 已经存在但 manifest 缺失，应优先走 `scripts/prepare_image_splits.py --manifest-only`，避免慢速文件系统上的软链接存在性检查。
- 若 GPU 利用率仍呈现间歇性峰值和长时间 0%，继续关注在线图片解码、增强和噪声特征提取；必要时考虑预生成噪声特征缓存。
- 如果训练已保存 `models/weights/image_best.pth` 但最终测试阶段失败，可直接运行 `.venv/bin/python scripts/evaluate_image_checkpoint.py` 生成测试结果，无需重新训练。
- 下一轮冲击 80% Val_Acc 时优先使用更多训练样本和更长 patience，例如 `IMAGE_MAX_SAMPLES_PER_CLASS=60000 IMAGE_PATIENCE=20 sbatch --partition=gpu slurm/train_image.slurm`，并关注 `image_best_acc.pth`。
- 若 batch 128 稳定且显存仍有余量，可继续试 `IMAGE_BATCH_SIZE=192`；如出现 OOM，优先降低 `IMAGE_NUM_WORKERS`/`IMAGE_PREFETCH_FACTOR`，再回退到 `IMAGE_BATCH_SIZE=64`。
- 若训练已保存 `image_best_acc.pth` 后因 OOM 中断，可先运行 `.venv/bin/python scripts/evaluate_image_checkpoint.py` 复用已保存 checkpoint 得到测试结果。

## Open Issues

- 当前尚未正式重构为 `datasets/` 和 `models/` 包目录，但 `train_image.py` 已增加根目录 fallback 导入，可在当前仓库结构下直接运行。
- `train_image.py` 已改为项目内默认路径，并支持通过 `IMAGE_DATA_ROOT`、`IMAGE_SAVE_DIR`、`IMAGE_RESULT_DIR` 覆盖；`image_preprocess.py` 中的自测示例路径仍待后续整理。
- 当前 `content/` 数据集本身仍是原始结构；`slurm/train_image.slurm` 会在缺少 `data/image` 时自动运行 `scripts/prepare_image_splits.py` 生成训练用软链接划分，但首次生成仍需服务器文件系统支持 symlink。
- `content/progress.json` 记录的部分计数与当前文件统计存在 1 张左右差异，需要后续核对是否是进度记录偏移或数据缺失。

## Architecture Decisions

- 初版提交先保留当前代码和数据原貌，不在提交前进行目录重构，避免影响慢速远端同步时的可追溯性。
- 数据集目录 `content/` 暂不加入忽略规则，因为用户需要通过 Git 将项目快照同步到远端服务器。
- 模型权重和训练检查点按可膨胀的大文件处理，默认通过 `.gitignore` 排除，后续如需同步特定权重应明确指定。
- 准确率提升优先不改模型结构，先修复数据管线与训练策略：保证 RGB 分支和噪声分支空间对齐，保留高频残差信息，并减少会破坏生成痕迹的强增强。
- Slurm 训练入口采用“先检查 GPU/CUDA，再自动准备数据划分，最后启动训练”的流程，避免作业占用 GPU 后因环境或目录问题静默失败。
