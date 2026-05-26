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
- `content/`：当前保存原始数据快照，包括 `fake_images/`、`real_images/` 和 `progress.json`。

## 开发规范

- 默认使用项目本地 `.venv`，不要使用系统 Python。
- 修改文件前先读取现有内容，保持最小修改，不做无关重构。
- 新增复杂逻辑时使用较详细的中文注释，说明用途、关键逻辑、重要分支、参数、返回值和不明显实现细节。
- 数据集小文件默认允许入库；训练生成的大模型权重、检查点、本地虚拟环境、缓存、密钥和日志不入库。
- 本项目是深度学习训练项目，长时间训练优先准备 Slurm 运行方式；GPU 任务默认使用 `aws` 分区、`gpo-ifv7xx` 账号和 `normal` QOS。

## TODO

- 进一步整理项目包结构；当前 `train_image.py` 已增加根目录 fallback 导入，后续可再正式重构为 `datasets/`、`models/` 包目录。
- 使用 `scripts/prepare_image_splits.py` 生成训练代码期望的 `DATA_ROOT/train|val|test/real|fake` 数据结构。
- 继续清理兼容性细节；`train_image.py` 已改为项目内默认路径并支持环境变量覆盖，`image_preprocess.py` 的自测示例路径后续仍可整理。
- 补充依赖文件，例如 `requirements.txt` 或 `pyproject.toml`。
- 准备 Slurm 提交脚本，并根据服务器实际 GPU、内存和数据路径调整资源参数。

## Current Status

仓库当前已完成 Git 初版快照提交，并已推送到 GitHub public 仓库 `https://github.com/zh23jemu/ai-fake-image`。远端 `master` 使用分批提交历史承载完整文件树，包含模型、数据集、预处理、训练脚本、大量图片数据、项目维护记录、忽略规则和 Slurm 训练入口。当前正在不改变既定模型结构的前提下提升 AI 生成图像检测准确率，重点修复数据增强、噪声特征、训练配置和数据划分流程。

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

## Next TODO

- 进一步整理项目运行入口，优先修复导入路径和数据目录结构不一致问题。
- 根据服务器环境创建 `.venv` 并安装 PyTorch CUDA 12.x 兼容依赖。
- 补充依赖声明文件，并验证远端服务器上的最小导入与训练启动流程。
- 远端服务器可从 GitHub public 仓库拉取当前完整快照，但首次 clone/pull 仍会因为图片数据量较大而耗时较长。
- 在服务器上先运行数据划分脚本，再使用新的训练配置重新训练，重点观察验证集 accuracy/F1 是否从 60% 段提升到 80% 目标附近。

## Open Issues

- 当前尚未正式重构为 `datasets/` 和 `models/` 包目录，但 `train_image.py` 已增加根目录 fallback 导入，可在当前仓库结构下直接运行。
- `train_image.py` 已改为项目内默认路径，并支持通过 `IMAGE_DATA_ROOT`、`IMAGE_SAVE_DIR`、`IMAGE_RESULT_DIR` 覆盖；`image_preprocess.py` 中的自测示例路径仍待后续整理。
- 当前 `content/` 数据集本身仍是原始结构，需要运行 `scripts/prepare_image_splits.py` 生成训练用划分目录。
- `content/progress.json` 记录的部分计数与当前文件统计存在 1 张左右差异，需要后续核对是否是进度记录偏移或数据缺失。

## Architecture Decisions

- 初版提交先保留当前代码和数据原貌，不在提交前进行目录重构，避免影响慢速远端同步时的可追溯性。
- 数据集目录 `content/` 暂不加入忽略规则，因为用户需要通过 Git 将项目快照同步到远端服务器。
- 模型权重和训练检查点按可膨胀的大文件处理，默认通过 `.gitignore` 排除，后续如需同步特定权重应明确指定。
- 准确率提升优先不改模型结构，先修复数据管线与训练策略：保证 RGB 分支和噪声分支空间对齐，保留高频残差信息，并减少会破坏生成痕迹的强增强。
