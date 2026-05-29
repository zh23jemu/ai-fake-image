# AI 生成图像检测项目

本项目用于训练二分类模型，判断输入图像是真实图像还是 AI 生成/篡改图像。模型结构为固定的双分支 CNN，输入由原始 RGB 图像和噪声/高频特征拼接成 6 通道张量。

## 技术栈

- Python 3.10 或 3.11
- PyTorch / TorchVision
- NumPy / SciPy / OpenCV / Pillow
- scikit-learn
- matplotlib / tqdm

## 目录说明

- `image_cnn.py`：模型结构定义。
- `image_dataset.py`：图像数据集读取、数据增强、噪声特征提取和采样逻辑。
- `train_image.py`：训练、验证、保存最佳模型和最终测试入口。
- `image_preprocess.py`：兼容旧流程的预处理函数。
- `scripts/prepare_image_splits.py`：把 `content/` 原始数据整理成训练需要的 `data/image/train|val|test/real|fake` 结构。
- `scripts/evaluate_image_checkpoint.py`：评估已经训练好的 checkpoint。
- `requirements.txt`：除 PyTorch / TorchVision 外的通用依赖。
- `models/weights/`：训练后模型权重和结果文件目录。

## 环境准备

在项目根目录创建并使用本地虚拟环境：

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
```

如果使用 NVIDIA GPU，安装 CUDA 12.x 兼容的 PyTorch：

```bash
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

如果只使用 CPU，可按 PyTorch 官网当前命令安装 CPU 版本：

```bash
.venv/bin/pip install torch torchvision
```

安装项目通用依赖：

```bash
.venv/bin/pip install -r requirements.txt
```

## 数据准备

原始数据应放在以下结构中：

```text
content/
  real_images/
    *.jpg
  fake_images/
    generator_name/
      *.jpg
```

生成训练、验证和测试划分：

```bash
.venv/bin/python scripts/prepare_image_splits.py --content-root content --output-root data/image --mode symlink
```

如果当前系统不支持软链接，可以改用复制模式：

```bash
.venv/bin/python scripts/prepare_image_splits.py --content-root content --output-root data/image --mode copy
```

## 普通 Python 训练

直接使用普通 Python 命令启动训练：

```bash
.venv/bin/python train_image.py
```

常用训练参数可以通过环境变量覆盖。例如，为了冲击更高 Accuracy，推荐使用自然比例采样：

```bash
IMAGE_TRAIN_SAMPLING=natural \
IMAGE_MAX_SAMPLES_PER_CLASS=180000 \
IMAGE_PATIENCE=25 \
IMAGE_EPOCHS=90 \
IMAGE_BATCH_SIZE=64 \
IMAGE_EVAL_BATCH_SIZE=128 \
.venv/bin/python train_image.py
```

如果显存较大，可以提高 batch：

```bash
IMAGE_TRAIN_SAMPLING=natural \
IMAGE_MAX_SAMPLES_PER_CLASS=180000 \
IMAGE_PATIENCE=25 \
IMAGE_EPOCHS=90 \
IMAGE_BATCH_SIZE=128 \
IMAGE_EVAL_BATCH_SIZE=256 \
.venv/bin/python train_image.py
```

训练完成后会在 `models/weights/` 下保存：

- `image_best.pth`：验证 F1 最优模型。
- `image_best_acc.pth`：验证 Accuracy 最优模型。
- `train_results.json`：训练和测试结果摘要。

曲线图会保存在 `results/` 下。

## 评估已训练模型

如果已经有训练好的模型，可直接评估：

```bash
.venv/bin/python scripts/evaluate_image_checkpoint.py
```

默认优先加载：

```text
models/weights/image_best_acc.pth
```

也可以手动指定 checkpoint：

```bash
IMAGE_CHECKPOINT=models/weights/image_best.pth .venv/bin/python scripts/evaluate_image_checkpoint.py
```

评估结果会保存到：

```text
models/weights/eval_results.json
```

## 已达到的参考结果

当前较优训练配置使用自然比例采样，最终测试结果达到：

```text
Accuracy: 81.66%
Precision: 82.41%
Recall: 89.28%
F1-Score: 85.71%
ROC AUC: 90.26%
```

## 调参过程

本项目的模型结构是固定的，因此主要从数据、训练策略、评估指标和运行效率几方面优化。

1. 修复双分支输入对齐问题
   模型输入由原图 RGB 和噪声/高频特征组成。早期训练中，两路图像增强可能使用不同的随机裁剪、翻转和旋转参数，导致 6 通道输入空间错位。后续改为成对增强，让原图和噪声分支始终使用同一组几何变换。

2. 改进噪声/高频特征
   原先高频残差会直接裁剪负值，容易丢失一半边缘和纹理信息。后续改为保留高频残差幅值，并使用分位数归一化，使模型能看到更完整的生成图像痕迹。

3. 优化训练稳定性
   调低学习率，减弱标签平滑，关闭可能冲淡噪声痕迹的测试时加噪增强，并保留类别权重为 `[1.0, 1.0]`，避免在已经平衡采样时额外偏向某一类。

4. 加速数据加载
   大量图像和软链接目录会导致启动阶段很慢，因此数据划分脚本会生成 manifest 文件，训练时直接读取清单。在线噪声提取也从 SciPy 实现改为 OpenCV `GaussianBlur/filter2D`，显著改善 GPU 等待 CPU 预处理的问题。

5. 保存 Accuracy 最优模型
   早期只保存验证 F1 最优模型，但用户要求 Accuracy 达到 80% 以上。后续额外保存 `image_best_acc.pth`，并在评估脚本中默认使用验证集 Accuracy 搜索最佳阈值。

6. 改为自然比例采样
   验证集和测试集中的 fake 图像比例高于 real 图像。如果训练集强制 real/fake 1:1，模型在 Accuracy 指标上会有类别先验偏差。最终达标方案使用 `IMAGE_TRAIN_SAMPLING=natural`，让训练分布更接近验证/测试分布。

最终推荐训练命令为：

```bash
IMAGE_TRAIN_SAMPLING=natural \
IMAGE_MAX_SAMPLES_PER_CLASS=180000 \
IMAGE_PATIENCE=25 \
IMAGE_EPOCHS=90 \
IMAGE_BATCH_SIZE=128 \
IMAGE_EVAL_BATCH_SIZE=256 \
.venv/bin/python train_image.py
```

该方案最终测试 Accuracy 达到 `81.66%`，F1 达到 `85.71%`，ROC AUC 达到 `90.26%`。

## 注意事项

- 训练数据量较大时建议使用 GPU。
- 如果出现内存不足，可以降低 `IMAGE_BATCH_SIZE`、`IMAGE_EVAL_BATCH_SIZE`、`IMAGE_NUM_WORKERS` 或 `IMAGE_PREFETCH_FACTOR`。
- 如果只需要评估已有模型，不需要重新训练，直接运行 `scripts/evaluate_image_checkpoint.py` 即可。
