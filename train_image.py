import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
import warnings
import random
import matplotlib.pyplot as plt
from collections import defaultdict
import json

warnings.filterwarnings('ignore')

# 导入自定义模块
try:
    from datasets.image_dataset import ImageFakeDataset
except ImportError:
    # 当前仓库初版将数据集文件放在项目根目录；保留 fallback，便于服务器 clone 后直接运行。
    from image_dataset import ImageFakeDataset

try:
    from models.image_cnn import DoubleBranchCNN
except ImportError:
    # 当前仓库初版将模型文件放在项目根目录；保留 fallback，避免仅因包目录未整理而训练失败。
    from image_cnn import DoubleBranchCNN


# ====================== 过拟合监控类 ======================
class OverfittingMonitor:
    """监控模型过拟合情况"""

    def __init__(self):
        self.history = defaultdict(list)
        self.alert_thresholds = {
            'loss_gap': 0.3,
            'acc_gap': 0.05,
        }

    def update(self, metrics):
        """更新监控指标"""
        for key, value in metrics.items():
            self.history[key].append(value)

    def check_overfitting(self, model=None):
        """检测过拟合情况并返回详细报告"""
        alerts = []
        metrics = {}

        # 1. 检查损失差距
        if len(self.history['train_loss']) > 0 and len(self.history['val_loss']) > 0:
            train_loss = self.history['train_loss'][-1]
            val_loss = self.history['val_loss'][-1]
            loss_gap = val_loss - train_loss
            metrics['loss_gap'] = loss_gap

            if loss_gap > self.alert_thresholds['loss_gap']:
                alerts.append(f"🔴 严重过拟合：损失差距 {loss_gap:.4f} > {self.alert_thresholds['loss_gap']}")
            elif loss_gap > self.alert_thresholds['loss_gap'] * 0.6:
                alerts.append(f"🟡 轻微过拟合：损失差距 {loss_gap:.4f}")
            else:
                alerts.append(f"✅ 损失差距正常：{loss_gap:.4f}")

        # 2. 检查F1趋势
        if len(self.history['val_f1']) > 5:
            recent_f1 = self.history['val_f1'][-5:]
            f1_std = np.std(recent_f1)
            f1_trend = np.polyfit(range(5), recent_f1, 1)[0]
            metrics['f1_std'] = f1_std
            metrics['f1_trend'] = f1_trend

            if f1_std > 0.05:
                alerts.append(f"🔴 F1不稳定：标准差 {f1_std:.4f}")

            if f1_trend < -0.005:
                alerts.append(f"🟡 F1呈下降趋势：斜率 {f1_trend:.4f}")

        # 3. 计算过拟合分数
        overfitting_score = self._calculate_overfitting_score(metrics)
        metrics['overfitting_score'] = overfitting_score

        return alerts, metrics

    def _calculate_overfitting_score(self, metrics):
        """计算过拟合分数（0-1，越高越可能过拟合）"""
        score = 0.0
        
        if 'loss_gap' in metrics:
            score += 0.5 * min(metrics['loss_gap'] / 0.5, 1.0)

        if 'f1_std' in metrics:
            score += 0.3 * min(metrics['f1_std'] / 0.1, 1.0)

        if 'f1_trend' in metrics:
            trend_normalized = min(max(-metrics['f1_trend'] / 0.02, 0), 1.0)
            score += 0.2 * trend_normalized

        return score

    def plot_training_curves(self, save_path=None):
        """绘制训练曲线（单页显示）"""
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # 1. Loss曲线
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        ax1.plot(epochs, self.history['train_loss'], 'o-', label='Training Loss', 
                color='#E74C3C', linewidth=2.5, markersize=6)
        ax1.plot(epochs, self.history['val_loss'], 's-', label='Validation Loss', 
                color='#3498DB', linewidth=2.5, markersize=6)
        ax1.set_xlabel('Epoch', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Loss', fontsize=14, fontweight='bold')
        ax1.set_title('Image Model Training Loss Curve', fontsize=16, fontweight='bold', pad=20)
        ax1.legend(fontsize=12, loc='upper right')
        ax1.grid(alpha=0.3, linestyle='--', linewidth=1)
        ax1.tick_params(axis='both', labelsize=12)
        plt.tight_layout()
        
        if save_path:
            loss_path = save_path.replace('.png', '_loss.png')
            plt.savefig(loss_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✅ 损失曲线已保存: {loss_path}")
        plt.close()
        
        # 2. Accuracy曲线
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        ax2.plot(epochs, self.history['train_acc'], 'o-', label='Training Accuracy', 
                color='#E74C3C', linewidth=2.5, markersize=6)
        ax2.plot(epochs, self.history['val_acc'], 's-', label='Validation Accuracy', 
                color='#3498DB', linewidth=2.5, markersize=6)
        ax2.set_xlabel('Epoch', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
        ax2.set_title('Image Model Training Accuracy Curve', fontsize=16, fontweight='bold', pad=20)
        ax2.legend(fontsize=12, loc='lower right')
        ax2.grid(alpha=0.3, linestyle='--', linewidth=1)
        ax2.tick_params(axis='both', labelsize=12)
        plt.tight_layout()
        
        if save_path:
            acc_path = save_path.replace('.png', '_accuracy.png')
            plt.savefig(acc_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✅ 准确率曲线已保存: {acc_path}")
        plt.close()
        
        # 3. F1曲线
        fig3, ax3 = plt.subplots(figsize=(10, 8))
        ax3.plot(epochs, self.history['val_f1'], '^-', label='Validation F1-Score', 
                color='#2ECC71', linewidth=2.5, markersize=6)
        ax3.set_xlabel('Epoch', fontsize=14, fontweight='bold')
        ax3.set_ylabel('F1-Score', fontsize=14, fontweight='bold')
        ax3.set_title('Image Model F1-Score Curve', fontsize=16, fontweight='bold', pad=20)
        ax3.legend(fontsize=12, loc='lower right')
        ax3.grid(alpha=0.3, linestyle='--', linewidth=1)
        ax3.tick_params(axis='both', labelsize=12)
        plt.tight_layout()
        
        if save_path:
            f1_path = save_path.replace('.png', '_f1.png')
            plt.savefig(f1_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✅ F1曲线已保存: {f1_path}")
        plt.close()
        
        # 4. 学习率曲线
        if 'learning_rate' in self.history:
            fig4, ax4 = plt.subplots(figsize=(10, 8))
            ax4.semilogy(epochs, self.history['learning_rate'], 'm-', linewidth=2.5)
            ax4.set_xlabel('Epoch', fontsize=14, fontweight='bold')
            ax4.set_ylabel('Learning Rate', fontsize=14, fontweight='bold')
            ax4.set_title('Learning Rate Schedule (Cosine Annealing)', fontsize=16, fontweight='bold', pad=20)
            ax4.grid(alpha=0.3, linestyle='--', linewidth=1)
            ax4.tick_params(axis='both', labelsize=12)
            plt.tight_layout()
            
            if save_path:
                lr_path = save_path.replace('.png', '_lr.png')
                plt.savefig(lr_path, dpi=300, bbox_inches='tight', facecolor='white')
                print(f"✅ 学习率曲线已保存: {lr_path}")
            plt.close()


# ====================== 核心配置 ======================
class Config:
    # 数据路径
    DATA_ROOT = os.environ.get("IMAGE_DATA_ROOT", "data/image")
    # 模型保存路径
    SAVE_DIR = os.environ.get("IMAGE_SAVE_DIR", "models/weights")
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 结果保存路径
    RESULT_DIR = os.environ.get("IMAGE_RESULT_DIR", "results")
    os.makedirs(RESULT_DIR, exist_ok=True)

    # 训练参数。当前模型结构固定，优先通过数据质量、稳定增强和优化策略提升准确率。
    # 这些参数支持环境变量覆盖，便于在 Slurm 服务器上按 GPU 显存和排队时间快速调速，
    # 例如：IMAGE_BATCH_SIZE=64 IMAGE_EPOCHS=60 sbatch slurm/train_image.slurm。
    BATCH_SIZE = int(os.environ.get("IMAGE_BATCH_SIZE", "16"))
    EPOCHS = int(os.environ.get("IMAGE_EPOCHS", "120"))
    LR = 1e-4  # 降低学习率，配合噪声分支更稳定地学习细粒度伪造痕迹
    MIN_LR = 1e-6
    WEIGHT_DECAY = 1e-4  # 恢复标准正则化
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 梯度累积
    ACCUMULATION_STEPS = int(os.environ.get("IMAGE_ACCUMULATION_STEPS", "4"))  # 有效batch=BATCH_SIZE×ACCUMULATION_STEPS

    # 类别权重。训练集已在 Dataset 内按 real/fake 平衡采样，继续偏置 fake 会降低整体 accuracy。
    CLASS_WEIGHTS = torch.tensor([1.0, 1.0]).to(DEVICE)

    # 早停参数
    PATIENCE = int(os.environ.get("IMAGE_PATIENCE", "30"))
    MIN_DELTA = 0.001

    # 数据增强
    USE_AUGMENTATION = True
    AUGMENT_PROB = 0.5

    # 标签平滑。AI 生成检测依赖细微伪造痕迹，过强平滑会降低模型判别边界的锐度。
    LABEL_SMOOTHING = 0.02

    # 测试时增强。AI 生成检测依赖细微频域/噪声痕迹，简单加噪 TTA 可能冲淡判别信号。
    USE_TTA = False
    TTA_TIMES = 7  # 增加TTA次数

    # Warmup设置
    WARMUP_EPOCHS = int(os.environ.get("IMAGE_WARMUP_EPOCHS", "8"))

    # 混合精度训练
    USE_AMP = True

    # 梯度裁剪
    GRAD_CLIP = 1.0

    # 数据加载优化
    NUM_WORKERS = int(os.environ.get("IMAGE_NUM_WORKERS", "2"))
    PIN_MEMORY = True
    PREFETCH_FACTOR = int(os.environ.get("IMAGE_PREFETCH_FACTOR", "2"))

    # 最优阈值搜索
    OPTIMIZE_THRESHOLD = True
    
    # Focal Loss（处理难样本）
    USE_FOCAL_LOSS = False

    # 训练集每类最多采样数量。设为0或负数表示不限制；默认多用一些数据以覆盖不同生成器。
    MAX_SAMPLES_PER_CLASS = int(os.environ.get("IMAGE_MAX_SAMPLES_PER_CLASS", "60000"))

# ====================== Focal Loss实现 ======================
class FocalLoss(nn.Module):
    """Focal Loss - 专注于难分类样本"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.alpha, reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss)
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ====================== 评估函数 ======================
def find_optimal_threshold(y_true, y_probs):
    """寻找最优分类阈值以最大化F1或Precision"""
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    best_threshold = 0.5
    best_f1 = 0.0
    best_precision = 0.0
    
    thresholds = np.arange(0.3, 0.8, 0.01)
    
    for threshold in thresholds:
        y_pred = (y_probs >= threshold).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # 优先保证precision >= 0.8，然后最大化f1
        if precision >= 0.8 and f1 > best_f1:
            best_f1 = f1
            best_precision = precision
            best_threshold = threshold
    
    # 如果没有找到precision>=0.8的阈值，找最接近的
    if best_precision < 0.8:
        for threshold in thresholds:
            y_pred = (y_probs >= threshold).astype(int)
            precision = precision_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            if abs(precision - 0.8) < abs(best_precision - 0.8) or \
               (abs(precision - 0.8) == abs(best_precision - 0.8) and f1 > best_f1):
                best_precision = precision
                best_f1 = f1
                best_threshold = threshold
    
    return best_threshold, best_precision, best_f1


def evaluate_model(model, dataloader, criterion, device, verbose=False, use_tta=False, tta_times=3, threshold=0.5):
    """
    评估函数（支持自定义阈值）
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            if use_tta and tta_times > 1:
                batch_probs = []
                
                # 原始预测
                outputs = model(imgs)
                probs = torch.softmax(outputs, dim=1)
                batch_probs.append(probs)
                
                # TTA增强预测
                for _ in range(tta_times - 1):
                    imgs_aug = imgs + torch.randn_like(imgs) * 0.01
                    outputs_aug = model(imgs_aug)
                    probs_aug = torch.softmax(outputs_aug, dim=1)
                    batch_probs.append(probs_aug)
                
                probs = torch.stack(batch_probs).mean(0)
            else:
                outputs = model(imgs)
                probs = torch.softmax(outputs, dim=1)

            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)

            # 使用自定义阈值
            preds = (probs[:, 1] >= threshold).long()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    # 计算指标
    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    # 计算ROC AUC
    try:
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
    except:
        roc_auc = 0.5
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])

    if verbose:
        print(f"  评估详情:")
        print(f"    分类阈值: {threshold:.2f}")
        print(f"    预测分布: {np.bincount(all_preds)}")
        print(f"    混淆矩阵: TN={cm[0, 0]}, FP={cm[0, 1]}, FN={cm[1, 0]}, TP={cm[1, 1]}")
        print(f"    各类别数量 - Real: {all_labels.count(0)}, Fake: {all_labels.count(1)}")

    return {
        'loss': avg_loss,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }


# ====================== 训练函数 ======================
def train_image_model():
    cfg = Config()
    print(f"✅ 设备：{cfg.DEVICE}")
    print("=" * 60)
    print("训练配置（优化版）:")
    print(f"  - 学习率: {cfg.LR}")
    print(f"  - 权重衰减: {cfg.WEIGHT_DECAY}")
    print(f"  - 标签平滑: {cfg.LABEL_SMOOTHING}")
    print(f"  - 数据增强: {cfg.USE_AUGMENTATION}")
    print(f"  - 测试时增强: {cfg.USE_TTA} (TTA次数={cfg.TTA_TIMES})")
    print(f"  - 训练轮数: {cfg.EPOCHS}")
    print(f"  - 批次大小: {cfg.BATCH_SIZE} × {cfg.ACCUMULATION_STEPS} = {cfg.BATCH_SIZE * cfg.ACCUMULATION_STEPS}")
    print("=" * 60)

    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 初始化过拟合监控器
    monitor = OverfittingMonitor()

    # 启用cudnn自动调优器
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # 加载数据集
    print("\n📥 加载数据集...")
    # 适当减少训练数据量：15000张/类，平衡性能和内存
    train_dataset = ImageFakeDataset(
        root=cfg.DATA_ROOT, 
        split="train", 
        is_train=True,
        max_samples=cfg.MAX_SAMPLES_PER_CLASS  # 限制训练集每类最多样本数，便于按显存和训练时间调整
    )
    val_dataset = ImageFakeDataset(root=cfg.DATA_ROOT, split="val", is_train=False)
    test_dataset = ImageFakeDataset(root=cfg.DATA_ROOT, split="test", is_train=False)

    # 优化的DataLoader配置 - 解决内存溢出问题
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        persistent_workers=False if cfg.NUM_WORKERS == 0 else True,
        prefetch_factor=cfg.PREFETCH_FACTOR if cfg.NUM_WORKERS > 0 else None
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        persistent_workers=False if cfg.NUM_WORKERS == 0 else True,
        prefetch_factor=cfg.PREFETCH_FACTOR if cfg.NUM_WORKERS > 0 else None
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        persistent_workers=False if cfg.NUM_WORKERS == 0 else True,
        prefetch_factor=cfg.PREFETCH_FACTOR if cfg.NUM_WORKERS > 0 else None
    )

    # 初始化模型
    print("\n🔧 初始化模型...")
    model = DoubleBranchCNN(num_classes=2).to(cfg.DEVICE)
    
    # 选择损失函数
    if cfg.USE_FOCAL_LOSS:
        criterion = FocalLoss(alpha=cfg.CLASS_WEIGHTS, gamma=2.0)
        print("   损失函数: Focal Loss (gamma=2.0)")
    else:
        criterion = nn.CrossEntropyLoss(
            weight=cfg.CLASS_WEIGHTS,
            label_smoothing=cfg.LABEL_SMOOTHING
        )
        print("   损失函数: CrossEntropyLoss")
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.LR,
        weight_decay=cfg.WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )

    # 学习率调度器（Warmup + ReduceLROnPlateau + CosineAnnealingLR）
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=cfg.WARMUP_EPOCHS
    )
    
    plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=cfg.MIN_LR
    )
    
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.EPOCHS - cfg.WARMUP_EPOCHS,
        eta_min=cfg.MIN_LR
    )
    
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[cfg.WARMUP_EPOCHS]
    )

    # 初始化混合精度训练
    scaler = torch.cuda.amp.GradScaler() if cfg.USE_AMP and torch.cuda.is_available() else None

    best_f1 = 0.0
    patience = 0
    best_epoch = 0

    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   总参数量: {total_params:,}")
    print(f"   可训练参数: {trainable_params:,}")
    print(f"   混合精度训练: {'✅ 启用' if scaler else '❌ 禁用'}")
    print(f"   DataLoader Workers: {cfg.NUM_WORKERS}")
    
    # 显示显存使用情况
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(cfg.DEVICE) / 1024**3
        reserved = torch.cuda.memory_reserved(cfg.DEVICE) / 1024**3
        print(f"   当前显存分配: {allocated:.2f} GB")
        print(f"   当前显存保留: {reserved:.2f} GB")
        print(f"   显存上限: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # 开始训练
    print("\n🚀 开始训练...")
    for epoch in range(cfg.EPOCHS):
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []

        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        monitor.history['learning_rate'].append(current_lr)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.EPOCHS}")
        optimizer.zero_grad(set_to_none=True)
        for batch_idx, (imgs, labels) in enumerate(pbar):
            # 优化：直接使用.cuda(non_blocking=True)加速传输
            imgs, labels = imgs.to(cfg.DEVICE, non_blocking=True), labels.to(cfg.DEVICE, non_blocking=True)

            # 混合精度训练
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                    loss = loss / cfg.ACCUMULATION_STEPS

                scaler.scale(loss).backward()

                is_update_step = (
                    (batch_idx + 1) % cfg.ACCUMULATION_STEPS == 0
                    or (batch_idx + 1) == len(train_loader)
                )

                if is_update_step:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.GRAD_CLIP)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)  # 更快的zero_grad
            else:
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss = loss / cfg.ACCUMULATION_STEPS
                loss.backward()

                is_update_step = (
                    (batch_idx + 1) % cfg.ACCUMULATION_STEPS == 0
                    or (batch_idx + 1) == len(train_loader)
                )

                if is_update_step:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.GRAD_CLIP)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            train_loss += loss.item() * imgs.size(0) * cfg.ACCUMULATION_STEPS
            train_preds.extend(outputs.argmax(1).cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({
                "loss": f"{loss.item() * cfg.ACCUMULATION_STEPS:.3f}",
                "lr": f"{current_lr:.2e}"
            })

        avg_train_loss = train_loss / len(train_loader.dataset)
        train_acc = accuracy_score(train_labels, train_preds)

        monitor.history['train_loss'].append(avg_train_loss)
        monitor.history['train_acc'].append(train_acc)

        # 验证
        val_results = evaluate_model(
            model, val_loader, criterion, cfg.DEVICE, verbose=False
        )

        # 更新监控历史
        monitor.history['val_loss'].append(val_results['loss'])
        monitor.history['val_acc'].append(val_results['accuracy'])
        monitor.history['val_f1'].append(val_results['f1'])
        monitor.history['val_precision'].append(val_results['precision'])
        monitor.history['val_recall'].append(val_results['recall'])

        print(f"\n Epoch {epoch + 1} | train_loss={avg_train_loss:.4f} | val_loss={val_results['loss']:.4f} | "
              f"Train_Acc={train_acc:.4f} | Val_Acc={val_results['accuracy']:.4f} | "
              f"P={val_results['precision']:.4f} | R={val_results['recall']:.4f} | "
              f"F1={val_results['f1']:.4f} | AUC={val_results['roc_auc']:.4f}")

        # 学习率调度（每个epoch结束时调用）
        scheduler.step()
        plateau_scheduler.step(val_results['loss'])  # 基于验证损失动态调整

        # 过拟合检测
        alerts, overfitting_metrics = monitor.check_overfitting(model)
        monitor.history['overfitting_score'].append(overfitting_metrics['overfitting_score'])

        if alerts:
            for alert in alerts:
                print(f"  {alert}")

        # 保存最佳模型（基于F1）
        if val_results['f1'] > best_f1 + cfg.MIN_DELTA:
            best_f1 = val_results['f1']
            patience = 0
            best_epoch = epoch + 1

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_f1': best_f1,
                'val_results': val_results,
                'monitor_history': dict(monitor.history),
                'config': {k: v for k, v in cfg.__dict__.items() if not k.startswith('_')}
            }
            torch.save(checkpoint, os.path.join(cfg.SAVE_DIR, "image_best.pth"))
            print(f"✅ 最优模型已保存 | F1={best_f1:.4f}, Acc={val_results['accuracy']:.4f} (Epoch {best_epoch})")
        else:
            patience += 1
            print(f"⚠️ 未改善：{patience}/{cfg.PATIENCE} | 最佳F1={best_f1:.4f} (Epoch {best_epoch})")

            if patience >= cfg.PATIENCE:
                print("🛑 早停触发，训练结束")
                break

    # 绘制训练曲线
    print("\n📈 生成训练曲线...")
    monitor.plot_training_curves(save_path=os.path.join(cfg.RESULT_DIR, "image_training_curves.png"))

    # ====================== 测试阶段 ======================
    print("\n" + "=" * 60)
    print(" 开始测试最佳模型...")

    # 加载最佳模型
    checkpoint = None
    if os.path.exists(os.path.join(cfg.SAVE_DIR, "image_best.pth")):
        checkpoint = torch.load(
            os.path.join(cfg.SAVE_DIR, "image_best.pth"),
            map_location=cfg.DEVICE
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ 加载最佳模型 (Epoch {checkpoint['epoch'] + 1})")
        print(f"   验证集表现: F1={checkpoint['best_f1']:.4f}, Acc={checkpoint['val_results']['accuracy']:.4f}")

    # 阈值优化（如果需要）
    optimal_threshold = 0.5
    if cfg.OPTIMIZE_THRESHOLD:
        print("\n🔍 正在搜索最优分类阈值...")
        val_results_for_threshold = evaluate_model(
            model, val_loader, criterion, cfg.DEVICE, verbose=False, threshold=0.5
        )
        val_probs = np.array(val_results_for_threshold['probabilities'])
        val_labels = np.array(val_results_for_threshold['labels'])
        
        optimal_threshold, opt_precision, opt_f1 = find_optimal_threshold(val_labels, val_probs)
        print(f"✅ 最优阈值: {optimal_threshold:.2f}")
        print(f"   对应指标: Precision={opt_precision:.4f}, F1={opt_f1:.4f}")

    # 标准测试（使用最优阈值）
    print(f"\n📊 标准测试结果 (阈值={optimal_threshold:.2f}):")
    test_results = evaluate_model(
        model, test_loader, criterion, cfg.DEVICE, verbose=True, use_tta=False, threshold=optimal_threshold
    )

    # 带TTA的测试
    test_results_tta = None
    if cfg.USE_TTA:
        print(f"\n📊 测试时增强(TTA)结果 (阈值={optimal_threshold:.2f}):")
        test_results_tta = evaluate_model(
            model, test_loader, criterion, cfg.DEVICE, verbose=True, use_tta=True, tta_times=cfg.TTA_TIMES, threshold=optimal_threshold
        )

    # 输出最终结果
    print("\n" + "=" * 60)
    print("🏆 最终测试结果")
    print("=" * 60)
    print(f"分类阈值: {optimal_threshold:.2f}")
    print(f"标准测试:")
    print(f"  Accuracy:  {test_results['accuracy']:.4f} ({test_results['accuracy']*100:.2f}%)")
    print(f"  Precision: {test_results['precision']:.4f}")
    print(f"  Recall:    {test_results['recall']:.4f}")
    print(f"  F1-Score:  {test_results['f1']:.4f}")
    print(f"  ROC AUC:   {test_results['roc_auc']:.4f}")

    if test_results_tta:
        print(f"\nTTA测试:")
        print(f"  Accuracy:  {test_results_tta['accuracy']:.4f} ({test_results_tta['accuracy']*100:.2f}%)")
        print(f"  Precision: {test_results_tta['precision']:.4f}")
        print(f"  Recall:    {test_results_tta['recall']:.4f}")
        print(f"  F1-Score:  {test_results_tta['f1']:.4f}")
        print(f"  ROC AUC:   {test_results_tta['roc_auc']:.4f}")

    print("=" * 60)

    # 保存训练历史和测试结果
    results_file = os.path.join(cfg.SAVE_DIR, "train_results.json")
    
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
    
    results_dict = {
        'config': {k: str(v) if torch.is_tensor(v) else v
                   for k, v in cfg.__dict__.items() if not k.startswith('_')},
        'best_epoch': best_epoch,
        'val_results': {
            'f1': float(best_f1),
            'accuracy': float(checkpoint['val_results']['accuracy']) if checkpoint else None
        },
        'test_results': {k: float(v) if isinstance(v, (np.number, float)) else v
                         for k, v in test_results.items()
                         if k not in ['confusion_matrix', 'predictions', 'labels', 'probabilities']},
        'training_history': {
            'train_loss': [float(x) for x in monitor.history['train_loss']],
            'val_loss': [float(x) for x in monitor.history['val_loss']],
            'train_acc': [float(x) for x in monitor.history['train_acc']],
            'val_acc': [float(x) for x in monitor.history['val_acc']],
            'val_f1': [float(x) for x in monitor.history['val_f1']],
            'learning_rate': [float(x) for x in monitor.history['learning_rate']]
        }
    }
    
    if test_results_tta:
        results_dict['test_results_tta'] = {
            k: float(v) if isinstance(v, (np.number, float)) else v
            for k, v in test_results_tta.items()
            if k not in ['confusion_matrix', 'predictions', 'labels', 'probabilities']
        }
    
    results_dict = convert_to_serializable(results_dict)
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2)

    print(f"\n✅ 详细结果已保存至: {results_file}")

    return model, test_results


# ====================== 主函数 ======================
if __name__ == "__main__":
    # Windows专属：设置多线程启动方式
    if os.name == 'nt':
        torch.multiprocessing.freeze_support()

    # 清空GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 开始训练
    try:
        train_image_model()
    except Exception as e:
        print(f"\n❌ 训练过程出错：{e}")
        # 出错时保存模型
        if os.path.exists(Config.SAVE_DIR):
            with open(os.path.join(Config.SAVE_DIR, "train_error.log"), "w") as f:
                f.write(str(e))
        raise e
