import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from image_cnn import DoubleBranchCNN
from image_dataset import ImageFakeDataset
from train_image import Config, evaluate_model, find_optimal_threshold


def main():
    """
    单独评估已经训练好的最佳 checkpoint。

    训练作业如果在最终测试阶段因为 PyTorch checkpoint 兼容性报错，可以直接运行
    本脚本默认优先复用面向 Val_Acc 保存的 `models/weights/image_best_acc.pth`；
    如该文件不存在，可通过 IMAGE_CHECKPOINT 指向 `image_best.pth`。checkpoint 是
    本项目训练脚本生成的可信本地文件，因此加载时显式使用 weights_only=False。
    """
    cfg = Config()
    default_checkpoint = Path(cfg.SAVE_DIR) / "image_best_acc.pth"
    if not default_checkpoint.exists():
        default_checkpoint = Path(cfg.SAVE_DIR) / "image_best.pth"
    checkpoint_path = Path(os.environ.get("IMAGE_CHECKPOINT", default_checkpoint))
    output_path = Path(os.environ.get("IMAGE_EVAL_OUTPUT", Path(cfg.SAVE_DIR) / "eval_results.json"))
    target_metric = os.environ.get("IMAGE_THRESHOLD_METRIC", "accuracy")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"未找到 checkpoint: {checkpoint_path}")

    print(f"✅ 设备：{cfg.DEVICE}")
    print(f"✅ 加载 checkpoint: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=cfg.DEVICE,
        weights_only=False,
    )

    model = DoubleBranchCNN(num_classes=2).to(cfg.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.CrossEntropyLoss(
        weight=cfg.CLASS_WEIGHTS,
        label_smoothing=cfg.LABEL_SMOOTHING,
    )

    val_dataset = ImageFakeDataset(root=cfg.DATA_ROOT, split="val", is_train=False)
    test_dataset = ImageFakeDataset(root=cfg.DATA_ROOT, split="test", is_train=False)

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        persistent_workers=False if cfg.NUM_WORKERS == 0 else True,
        prefetch_factor=cfg.PREFETCH_FACTOR if cfg.NUM_WORKERS > 0 else None,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        persistent_workers=False if cfg.NUM_WORKERS == 0 else True,
        prefetch_factor=cfg.PREFETCH_FACTOR if cfg.NUM_WORKERS > 0 else None,
    )

    print("🔍 搜索验证集最优分类阈值...")
    val_results = evaluate_model(model, val_loader, criterion, cfg.DEVICE, verbose=False, threshold=0.5)
    val_probs = np.array(val_results["probabilities"])
    val_labels = np.array(val_results["labels"])
    threshold, opt_precision, opt_f1 = find_optimal_threshold(
        val_labels,
        val_probs,
        target_metric=target_metric,
    )
    print(f"✅ 最优阈值: {threshold:.2f} | Precision={opt_precision:.4f} | F1={opt_f1:.4f}")

    print("📊 评估测试集...")
    test_results = evaluate_model(
        model,
        test_loader,
        criterion,
        cfg.DEVICE,
        verbose=True,
        use_tta=False,
        threshold=threshold,
    )

    print("🏆 测试结果")
    print(f"  Accuracy:  {test_results['accuracy']:.4f} ({test_results['accuracy'] * 100:.2f}%)")
    print(f"  Precision: {test_results['precision']:.4f}")
    print(f"  Recall:    {test_results['recall']:.4f}")
    print(f"  F1-Score:  {test_results['f1']:.4f}")
    print(f"  ROC AUC:   {test_results['roc_auc']:.4f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": str(checkpoint_path),
                "checkpoint_epoch": int(checkpoint.get("epoch", -1)) + 1,
                "best_val_f1": float(checkpoint.get("best_f1", 0.0)),
                "best_val_accuracy": float(checkpoint.get("best_accuracy", checkpoint.get("val_results", {}).get("accuracy", 0.0))),
                "threshold_metric": target_metric,
                "threshold": float(threshold),
                "threshold_precision": float(opt_precision),
                "threshold_f1": float(opt_f1),
                "test_results": {
                    k: float(v) if isinstance(v, (np.number, float)) else v
                    for k, v in test_results.items()
                    if k not in ["confusion_matrix", "predictions", "labels", "probabilities"]
                },
                "confusion_matrix": test_results["confusion_matrix"].tolist(),
            },
            f,
            indent=2,
        )
    print(f"✅ 评估结果已保存: {output_path}")


if __name__ == "__main__":
    main()
