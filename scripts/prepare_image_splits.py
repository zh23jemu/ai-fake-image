import argparse
import os
import random
import shutil
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def collect_images(root):
    """
    递归收集目录下的图像文件。

    Args:
        root: 需要扫描的目录路径。

    Returns:
        list[Path]: 按路径排序后的图像文件列表，确保多次划分时顺序稳定。
    """
    root = Path(root)
    if not root.exists():
        return []

    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def split_items(items, train_ratio, val_ratio, seed):
    """
    按固定随机种子划分 train/val/test。

    Args:
        items: 待划分文件列表。
        train_ratio: 训练集比例。
        val_ratio: 验证集比例。
        seed: 随机种子。

    Returns:
        tuple[list[Path], list[Path], list[Path]]: train、val、test 三个列表。
    """
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)

    train_end = int(len(shuffled) * train_ratio)
    val_end = train_end + int(len(shuffled) * val_ratio)
    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


def ensure_parent(path):
    """创建目标文件的父目录。"""
    path.parent.mkdir(parents=True, exist_ok=True)


def link_or_copy(src, dst, mode):
    """
    将源图像放入划分后的目标目录。

    Args:
        src: 原始图像路径。
        dst: 目标路径。
        mode: 写入模式。`symlink` 节省空间，`hardlink` 更像真实文件，`copy` 最兼容但最占空间。
    """
    ensure_parent(dst)
    if dst.exists():
        return

    if mode == "symlink":
        os.symlink(src.resolve(), dst)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"未知写入模式：{mode}")


def export_split(items, output_root, split, label, mode, source_root):
    """
    将一个类别的某个 split 导出到训练代码期望的目录结构。

    目标结构为：
        output_root/train/real/*.jpg
        output_root/train/fake/<生成器名>/*.jpg
        output_root/val/real/*.jpg
        output_root/val/fake/<生成器名>/*.jpg

    fake 图像保留生成器子目录，便于后续按生成器分析错误样本；
    Dataset 已支持递归读取，因此不会影响训练。
    """
    for src in items:
        rel = src.relative_to(source_root)
        dst = output_root / split / label / rel
        link_or_copy(src, dst, mode)


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="把 content 原始图像整理成 train/val/test/real/fake 训练目录。"
    )
    parser.add_argument("--content-root", default="content", help="原始 content 目录路径。")
    parser.add_argument("--output-root", default="data/image", help="输出训练数据目录。")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="训练集比例。")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="验证集比例。")
    parser.add_argument("--seed", type=int, default=42, help="随机划分种子。")
    parser.add_argument(
        "--mode",
        choices=["symlink", "hardlink", "copy"],
        default="symlink",
        help="输出方式：symlink 最省空间；Windows 无权限时可改 hardlink 或 copy。",
    )
    return parser.parse_args()


def main():
    """执行数据划分。"""
    args = parse_args()
    content_root = Path(args.content_root)
    output_root = Path(args.output_root)

    real_root = content_root / "real_images"
    fake_root = content_root / "fake_images"

    real_images = collect_images(real_root)
    fake_images = collect_images(fake_root)

    if not real_images or not fake_images:
        raise RuntimeError(
            "未找到 real 或 fake 图像，请确认 content/real_images 和 content/fake_images 是否存在。"
        )

    real_train, real_val, real_test = split_items(
        real_images, args.train_ratio, args.val_ratio, args.seed
    )
    fake_train, fake_val, fake_test = split_items(
        fake_images, args.train_ratio, args.val_ratio, args.seed
    )

    export_split(real_train, output_root, "train", "real", args.mode, real_root)
    export_split(real_val, output_root, "val", "real", args.mode, real_root)
    export_split(real_test, output_root, "test", "real", args.mode, real_root)

    export_split(fake_train, output_root, "train", "fake", args.mode, fake_root)
    export_split(fake_val, output_root, "val", "fake", args.mode, fake_root)
    export_split(fake_test, output_root, "test", "fake", args.mode, fake_root)

    print("数据划分完成：")
    print(f"  real: train={len(real_train)}, val={len(real_val)}, test={len(real_test)}")
    print(f"  fake: train={len(fake_train)}, val={len(fake_val)}, test={len(fake_test)}")
    print(f"  output_root={output_root}")
    print(f"  mode={args.mode}")


if __name__ == "__main__":
    main()
