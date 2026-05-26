import os
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np

random.seed(42)

IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]


def _to_normalized_tensor(img):
    """将 PIL 图像转换为模型使用的标准化 Tensor。"""
    tensor = TF.to_tensor(img)
    return TF.normalize(tensor, mean=IMAGE_MEAN, std=IMAGE_STD)


class PairedImageNoiseTransform:
    """
    对原图和噪声图执行成对增强。

    模型的 6 通道输入由“原图 RGB + 噪声/高频特征 RGB”拼接而成，因此两路特征必须
    在空间位置上严格对齐。原来的实现对两张图分别调用随机 transform，随机裁剪、
    翻转和旋转参数会各不相同，模型实际看到的是错位特征，这会显著拉低准确率。
    """

    def __init__(self, is_train):
        self.is_train = is_train
        self.color_jitter = transforms.ColorJitter(
            brightness=0.08,
            contrast=0.08,
            saturation=0.06,
            hue=0.02,
        )

    def __call__(self, img, noise_img):
        """
        Args:
            img: 原始 RGB 图像。
            noise_img: 从同一张原图提取出的噪声/高频特征图。

        Returns:
            (img_tensor, noise_tensor): 空间增强完全一致的两路标准化张量。
        """
        if self.is_train:
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                img,
                scale=(0.9, 1.0),
                ratio=(0.95, 1.05),
            )
            img = TF.resized_crop(
                img, i, j, h, w, (224, 224), interpolation=InterpolationMode.BILINEAR
            )
            noise_img = TF.resized_crop(
                noise_img, i, j, h, w, (224, 224), interpolation=InterpolationMode.BILINEAR
            )

            if random.random() < 0.5:
                img = TF.hflip(img)
                noise_img = TF.hflip(noise_img)

            angle = random.uniform(-5.0, 5.0)
            img = TF.rotate(img, angle, interpolation=InterpolationMode.BILINEAR)
            noise_img = TF.rotate(noise_img, angle, interpolation=InterpolationMode.BILINEAR)

            # 只对原图做很轻的颜色扰动，避免直接破坏噪声分支中的生成痕迹。
            if random.random() < 0.3:
                img = self.color_jitter(img)
        else:
            img = TF.resize(img, 256, interpolation=InterpolationMode.BILINEAR)
            noise_img = TF.resize(noise_img, 256, interpolation=InterpolationMode.BILINEAR)
            img = TF.center_crop(img, 224)
            noise_img = TF.center_crop(noise_img, 224)

        return _to_normalized_tensor(img), _to_normalized_tensor(noise_img)

class ImageFakeDataset(Dataset):
    def __init__(self, root, split="train", is_train=False, transform=None, max_samples=None):
        """
        数据集类
        
        Args:
            root: 数据根目录
            split: 数据集分割 (train/val/test)
            is_train: 是否为训练集
            transform: 数据增强变换
            max_samples: 每个类别最大样本数（仅对训练集有效，None表示不限制）
        """
        self.root = root
        self.split = split
        self.is_train = is_train
        self.classes = ["real", "fake"]

        base = os.path.join(root, split)
        real_dir = os.path.join(base, "real")
        fake_dir = os.path.join(base, "fake")

        def get_files(d):
            if not os.path.exists(d):
                return []

            files = []
            for current_dir, _, filenames in os.walk(d):
                for f in filenames:
                    if f.lower().endswith((".jpg", ".png", ".jpeg", ".bmp")):
                        files.append(os.path.join(current_dir, f))
            return files

        self.real = get_files(real_dir)
        self.fake = get_files(fake_dir)

        # 优化：训练集强制类别平衡，验证/测试集保留全部数据
        if split == "train":
            # 训练集：确保类别平衡
            n = min(len(self.real), len(self.fake))
            
            # 如果指定了max_samples，则进一步限制数据量
            if max_samples is not None and max_samples > 0:
                n = min(n, max_samples)
                print(f"ℹ️  {split} 集数据量限制为每类 {max_samples} 张")
            
            self.real = random.sample(self.real, n)
            self.fake = random.sample(self.fake, n)
            print(f"✅ {split} 集平衡完成 | real={n}, fake={n}, 总计={2*n}")
        else:
            # 验证和测试集：使用全部数据，不进行采样
            n_real = len(self.real)
            n_fake = len(self.fake)
            print(f"✅ {split} 集（完整数据）| real={n_real}, fake={n_fake}, 总计={n_real + n_fake}")

        self.data = [(p,0) for p in self.real] + [(p,1) for p in self.fake]
        random.shuffle(self.data)

        # 原图与噪声图必须使用同一组几何增强参数，否则双分支输入会空间错位。
        self.transform = transform if transform is not None else PairedImageNoiseTransform(is_train)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        
        try:
            # 使用PIL直接读取，避免多次转换
            img = Image.open(path).convert('RGB')
        except Exception as e:
            print(f"Warning: Failed to load {path}, using placeholder")
            img = Image.new('RGB', (224, 224), color='gray')
        
        # 先从原图提取噪声特征，再对原图和噪声图执行成对增强，保证 6 通道空间对齐。
        noise_feature = extract_noise_feature_fast(img)
        img_tensor, noise_tensor = self.transform(img, noise_feature)
        
        # 合并张量
        combined = torch.cat([img_tensor, noise_tensor], dim=0)
        
        # 释放中间变量
        del img, noise_feature
        
        return combined, torch.tensor(label, dtype=torch.long)


def extract_noise_feature_fast(img_pil):
    """优化版：快速提取噪声特征，减少内存占用"""
    # 转换为numpy（使用float32以兼容scipy）
    img_np = np.array(img_pil, dtype=np.float32) / 255.0
    
    from scipy import ndimage
    # 高斯滤波
    blur = ndimage.gaussian_filter(img_np, sigma=1.5)
    
    # 计算噪声
    noise = img_np - blur
    
    # 高通滤波核
    kernel = np.array([[-1, -1, -1], 
                       [-1,  8, -1], 
                       [-1, -1, -1]], dtype=np.float32) / 8.0
    
    # 对每个通道应用卷积
    high_pass = np.zeros_like(noise)
    for c in range(3):
        high_pass[:,:,c] = ndimage.convolve(noise[:,:,c], kernel, mode='reflect')
    
    # 使用高频残差的幅值作为噪声线索，而不是直接裁剪负值。
    # 直接 np.clip(high_pass, 0, 1) 会把所有负残差清零，丢掉一半边缘/纹理信息；
    # 对 AI 生成检测来说，这类双向高频残差往往正是区分真实图像和生成图像的关键证据。
    high_pass = np.abs(high_pass)
    max_value = np.percentile(high_pass, 99.5)
    if max_value > 1e-6:
        high_pass = high_pass / max_value
    high_pass = np.clip(high_pass, 0, 1)
    high_pass_pil = Image.fromarray((high_pass * 255).astype(np.uint8))
    
    # 释放中间变量
    del img_np, blur, noise, high_pass
    
    return high_pass_pil
