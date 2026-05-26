import os
import random
import torch
from torch.utils.data import Dataset
from preprocess.image_preprocess import preprocess_image_fast
from torchvision import transforms
from PIL import Image
import numpy as np

random.seed(42)

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
            return [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(('.jpg','.png','.jpeg','.bmp'))] if os.path.exists(d) else []

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

        # 优化的数据增强（平衡增强与特征保持）
        if is_train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # 缩小裁剪范围，保留更多原始信息
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05),  # 减弱颜色扰动
                transforms.RandomRotation(degrees=10),  # 降低旋转角度
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.15, scale=(0.02, 0.15))  # 降低擦除概率
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])

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
        
        # 优化：先应用transform再提取噪声，减少中间变量
        if self.transform:
            img_tensor = self.transform(img)
        
        # 提取噪声特征
        noise_feature = extract_noise_feature_fast(img)
        
        if self.transform:
            noise_tensor = self.transform(noise_feature)
        
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
    
    # 归一化并转换回uint8
    high_pass = np.clip(high_pass, 0, 1)
    high_pass_pil = Image.fromarray((high_pass * 255).astype(np.uint8))
    
    # 释放中间变量
    del img_np, blur, noise, high_pass
    
    return high_pass_pil
