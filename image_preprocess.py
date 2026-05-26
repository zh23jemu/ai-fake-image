import cv2
import numpy as np
import torch
import random
import os
from PIL import Image
from scipy import ndimage

IMAGE_SIZE = (224, 224)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMAGE_DATA_ROOT = r"E:\Python\AI_Detection\data\image"

def preprocess_image_fast(img_path, target_size=(224, 224)):
    """快速预处理函数（用于DataLoader多进程）- 增强版"""
    # 使用PIL快速读取
    img = Image.open(img_path).convert('RGB')
    
    # 转换为numpy（使用float32确保兼容性）
    img_np = np.array(img, dtype=np.float32) / 255.0
    
    # 多尺度噪声特征提取
    blur1 = ndimage.gaussian_filter(img_np, sigma=0.8)
    blur2 = ndimage.gaussian_filter(img_np, sigma=1.5)
    blur3 = ndimage.gaussian_filter(img_np, sigma=3.0)
    
    noise1 = img_np - blur1
    noise2 = img_np - blur2
    noise3 = img_np - blur3
    
    kernel = np.array([[-1, -1, -1], 
                       [-1,  8, -1], 
                       [-1, -1, -1]], dtype=np.float32) / 8.0
    
    high_pass1 = np.zeros_like(noise1)
    high_pass2 = np.zeros_like(noise2)
    high_pass3 = np.zeros_like(noise3)
    
    for c in range(3):
        high_pass1[:,:,c] = ndimage.convolve(noise1[:,:,c], kernel, mode='reflect')
        high_pass2[:,:,c] = ndimage.convolve(noise2[:,:,c], kernel, mode='reflect')
        high_pass3[:,:,c] = ndimage.convolve(noise3[:,:,c], kernel, mode='reflect')
    
    # 融合多尺度噪声特征（加权）
    high_pass = 0.5 * high_pass1 + 0.3 * high_pass2 + 0.2 * high_pass3
    high_pass = np.clip(high_pass, 0, 1)
    
    # 合并并转换
    combined = np.concatenate([img_np, high_pass], axis=-1)
    combined = np.transpose(combined, (2, 0, 1))
    
    return torch.from_numpy(combined).float()

# 保留原有函数用于向后兼容
def resize_image(img, target_size=IMAGE_SIZE, keep_ratio=True):
    if isinstance(img, Image.Image):
        img = np.array(img)
    h, w = img.shape[:2]
    tw, th = target_size
    if keep_ratio:
        scale = min(tw / w, th / h)
        nw, nh = int(w * scale), int(h * scale)
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        top = (th - nh) // 2
        bottom = th - nh - top
        left = (tw - nw) // 2
        right = tw - nw - left
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    else:
        img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_LINEAR)
    return img

def normalize_image(img, mean=MEAN, std=STD):
    img = img.astype(np.float32) / 255.0
    img = (img - mean) / std
    return img

def extract_noise_feature(img):
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    noise = cv2.absdiff(img, blur)
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    high_pass = cv2.filter2D(noise, -1, kernel)
    return high_pass

def random_flip(img, p=0.5):
    if random.random() < p:
        img = cv2.flip(img, 1)
    if random.random() < p:
        img = cv2.flip(img, 0)
    return img

def random_brightness(img, brightness_range=(0.8, 1.2)):
    if random.random() < 0.5:
        alpha = random.uniform(*brightness_range)
        img = np.clip(img * alpha, 0, 255).astype(np.uint8)
    return img

def add_gaussian_noise(img, mean=0, std=10, p=0.3):
    if random.random() < p:
        noise = np.random.normal(mean, std, img.shape).astype(np.float32)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
    return img

def preprocess_image(img_path, is_train=True, return_tensor=True, target_size=(224, 224)):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像：{img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    noise_img = extract_noise_feature(img)

    if is_train:
        if np.random.rand() > 0.5:
            img = cv2.flip(img, 1)
            noise_img = cv2.flip(noise_img, 1)
        if np.random.rand() > 0.5:
            alpha = 0.8 + np.random.rand() * 0.4
            img = np.clip(img * alpha, 0, 255).astype(np.uint8)
        if np.random.rand() > 0.5:
            beta = -10 + np.random.rand() * 20
            img = np.clip(img + beta, 0, 255).astype(np.uint8)

    img = img / 255.0
    noise_img = noise_img / 255.0
    combined = np.concatenate([img, noise_img], axis=-1)
    combined = np.transpose(combined, (2, 0, 1))

    if return_tensor:
        combined = torch.from_numpy(combined).float()
    return combined

def find_first_image():
    for split in ["train", "val", "test"]:
        for cls in ["real", "fake"]:
            cls_dir = os.path.join(IMAGE_DATA_ROOT, split, cls)
            if os.path.exists(cls_dir):
                img_files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
                if img_files:
                    return os.path.join(cls_dir, img_files[0])
    return None

if __name__ == "__main__":
    test_img_path = find_first_image()
    if test_img_path is None:
        print("❌ 未找到图像")
    else:
        print(f"✅ 测试图像：{test_img_path}")
        img_tensor = preprocess_image(test_img_path, is_train=True)
        print(f"输出形状：{img_tensor.shape}")