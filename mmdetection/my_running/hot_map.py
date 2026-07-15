import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

def overlay_heatmap_on_image(img, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    将热力图叠加到原图上。
    Args:
        img: 原图，RGB 或 BGR 格式，shape (H, W, 3)
        heatmap: 特征图或注意力图，shape (H, W)
        alpha: 热力图透明度（0~1）
        colormap: 颜色映射方式，默认 JET
    Returns:
        overlayed_img: 叠加后的图像
    """

    # 归一化热力图到 [0, 255]
    heatmap = np.uint8(255 * (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8))
    print(heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, colormap)

    # 如果原图是灰度图 -> 转成三通道
    if len(img.shape) == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # 尺寸对齐（若特征图比原图小）
    heatmap_color = cv2.resize(heatmap_color, (img.shape[1], img.shape[0]))

    # 叠加
    overlayed_img = cv2.addWeighted(heatmap_color, alpha, img, 1 - alpha, 0)
    overlayed_img = cv2.cvtColor(overlayed_img, cv2.COLOR_BGR2RGB)
    return overlayed_img



rgb = torch.load(r"E:\pycharm\project\Swin-Deformable\mmdetection\work_dirs\visible\rgb.pt",weights_only=False)
rgb_fuse= torch.load(r"E:\pycharm\project\Swin-Deformable\mmdetection\work_dirs\visible\rgb_fuses.pt",weights_only=False)
img = rgb[0].permute(1, 2, 0).detach().cpu().numpy()  # (H,W,C)
img = (img - img.min()) / (img.max() - img.min())  # 归一化到0-1
img = np.uint8(img * 255)
heatmap = rgb_fuse[0][0, 15].detach().cpu().numpy()  # 取第1通道
overlay = overlay_heatmap_on_image(img, heatmap, alpha=1)
cv2.imshow('overlay', overlay)
cv2.waitKey(0)
plt.imshow(overlay)
plt.axis('off')
plt.show()
