import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

def overlay_heatmap_on_image(img, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    将热力图叠加到原图上。
    """
    # 确保 heatmap 是 uint8 单通道
    if heatmap.dtype != np.uint8:
        heatmap = np.uint8(255 * (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8))

    heatmap_color = cv2.applyColorMap(heatmap, colormap)

    # 如果原图是灰度图 -> 转成三通道
    if len(img.shape) == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # 尺寸对齐
    heatmap_color = cv2.resize(heatmap_color, (img.shape[1], img.shape[0]))

    # 叠加
    overlayed_img = cv2.addWeighted(heatmap_color, alpha, img, 1 - alpha, 0)
    overlayed_img = cv2.cvtColor(overlayed_img, cv2.COLOR_BGR2RGB)
    return overlayed_img


# ========== 使用示例 ==========
# 读取原图（可以是RGB图像）
rgb = torch.load(r"E:\pycharm\project\work_dirs\visible\rgb.pt", weights_only=False)
ir_fuse = torch.load(r"E:\pycharm\project\work_dirs\visible\ir_fuses.pt", weights_only=False)
rgb_fuse = torch.load(r"E:\pycharm\project\work_dirs\visible\rgb_fuses.pt", weights_only=False)

img = rgb[0].permute(1, 2, 0).detach().cpu().numpy()  # (H,W,C)
img = (img - img.min()) / (img.max() - img.min())
img = np.uint8(img * 255)

# === 取出热力图特征 ===
rgb_feature = rgb_fuse[0][0, 15].detach().cpu().numpy()  # 取第16个通道
ir_feature = ir_fuse[0][0, 15].detach().cpu().numpy()  # 取第16个通道
add=rgb_feature+ir_feature
mul=rgb_feature*ir_feature
rgb_feature = np.uint8(255 * (rgb_feature - np.min(rgb_feature)) / (np.max(rgb_feature) - np.min(rgb_feature) + 1e-8))
ir_feature = np.uint8(255 * (ir_feature - np.min(ir_feature)) / (np.max(ir_feature) - np.min(ir_feature) + 1e-8))
add=np.uint8(255 * (add - np.min(add)) / (np.max(add) - np.min(add) + 1e-8))
mul=np.uint8(255 * (mul - np.min(mul)) / (np.max(mul) - np.min(mul) + 1e-8))
end= mul-add
end=np.uint8((end - np.min(end)) / (np.max(end) - np.min(end) + 1e-8)*255)
print(rgb_feature)

# === 显示各阶段结果 ===
cv2.namedWindow("rgb", cv2.WINDOW_NORMAL)
cv2.resizeWindow("rgb", 200, 200)
cv2.imshow("rgb", rgb_feature)

cv2.namedWindow("ir", cv2.WINDOW_NORMAL)
cv2.resizeWindow("ir", 200, 200)
cv2.imshow("ir", ir_feature)

cv2.namedWindow("add", cv2.WINDOW_NORMAL)
cv2.resizeWindow("add", 200, 200)
cv2.imshow("add", add)

cv2.namedWindow("mul", cv2.WINDOW_NORMAL)
cv2.resizeWindow("mul", 200, 200)
cv2.imshow("mul", mul)

cv2.namedWindow("end", cv2.WINDOW_NORMAL)
cv2.resizeWindow("end", 200, 200)
cv2.imshow("end", end)

cv2.namedWindow('rgb_hot', cv2.WINDOW_AUTOSIZE)
cv2.resizeWindow('rgb_hot', 200, 200)
rgb_hot=overlay_heatmap_on_image(img, rgb_feature)
cv2.imshow('rgb_hot', rgb_hot)

cv2.namedWindow('ir_hot', cv2.WINDOW_AUTOSIZE)
cv2.resizeWindow('ir_hot', 200, 200)
ir_hot=overlay_heatmap_on_image(img, ir_feature)
cv2.imshow('ir_hot', ir_hot)

cv2.namedWindow('add_hot', cv2.WINDOW_AUTOSIZE)
cv2.resizeWindow('add_hot', 200, 200)
add_hot=overlay_heatmap_on_image(img, add)
cv2.imshow('add_hot', add_hot)

cv2.namedWindow('mul_hot', cv2.WINDOW_AUTOSIZE)
cv2.resizeWindow('mul_hot', 200, 200)
mul_hot=overlay_heatmap_on_image(img, mul)
cv2.imshow('mul_hot', mul_hot)

cv2.namedWindow('end_hot', cv2.WINDOW_AUTOSIZE)
cv2.resizeWindow('end_hot', 200, 200)
end_hot=overlay_heatmap_on_image(img, end)
cv2.imshow('end_hot', end_hot)

# ===== 验证注意力强度 =====
# 取汽车区域与背景区域的像素值进行比较
h, w = rgb_feature.shape

# 你可以自己调整这几个点坐标（大致位于汽车与背景位置）
car_y, car_x = h // 2, w // 2           # 大致在图像中心（汽车位置）
bg_y, bg_x = int(h * 0.1), int(w * 0.1) # 左上角背景位置
cv2.waitKey(0)
cv2.destroyAllWindows()
