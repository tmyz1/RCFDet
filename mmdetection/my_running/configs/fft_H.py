import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt


def _to_uint8_image(img):
    arr = np.nan_to_num(np.asarray(img)).astype(np.float32)
    if arr.max() <= 1.5:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)

def _normalize_u8(x):
    x = np.asarray(x, dtype=np.float32)
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.uint8)
    return np.uint8((x - mn) / (mx - mn) * 255.0)
"""
高斯高通滤波器
"""
def fft_H(img,sigma= 1e-6,pad = None):
    low = cv2.GaussianBlur(img, (0, 0), 20)
    high = img.astype(np.float32) - low.astype(np.float32)

    return high
"""
平滑函数
"""
def smooth(img):
    denoised = cv2.GaussianBlur(
        img,
        ksize=(7, 7),
        sigmaX=1.2,
        sigmaY=1.2,
        borderType=cv2.BORDER_REFLECT_101
    )
    return denoised
"""
根据输入的图像显示对应的直方图
"""
def show_hist(img):
    hist = cv2.calcHist(
        [img.astype("float32")],  # 输入图像，必须用列表包装
        [0],  # 要统计的通道编号
        None,  # 掩膜，None 表示统计整张图
        [256],  # 直方图区间数量
        [float(img.min()), float(img.max())]  # 像素值范围
    )

    plt.plot(hist)
    plt.xlim([0, 256])
    plt.xlabel("Pixel value")
    plt.ylabel("Pixel count")
    plt.show()


def get_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--rgb_image', default=r"D:\data\M3FD\cocodataset\images\visible\00000.png")
    parser.add_argument('--ir_image', default=r"D:\data\M3FD\cocodataset\images\infrared\00000.png")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    rgb_img = cv2.imread(args.rgb_image)
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
    ir_img = cv2.imread(args.ir_image)
    ir_img = cv2.cvtColor(ir_img, cv2.COLOR_BGR2GRAY)
    rgb_fft = fft_H(rgb_img)
    ir_fft = fft_H(ir_img)
    rgb_smooth = smooth(rgb_fft)
    ir_smooth = smooth(ir_fft)
    cv2.imshow("rgb_smooth", rgb_smooth)
    cv2.imshow("ir_smooth", ir_smooth)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
