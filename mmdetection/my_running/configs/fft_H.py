import cv2
import numpy as np

def fft_H(img, sigma=10):
    """
    使用 Gaussian 高通滤波的 FFT 高频增强
    sigma 越大，高频越强
    """
    # 转 float32
    image_float32 = np.float32(img) / 255.0

    # 傅里叶变换
    dft = cv2.dft(image_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # 构造 Gaussian High-Pass mask
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2

    # 计算距离矩阵 D(u,v)
    u = np.arange(rows)
    v = np.arange(cols)
    V, U = np.meshgrid(v, u)
    D2 = (U - crow)**2 + (V - ccol)**2

    # Gaussian High-pass filter
    H = 1 - np.exp(-D2 / (2 * (sigma**2)))
    H = np.repeat(H[:, :, np.newaxis], 2, axis=2)  # 扩展到两个通道（实部/虚部）

    # 应用高通滤波器
    fshift = dft_shift * H

    # 反变换
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    # 归一化
    img_back_norm = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)

    return img_back_norm.astype(np.uint8)

def region_select_fft(rgb_fft, ir_fft, block=10):
    h, w = rgb_fft.shape
    add = np.zeros_like(rgb_fft, dtype=np.uint8)
    cnt1=0
    cnt2 =0

    for y in range(0, h, block):
        for x in range(0, w, block):

            y2 = min(y + block, h)
            x2 = min(x + block, w)

            # 当前10x10区域
            region_1 = rgb_fft[y:y2, x:x2]
            region_2 = ir_fft[y:y2, x:x2]
            mean_1 = np.mean(region_1)
            mean_2 = np.mean(region_2)


            if mean_1 > mean_2:
                # 使用 RGB 高频
                add[y:y2, x:x2] = rgb_fft[y:y2, x:x2]
                cnt1+=1
            else:
                # 使用 IR 高频
                add[y:y2, x:x2] = ir_fft[y:y2, x:x2]
                cnt2+=1

    return add,cnt1-cnt2

def get_score(img, number, block):
    H,W = img.shape
    sum = (H // block) * (W // block)
    num = abs(number)
    if number > 0:
        end = (num+sum)/(sum*2)
    else:
        end = num/(sum*2)
    return  end



# 读取图像(00788,00865,01298,02542,02683,03862,00325)
rgb = cv2.imread(r"D:\data\M3FD\cocodataset\images\visible\01298.png")
ir = cv2.imread(r"D:\data\M3FD\cocodataset\images\infrared\01298.png")
# 转灰度
rgb_gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
ir_gray = cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY)

# 调用高通函数
rgb_fft = fft_H(rgb_gray)
ir_fft = fft_H(ir_gray)
block = 1
add,count = region_select_fft(rgb_fft, ir_fft, block)
final= get_score(rgb_fft, count, block)
add3 = cv2.cvtColor(add, cv2.COLOR_GRAY2BGR)
end = cv2.addWeighted(rgb, 0.5,add3,0.5,1)
end_2 = cv2.addWeighted(ir,0.5,add3,0.5,1)
add_3 = cv2.add(rgb_fft,ir_fft)
end_3 = cv2.addWeighted(rgb, 0.5,add3,0.5,0)
end_4 = cv2.add(ir,add3)

# 显示
# cv2.imshow("rgb", rgb_fft)
# cv2.imshow("ir", ir_fft)
# cv2.imshow('end',end)
cv2.imshow('end_2',end_2)

cv2.waitKey(0)
