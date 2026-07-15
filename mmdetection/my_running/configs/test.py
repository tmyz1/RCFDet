import cv2
import numpy as np
import os

# 输入图片路径
input_path = r"C:\Users\tmyz1\Desktop\系统科学\小论文\RCFDet_Revision-1\RCFDet_Revision-1\pictures\图片1.png"

# 输出图片路径
output_path = r"C:\Users\tmyz1\Desktop\系统科学\小论文\RCFDet_Revision-1\RCFDet_Revision-1\pictures\图片1_noise.png"

# 读取图片（支持中文路径）
img = cv2.imdecode(
    np.fromfile(input_path, dtype=np.uint8),
    cv2.IMREAD_GRAYSCALE
)

# 添加轻微高斯噪声
noise = np.random.normal(0, 7, img.shape)

noisy_img = img.astype(np.float32) + noise
noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

# 保存图片（支持中文路径）
cv2.imencode('.png', noisy_img)[1].tofile(output_path)

# 显示
cv2.imshow("Noisy Image", noisy_img)

cv2.waitKey(0)
cv2.destroyAllWindows()

print("保存成功：", output_path)