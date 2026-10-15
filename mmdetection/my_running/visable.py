import json
import os
import cv2
import torch


def save_image(image, save_path):
    """保存图像，并兼容 Windows 中文路径。"""
    extension = os.path.splitext(save_path)[1] or '.png'
    success, encoded_image = cv2.imencode(extension, image)
    if not success:
        raise RuntimeError(f'图像编码失败: {save_path}')

    encoded_image.tofile(save_path)
    if not os.path.isfile(save_path):
        raise RuntimeError(f'图像保存失败: {save_path}')

gt_json = r"D:\data\M3FD\cocodataset\annotations\test.json"
dp_json = r"D:\pycharm work\weight\test.bbox (1).json"
image_path = r"D:\data\M3FD\cocodataset\images\visible"
output_path = r"D:\毕业论文\常用图片"
"""
读取图片
329:黑暗环境
364：雾天
394：低质量图像
429：高曝光汽车
714：雾天
789：雾天+ 小目标
1344: 超小目标
1434:雾天
1554：白天完全被遮盖
1874:超小目标
1959:小目标+多目标
"""
id = 1959
picture_id = id
picture_id = f'{picture_id:05d}.png'
picture_id = os.path.join(image_path, picture_id)
picture1 = cv2.imread(picture_id)
picture2 = cv2.imread(picture_id)

#打开目标标注文件，找到对应的标框
gt_json = json.load(open(gt_json))
annotations = gt_json['annotations']
switch = False
gt_bboxes = []
for annotation in annotations:
    if annotation['image_id'] == id:
        switch = True
        gt_bboxes.append(annotation['bbox'])
    if annotation['image_id'] != id and switch:
        break

gt_length = len(gt_bboxes)
for bbox in gt_bboxes:
    x, y, w, h = map(int, bbox)
    cv2.rectangle(picture1, (x, y), (x + w, y + h), (0, 255, 0), 2)

#打开预测标注文件，找到对应的标框
dp_json = json.load(open(dp_json))
switch = False
dp_bboxes = []
for annotation in dp_json:
    if annotation['image_id'] == id:
        switch = True
        dp_bboxes.append([annotation['bbox'],annotation['score']])
    if annotation['image_id'] != id and switch:
        break

dp_bboxes = sorted(dp_bboxes, key=lambda x: x[1], reverse=True)
dp_bboxes = dp_bboxes[:gt_length]
for bbox_info in dp_bboxes:
    bbox, score = bbox_info
    x, y, w, h = map(int, bbox)
    cv2.rectangle(picture2, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # 防止置信度文字画出图片上边界
    text_y = max(10, y - 5)
    cv2.putText(picture2, f'{score:.2f}', (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

os.makedirs(output_path, exist_ok=True)
gt_save_path = os.path.join(output_path, f'{id:05d}_gt.png')
pred_save_path = os.path.join(output_path, f'{id:05d}_pred.png')

save_image(picture1, gt_save_path)
save_image(picture2, pred_save_path)
print(f'真实标注图已保存: {gt_save_path}')
print(f'预测结果图已保存: {pred_save_path}')

cv2.imshow('picture', picture1)
cv2.imshow("image", picture2)
cv2.waitKey(0)

