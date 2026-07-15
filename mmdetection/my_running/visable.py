import json
import os
import cv2
gt_json = r"D:\data\M3FD\cocodataset\annotations\test.json"
dp_json = r"D:\pycharm work\weight\test.bbox (1).json"
image_path = r"D:\data\M3FD\cocodataset\images\visible"

#读取图片
id = 1229
picture_id = id
picture_id = f'{picture_id:05d}.png'
picture_id = os.path.join(image_path, picture_id)
picture = cv2.imread(picture_id)

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
    cv2.rectangle(picture, (x, y), (x + w, y + h), (0, 255, 0), 2)

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
    cv2.rectangle(picture, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # 防止置信度文字画出图片上边界
    text_y = max(10, y - 5)
    cv2.putText(picture, f'{score:.2f}', (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

cv2.imshow("image", picture)
cv2.waitKey(0)

