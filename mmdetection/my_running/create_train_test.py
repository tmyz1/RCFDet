import json
import os

# ======== 配置路径 ========
src_ann = r"C:\Users\tmyz1\Desktop\M3FD\M3FD_Detection\coco_data_format.json"
save_dir = r"C:\Users\tmyz1\Desktop\M3FD\M3FD_Detection"

train_ann_path = os.path.join(save_dir, "train.json")
test_ann_path = os.path.join(save_dir, "test.json")

with open(src_ann, 'r', encoding='utf-8') as f:
    coco_data = json.load(f)

images = coco_data.get("images", [])
annotations = coco_data.get("annotations", [])
categories = coco_data.get("categories", [])
info = coco_data.get("info", {})
licenses = coco_data.get("licenses", [])

# ======== 按 4:1 周期划分（不打乱） ========
train_images = []
test_images = []

for i, img in enumerate(images):
    if (i % 5) == 4:
        test_images.append(img)   # 每第 5 张进 test
    else:
        train_images.append(img)  # 其他 4 张进 train

train_ids = {img["id"] for img in train_images}
test_ids = {img["id"] for img in test_images}

train_annotations = [ann for ann in annotations if ann["image_id"] in train_ids]
test_annotations = [ann for ann in annotations if ann["image_id"] in test_ids]

# ======== 构造新的 COCO 文件结构 ========
train_json = {
    "info": info,
    "licenses": licenses,
    "images": train_images,
    "annotations": train_annotations,
    "categories": categories
}
test_json = {
    "info": info,
    "licenses": licenses,
    "images": test_images,
    "annotations": test_annotations,
    "categories": categories
}

with open(train_ann_path, 'w', encoding='utf-8') as f:
    json.dump(train_json, f, ensure_ascii=False, indent=2)

with open(test_ann_path, 'w', encoding='utf-8') as f:
    json.dump(test_json, f, ensure_ascii=False, indent=2)

print("数据集按 4:1 周期划分完成（不打乱顺序）！")
print(f"训练集: {len(train_images)} 张图片, 标注 {len(train_annotations)} 条")
print(f"测试集: {len(test_images)} 张图片, 标注 {len(test_annotations)} 条")
print(f"train.json 已保存到: {train_ann_path}")
print(f"test.json 已保存到: {test_ann_path}")
