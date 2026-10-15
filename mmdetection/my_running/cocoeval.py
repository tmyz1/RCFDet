from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import sys
#save_path = r"E:\pycharm\project\Swin-DINO rar\all\0.872.txt"
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


#sys.stdout = Logger(save_path)

cocoGt = COCO(r"D:\data\M3FD\cocodataset\annotations\test.json")
print(cocoGt)
cocoDt = cocoGt.loadRes(
    r"D:\pycharm work\weight\test.bbox.json")

cocoEval = COCOeval(cocoGt, cocoDt, iouType="bbox")
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

# 获取所有类别
cat_ids = cocoGt.getCatIds()
cat_names = [cocoGt.loadCats([cid])[0]["name"] for cid in cat_ids]

print("=" * 80)
print("COCO 分类别检测评估结果")
print("=" * 80)

# 分类别评估
for cid, cname in zip(cat_ids, cat_names):
    cocoEval = COCOeval(cocoGt, cocoDt, iouType="bbox")
    cocoEval.params.catIds = [cid]  # 只评估某一类
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print(f"\nCategory: {cname}")
    print(f"mAP@[0.5:0.95]: {cocoEval.stats[0]:.4f}")
    print(f"mAP@0.5:       {cocoEval.stats[1]:.4f}")
    print(f"mAP@0.75:      {cocoEval.stats[2]:.4f}")
    print("-" * 40)
#print("\n所有类别评估完毕结果已保存至：", save_path)
