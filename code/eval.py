from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

gt_path = "./code/config/COCO_format_v3_test.json"
pred_path = "./code/predict/predict_3cad_lora.json"

coco_gt = COCO(gt_path)
coco_dt = coco_gt.loadRes(pred_path)

cats = {}
for i in coco_gt.cats.values():
    cats[i['id']] = i['name']

print("GT annos:", len(coco_gt.getAnnIds()))
print("Pred annos:", len(coco_dt.getAnnIds()))
print('.' * 100)

coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
import numpy as np

print(coco_eval.params.iouThrs)
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

# 获取每个类别的 AP（在类别数量与 coco_gt.getCatIds() 一一对应）
precision_per_category = coco_eval.eval['precision']
average_precisions = {}

# 在加载的 JSON 文件的 categories 字段中，包含了数据集中定义的所有类别。
for i, cat_id in enumerate(coco_gt.getCatIds()):
    ap = precision_per_category[0, :, i, 0, -1]  # ap50
    average_precisions[cat_id] = ap.mean()

# 输出每个类别上的AP
sum_ap = 0
for cat_id, ap_value in average_precisions.items():
    print(f"Category ID {cat_id}, {cats[cat_id]}: Average Precision = {ap_value:.3f}")
    sum_ap += ap_value
print(f'@[IOU=0.5] mAP: {(sum_ap / len(coco_gt.getCatIds())):.3f}')
