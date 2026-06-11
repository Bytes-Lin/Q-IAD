import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import re
import time
import math
import json
import random

import torch
from PIL import Image, ImageDraw, ImageFont
from peft import PeftConfig, PeftModel
from tqdm import tqdm
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLProcessor

from models.qwen25_vl import NewQwen


def smart_resize(
        height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 12845056
        # 14 * 14 * 4 * 1280
):
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def resize_bbox(bbox, new_h, new_w, orig_h, orig_w):
    """
    对单个 bbox 进行缩放。

    参数：
        bbox: List[int] [x1, y1, x2, y2]
        orig_h, orig_w: 原图尺寸
        new_h, new_w: 缩放后的尺寸

    返回：
        缩放后的 bbox: [x1', y1', x2', y2']
    """
    scale_w = new_w / orig_w
    scale_h = new_h / orig_h

    x1, y1, x2, y2 = bbox
    x1_new = max(0, min(round(x1 * scale_w), new_w))
    y1_new = max(0, min(round(y1 * scale_h), new_h))
    x2_new = max(0, min(round(x2 * scale_w), new_w))
    y2_new = max(0, min(round(y2 * scale_h), new_h))

    return [x1_new, y1_new, x2_new, y2_new]


def extract_all_predictions(content):
    """
    从模型输出中提取所有预测框和标签。
    期望输出类似:
    [{"bbox_2d": [x1, y1, x2, y2], "label": "defect"}, {...}]
    """
    # answer_tag_pattern = r'<answer>(.*?)</answer>'
    json_pattern = r'```json(.*?)```'

    content_answer_match = re.search(json_pattern, content, re.DOTALL)
    if not content_answer_match:
        return []

    content_answer = content_answer_match.group(1).strip()

    try:
        preds = json.loads(content_answer)
        if isinstance(preds, list):
            return preds
    except Exception:
        pass

    # 如果不是严格 JSON，用正则提取 bbox
    bbox_label_pattern = r'\{\s*"bbox_2d"\s*:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\],\s*"label"\s*:\s*"([^"]+)"\s*\}'
    matches = re.findall(bbox_label_pattern, content_answer)

    results = []
    for m in matches:
        x1, y1, x2, y2, label = m
        results.append({
            "bbox_2d": [int(x1), int(y1), int(x2), int(y2)],
            "label": label
        })
    return results


def load_model(model_id, model_path, lora_model_path, device_map, is_lora: bool = False):
    model = NewQwen.from_pretrained(model_path,
                                    torch_dtype=torch.bfloat16,
                                    device_map=device_map).eval()
    processor = Qwen2_5_VLProcessor.from_pretrained(pretrained_model_name_or_path=model_id, use_fast=True)
    processor.tokenizer.padding_side = "left"

    weights = {k: v.to(torch.float).clone() for k, v in model.state_dict().items() if 'text_projection' in k}

    if is_lora:
        config = PeftConfig.from_pretrained(lora_model_path)
        print(config)
        model = PeftModel.from_pretrained(model, model_id=lora_model_path, config=config)
        model = model.merge_and_unload()  # lora参数量合并

    model.load_state_dict(torch.load(f'{lora_model_path}/text_projection.bin', map_location=device_map),
                          strict=False)
    model.load_state_dict(torch.load(f'{lora_model_path}/merger.bin', map_location=device_map),
                          strict=False)
    model.load_state_dict(torch.load(f'{lora_model_path}/vision_merge.bin', map_location=device_map),
                          strict=False)
    weights_ = {k: v.to(torch.float) for k, v in model.state_dict().items() if 'text_projection' in k}
    for key in weights:
        print(f'{key}, {weights[key].equal(weights_[key])}')

    print(model)
    print(f'parameters: {sum(param.numel() for _, param in model.named_parameters())}')
    return model, processor


def eval_and_save_coco(
        model_id, model_path, lora_model_path, test_datasets, data_root, image_root, question_template,
        output_coco_pred, gt_coco_file, device_map, is_lora, batch_size=1, sample_num=10000, seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    model, processor = load_model(model_id, model_path, lora_model_path, device_map, is_lora=is_lora)

    # 读取 GT COCO 文件，建立映射
    with open(gt_coco_file, 'r', encoding='utf-8') as f:
        coco_gt = json.load(f)
    image_to_id = {img["file_name"]: img["id"] for img in coco_gt["images"]}
    category_to_id = {cat["name"]: cat["id"] for cat in coco_gt["categories"]}

    coco_predictions = []

    for ds in test_datasets:
        print(f"Processing {ds}...")

        ds_path = os.path.join(data_root, f"{ds}.jsonl")
        with open(ds_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f if line.strip()]
        random.shuffle(data)
        data = data[:sample_num]
        messages = []

        for x in data:
            image_path = os.path.join(image_root, x['image'])

            messages.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_path},
                            {"type": "text", "text": question_template.format(Question=x['conversations'][0]['value'])}
                        ]
                    }
                ]
            )

        all_outputs = []
        for i in tqdm(range(0, len(messages), batch_size)):
            batch_messages = messages[i:i + batch_size]
            text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in
                    batch_messages]
            image_inputs, video_inputs = process_vision_info(batch_messages)
            inputs = processor(
                text=text, images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt"
            ).to(device_map)

            generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=128, temperature=0.2)
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            batch_output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
            print(batch_output_text)
            all_outputs.extend(batch_output_text)

        for input_example, model_output in zip(data, all_outputs):
            predictions = extract_all_predictions(model_output)
            print(predictions)
            image_file = input_example["image"]
            image_path = os.path.join(image_root, input_example['image'])
            img = Image.open(image_path).convert('RGB')
            orig_w, orig_h = img.size
            new_h, new_w = smart_resize(orig_h, orig_w)

            if image_file not in image_to_id:
                print(f"[WARN] {image_file} not found in GT, skipping")
                continue
            image_id = image_to_id[image_file]

            gt_boxes, gt_labels = [i['bbox_2d'] for i in input_example['conversations'][1]['value']], [i['label'] for i in input_example['conversations'][1]['value']]

            for pred in predictions:
                try:
                    bbox = pred["bbox_2d"]
                    bbox_abs_resized = resize_bbox(bbox, orig_h, orig_w, new_h, new_w)
                except:
                    print(image_file)
                    continue
                x_min, y_min, x_max, y_max = bbox_abs_resized
                width, height = x_max - x_min, y_max - y_min

                try:
                    label = pred["label"]
                except:
                    print(image_file)
                    continue

                if label not in category_to_id:
                    print(f"[WARN] Unknown label {label}, skipping")
                    continue
                category_id = category_to_id[label]
                score = pred.get("score", 1.0)

                coco_predictions.append({
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x_min, y_min, width, height],
                    "score": score
                })

            draw = ImageDraw.Draw(img)
            for gt_box, gt_label in zip(gt_boxes, gt_labels):
                draw.rectangle(gt_box, outline='green', width=3)
                size = int(0.1 * int(math.sqrt((gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1]))))
                size = 15 if size < 15 else size
                font = ImageFont.truetype(font='Ubuntu-C.ttf', size=size)
                draw.text((gt_box[0] + size // 3, gt_box[1] + size // 3), text=gt_label, fill='green', font=font)
            try:
                for pred in predictions:
                    bbox, label = pred["bbox_2d"], pred["label"]
                    draw.rectangle(bbox, outline='red', width=3)
                    size = int(0.1 * int(math.sqrt((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))))
                    size = 15 if size < 15 else size
                    font = ImageFont.truetype(font='Ubuntu-C.ttf', size=size)
                    draw.text((bbox[0] + size // 3, bbox[1] + size // 3), text=pred["label"], fill='red', font=font)
            except:
                pass
            img.save(os.path.join('ia3_quaternion_3b', image_file.split('/')[-1]))

    # 保存 COCO 格式预测
    os.makedirs(os.path.dirname(output_coco_pred), exist_ok=True)
    with open(output_coco_pred, 'w', encoding='utf-8') as f:
        json.dump(coco_predictions, f, ensure_ascii=False, indent=4)

    print(f"✅ 预测结果已保存为 COCO 格式：{output_coco_pred}")


if __name__ == '__main__':
    model_id = 'Qwen/Qwen2.5-VL-3B-Instruct'
    model_path = './model/quaternion_new/inside_5_4_1_1000'
    lora_model_path = './model/quaternion_new/checkpoints_mv/SFT_model_quaternion_v15/model'

    data_root = './code'
    test_datasets = ['config/mvtec_ad_test_v4']  # data_root + path
    image_root = './dataset/mvtec_ad'
    gt_coco_file = './code/config/COCO_format_mvtec_test_v4.json'
    output_coco_pred = 'predict/predict.json'
    is_lora = True
    device_map = 'cuda:0'
    question_template = ("{Question}")

    start = time.time()
    eval_and_save_coco(
        model_id=model_id,
        model_path=model_path,
        lora_model_path=lora_model_path,
        data_root=data_root,
        test_datasets=test_datasets,
        image_root=image_root,
        question_template=question_template,
        output_coco_pred=output_coco_pred,
        gt_coco_file=gt_coco_file,
        device_map=device_map,
        batch_size=1,
        is_lora=is_lora
    )
    end = time.time()
    print(f'time cost: {int(end - start)}s')
