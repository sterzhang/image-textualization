from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import json
from tqdm import tqdm
import warnings
import argparse
import sys
import os

warnings.filterwarnings("ignore")  

original_stdout = sys.stdout
sys.stdout = open(os.devnull, 'w')

def main(args):
    model = load_model(args.model_config, args.model_checkpoint)
    BOX_THRESHOLD = args.box_threshold
    TEXT_THRESHOLD = args.text_threshold
    sys.stdout = original_stdout
    total_lines = args.end_line - args.start_line

    with open(args.input_file, 'r') as input_file:
        with open(args.output_file, 'a') as output_file:
            for i, line in enumerate(tqdm(input_file, total=total_lines, desc="Filter hallucination objects from the description")):
                if i < args.start_line:
                    continue  # Skip lines until the start_line is reached
                if i >= args.end_line:
                    break
                json_obj = json.loads(line)
                img = json_obj.get("image")
                # desc = json_obj.get("description")
                extr_obj = json_obj.get("extr_obj_fr_desc") 

                des_exist_obj_idx = []
                for idx, obj in enumerate(extr_obj):
                    img_path = os.path.join(args.image_folder, img)
                    image_source, image = load_image(img_path)
                    
                    boxes, logits, phrases = predict(
                        model=model,
                        image=image,
                        caption=obj,
                        box_threshold=BOX_THRESHOLD,
                        text_threshold=TEXT_THRESHOLD
                    )

                    # tmp visualize
                    # annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
                    # cv2.imwrite("./annotated_image.jpg", annotated_frame)

                    if phrases == []:
                        continue
                    else:
                        des_exist_obj_idx.append(idx)

                del_obj = [extr_obj[idx] for idx in range(len(extr_obj)) if idx not in des_exist_obj_idx]

                after_filter_line = {
                    "image": img,
                    "del_obj_from_desc": del_obj,
                }
                output_file.write(json.dumps(after_filter_line) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter hallucination objects from the image")
    parser.add_argument("--model_config", type=str, default="./stage2/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py", help="Path to model config file")
    parser.add_argument("--model_checkpoint", type=str, default="./ckpt/groundingdino_swinb_ogc.pth", help="Path to model checkpoint file")
    parser.add_argument("--box_threshold", type=float, default=0.40, help="Box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="Text threshold")
    parser.add_argument("--image_folder", type=str, default="")
    parser.add_argument("--input_file", type=str, default="./annotation/stage1/obj_extr_from_img.jsonl", help="Path to input file")
    parser.add_argument("--output_file", type=str, default="./annotation/stage2/fil_obj_from_img.jsonl", help="Path to output file")
    parser.add_argument("--start_line", type=int, default=0, help="Start reading the input file from this line number (if fail in line x, then set this value as x to continue generate).")
    parser.add_argument("--end_line", type=int, default=999999999)
    args = parser.parse_args()

    main(args)
"""
CUDA_VISIBLE_DEVICES=5 python filter/filter_fr_desc.py     --model_config ./filter/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py     --model_checkpoint ./ckpt/groundingdino_swinb_ogc.pth     --box_threshold 0.30     --text_threshold 0.15     --input_file /home/zhangjianshu/Mercury/jsonl/fine_grained_annotation/extract_obj_gpt4_5w.jsonl     --output_file /home/zhangjianshu/Mercury/jsonl/fine_grained_annotation/hal_obj_gpt4_5w.jsonl     --image_folder /home/zhangjianshu/LLaVA/dataset/sharegpt4v/images/coco/train2017     --start_line 43353  --end_line 49715
"""