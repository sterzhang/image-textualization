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
    with open(args.input_file_path, 'r') as input_file:
        with open(args.output_file_path, 'a') as output_file:
            for i, line in enumerate(tqdm(input_file, total=total_lines, desc="Filter hallucination from the extraction objects")):
                if i < args.start_line:
                    continue  # Skip lines until the start_line is reached
                if i >= args.end_line:
                    break
                json_obj = json.loads(line)
                img = json_obj.get("image")
                objs = json_obj.get("objects") 
                bounding_boxes = json_obj.get("bounding_boxes")

                exist_obj_idx = []
                for idx, obj in enumerate(objs):
                    # img_path = f"./dataset/{img}"
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

                    if phrases:
                        exist_obj_idx.append(idx)

                exist_objs = [objs[idx] for idx in exist_obj_idx]
                exist_bounding_boxes = [bounding_boxes[idx] for idx in exist_obj_idx] if boxes is not None else None

                after_filter_line = {
                    "image": img,
                    "exist_obj_from_img": exist_objs,
                    "bounding_boxes": exist_bounding_boxes,
                    "description": desc
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
    parser.add_argument("--end_line", type=int)
    args = parser.parse_args()

    main(args)

        