import os
import shutil
import json
import torch
import argparse
from tqdm import tqdm
from diffusers import Transformer2DModel, PixArtSigmaPipeline
from transformers import CLIPImageProcessor, CLIPModel
from PIL import Image
import cv2
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description="Generate images from prompts using PixArtSigmaPipeline and compare using CLIP.")
parser.add_argument("--hf_endpoint", type=str, default="https://hf-mirror.com", help="Hugging Face endpoint")
parser.add_argument("--device", type=str, default="cuda:7", help="Device to use for computation")
parser.add_argument("--weight_dtype", type=str, default="float16", help="Data type for weights")
parser.add_argument("--json_path", type=str, required=True, help="Path to input JSONL file")
parser.add_argument("--output_dir", type=str, default="outputs/seed", help="Output directory")
parser.add_argument("--record_dir", type=str, default="", help="Output directory")
parser.add_argument("--start_line", type=int, default=1, help="Start line for processing")
parser.add_argument("--end_line", type=int, default=1000, help="End line for processing")
parser.add_argument("--seed", type=int, default=23, help="Random seed")
args = parser.parse_args()

os.environ['HF_ENDPOINT'] = args.hf_endpoint

device = torch.device(args.device if torch.cuda.is_available() else "cpu")
weight_dtype = getattr(torch, args.weight_dtype)

transformer = Transformer2DModel.from_pretrained(
    "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS", 
    subfolder='transformer', 
    torch_dtype=weight_dtype,
    use_safetensors=True,
    safety_checker=None,
)
pipe = PixArtSigmaPipeline.from_pretrained(
    "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers",
    transformer=transformer,
    torch_dtype=weight_dtype,
    use_safetensors=True,
    safety_checker=None,
)
pipe.to(device)

name = os.path.basename(args.json_path).split(".")[0]
output_path = os.path.join(args.output_dir, name)
os.makedirs(output_path, exist_ok=True)

with open(args.json_path, "r") as f:
    data = [json.loads(line) for line in f]
    # original_description is origin prompts
    origin_prompts = [d["original_description"] for d in data]
    # modified_description is modified prompts
    modified_prompts = [d["modified_description"] for d in data]
    base_path = "./dataset/coco"
    image_paths = [os.path.join(base_path, d["image"]) for d in data]

ori_dir = os.path.join(output_path, f"ori-{args.seed}")
os.makedirs(ori_dir, exist_ok=True)
mod_dir = os.path.join(output_path, f"mod-{args.seed}")
os.makedirs(mod_dir, exist_ok=True)

for i, (orig_prompt, mod_prompt, image_path) in tqdm(enumerate(zip(origin_prompts, modified_prompts, image_paths)), total=len(origin_prompts)):
    if i < args.start_line:
        continue
    if i > args.end_line:
        break

    try:
        with Image.open(image_path) as img:
            width, height = img.size
    except Exception as e:
        print(f"Error reading {image_path}: {e}")
        continue
    
    filename = os.path.basename(image_path)

    image = pipe(
        orig_prompt,
        width=512,
        height=512,
        generator=torch.Generator("cpu").manual_seed(args.seed),
    ).images[0]
    image.save(os.path.join(ori_dir, filename))

    image = pipe(
        mod_prompt,
        width=512,
        height=512,
        generator=torch.Generator("cpu").manual_seed(args.seed),
    ).images[0]
    image.save(os.path.join(mod_dir, filename))
    
    
model_ID = "openai/clip-vit-base-patch16"
clip_model = CLIPModel.from_pretrained(model_ID).to(device)
preprocess = CLIPImageProcessor.from_pretrained(model_ID)

def load_and_preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = preprocess(image, return_tensors="pt")["pixel_values"]
    return image

def clip_img_score(img1_path, img2_path):
    try:
        image_a = load_and_preprocess_image(img1_path).to(device)
        image_b = load_and_preprocess_image(img2_path).to(device)
        with torch.no_grad():
            embedding_a = clip_model.get_image_features(image_a)
            embedding_b = clip_model.get_image_features(image_b)
        similarity_score = torch.nn.functional.cosine_similarity(embedding_a, embedding_b)
        return similarity_score.item()
    except Exception as e:
        print(f"Error comparing {img1_path} and {img2_path}: {e}")
        return None

coco2017_train_dir = "./dataset/coco"
original_dir = f"{output_path}/ori-{args.seed}"
modified_dir = f"{output_path}/mod-{args.seed}"
original_scores = []
modified_scores = []

diff_results = []

with open(args.json_path, "r") as f:
    data = [json.loads(line) for line in f]
    for i, d in tqdm(enumerate(data), total=len(data)):
        if i < args.start_line:
            continue
        if i > args.end_line:
            break
        image_file_name = d["image"]
        compare_img = os.path.join(coco2017_train_dir, image_file_name)
        original_img = os.path.join(original_dir, image_file_name)
        modified_img = os.path.join(modified_dir, image_file_name)
        original_score = clip_img_score(compare_img, original_img)
        modified_score = clip_img_score(compare_img, modified_img)
        if original_score is not None:
            original_scores.append(original_score)
        if modified_score is not None:
            modified_scores.append(modified_score)
        if original_score is not None and modified_score is not None:
            if original_score > modified_score:
                print("skip")
                continue
            
            score_difference = modified_score - original_score
            
            diff_results.append({
                "image": image_file_name,
                "original_score": original_score,
                "modified_score": modified_score,
                "score_difference": score_difference
            })


if original_scores and modified_scores:
    average_original_score = sum(original_scores) / len(original_scores)
    average_modified_score = sum(modified_scores) / len(modified_scores)
    inc = average_modified_score - average_original_score

    result_str = (
        f"{name}\n"
        f"seed {args.seed}\n"
        f"count {len(original_scores)}\n"
        f"Average original score: {average_original_score}\n"
        f"Average modified score: {average_modified_score}\n"
        f"Increase: {inc}\n"
    )

    
    output_txt_path = os.path.join(output_path, f"results_seed_{args.seed}.txt")
    with open(output_txt_path, "w") as result_file:
        result_file.write(result_str)

    print(result_str)

else:
    print("No valid scores to compute.")


diff_bench_file_path = args.record_dir
with open(diff_bench_file_path, "w") as diff_file:
    for result in diff_results:
        diff_file.write(json.dumps(result) + "\n")

print("Results recorded in", diff_bench_file_path)
