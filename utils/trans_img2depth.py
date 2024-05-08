from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image
import torch
import numpy as np
import json
from tqdm import tqdm
import os
import warnings
import argparse

warnings.filterwarnings("ignore")  


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute depth of focus')
    parser.add_argument('--input_file', type=str, help='Path to the input JSONL file')
    parser.add_argument('--output_folder', type=str, help='Path to the output folder')
    parser.add_argument('--image_folder', type=str, help='Path to the folder containing images')
    parser.add_argument('--start_line', type=int, default=0, help='Start line in the input file')
    parser.add_argument('--end_line', type=int, default=0, help='End line in the input file')
    args = parser.parse_args()

    input_file = args.input_file
    output_folder = args.output_folder
    image_folder = args.image_folder

    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder)

    with open(input_file, 'r') as f:
        total_lines = sum(1 for _ in f)

    with open(input_file, 'r') as jsonl_file:
        for i, line in enumerate(tqdm(jsonl_file, total=total_lines, desc="get the depth of focus for objects")):
            if i < args.start_line:
                continue
            
            if i >= args.end_line:
                break
            data = json.loads(line)
            image_name = data.get('image', '')
            output_path = os.path.join(output_folder, image_name)
            # output_directory = os.path.dirname(output_path)
            if os.path.exists(output_path):
                print(f"Skip {output_path}")
                continue

            if image_name:
                image_path = os.path.join(image_folder, image_name)
                image = Image.open(image_path) 
                image_array = np.array(image)

                # ignore ndim = 2
                if image_array.ndim == 2:               
                    continue

                checkpoint = "vinvino02/glpn-nyu"

                image_processor = AutoImageProcessor.from_pretrained(checkpoint)
                model = AutoModelForDepthEstimation.from_pretrained(checkpoint)
                # image_processor = AutoImageProcessor.from_pretrained(checkpoint, force_download=True,resume_download=False)
                # model = AutoModelForDepthEstimation.from_pretrained(checkpoint,  force_download=True, resume_download=False)
                pixel_values = image_processor(image, return_tensors="pt").pixel_values

                with torch.no_grad():
                    outputs = model(pixel_values)
                    predicted_depth = outputs.predicted_depth  

                prediction = torch.nn.functional.interpolate( 
                    predicted_depth.unsqueeze(1),
                    size=image.size[::-1],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

                output = prediction.numpy() 
                formatted = (output * 255 / np.max(output)).astype("uint8") 
                depth = Image.fromarray(formatted) 

                output_path = os.path.join(output_folder, image_name)
                output_directory = os.path.dirname(output_path)
                if not os.path.exists(output_directory):
                    os.makedirs(output_directory)
                depth.save(output_path)
