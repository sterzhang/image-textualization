import copy
import cv2
from segment_anything import SamPredictor
import torch
from segment_anything import sam_model_registry
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pycocotools import mask as maskUtils
import json
from tqdm import tqdm
import multiprocessing
from collections import OrderedDict
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image
import torch
import numpy as np
import json
from tqdm import tqdm
import argparse

MODEL_TYPE = "vit_h"
sam = sam_model_registry[MODEL_TYPE](checkpoint="./ckpt/sam_vit_h_4b8939.pth")
sam.cuda()
mask_predictor = SamPredictor(sam)

def binary_mask_to_polygon(binary_mask):
    contours, _ = cv2.findContours(
        binary_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    segmentation = []
    for contour in contours:
        contour = contour.flatten().tolist()
        # Skip invalid polygons (less than 6 points)
        if len(contour) < 6:
            continue
        segmentation.append(contour)
    
    return segmentation

def polygons_to_binary_mask(polygons, image_shape):
    mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.fillPoly(mask, polygons, color=255)
    return mask

def plot_images(img, mask, box, mask_polygon, save_path):
    img = np.array(img)
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    w_ori, h_ori, _ = img.shape
    mask = mask > 0.5
    masked_image = img.copy()
    masked_image = np.where(mask.astype(int)[..., None],
                            np.array([0, 255, 0], dtype='uint8'),
                            masked_image)
    masked_image = cv2.addWeighted(img, 0.3, masked_image, 0.7, 0)
    plt.imshow(masked_image)
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                             edgecolor='r', facecolor='none')
    plt.gca().add_patch(rect)
    for poly in mask_polygon:
        poly = patches.Polygon(poly, closed=True, fill=False, edgecolor='b')
        plt.gca().add_patch(poly)
    plt.savefig(save_path)  
    plt.close()

def segment(predictor, boxes):
    resulting_masks = []
    for box in boxes:
        masks, scores, logits = predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        resulting_masks.append(masks[index])
    return resulting_masks

def calculate_partition_indices(data, num_partitions):
    partition_size = len(data) // num_partitions
    remainder = len(data) % num_partitions

    partition_indices = [(i * partition_size + min(i, remainder), (i + 1) * partition_size + min(i + 1, remainder)) for i in range(num_partitions)]
    return partition_indices

def get_data_subset(data, partition_indices, partition_number):
    start, end = partition_indices[partition_number]
    return data[start:end]

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Segment and analyze images')
    parser.add_argument('--input_path', type=str, help='Path to the input annotation file')
    parser.add_argument('--output_path', type=str, help='Path to save the output annotation file')
    parser.add_argument('--start_line', type=int, help='Path to save the output annotation file')
    parser.add_argument('--end_line', type=int, default=9999999, help='Path to save the output annotation file')
    parser.add_argument('--image_folder', type=str, help='Path to the folder containing images')
    parser.add_argument('--image_depth_folder', type=str, help='Path to the folder containing depth images')
    args = parser.parse_args()

    file_path_in = args.input_path
    file_path_out = args.output_path
    image_folder = args.image_folder
    image_depth_folder = args.image_depth_folder
    os.makedirs(os.path.dirname(file_path_out), exist_ok=True)

    total_lines = sum(1 for _ in open(file_path_in))
    with open(file_path_in, "r") as f_in, open(file_path_out, "w") as f_out:
        for i, line in enumerate(tqdm(f_in, total=total_lines, desc="fine-grained mask")):
            if i < args.start_line:
                continue
            if i >= args.end_line:
                break
            data = json.loads(line)
            image = data.get('image', '')
            bounding_boxes = data.get('bounding_boxes', '')
            objects = data.get('exist_obj_from_img', '')
            # objects = data.get('objects', '')

            # del_obj_from_desc = data.get('del_obj_from_desc')

            image_path = os.path.join(image_folder, image) 

            image_check = Image.open(image_path) 
            image_array_check = np.array(image_check)
            # ignore ndim = 2
            if image_array_check.ndim == 2:               
                continue    

            image_bgr = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            mask_predictor.set_image(image_rgb)
            boxes = np.array(bounding_boxes)
            masks = segment(mask_predictor, boxes) 
            polygons = [binary_mask_to_polygon(mask) for mask in masks]
            polygons_reshaped = [[np.array(poly).reshape(-1, 2) for poly in polygon] for polygon in polygons]

            # visualize
            # for mask, box, polygon in zip(masks, boxes, polygons_reshaped):
            #     plot_images(image_rgb, mask, box, polygon, "./sam_show.png")

            # depth information
            # image_depth_path = f"./stage3/img_depth/{os.path.basename(image)}"
            image_depth_path = os.path.join(image_depth_folder, image)
            image_depth = Image.open(image_depth_path)
            width, height = image_depth.size
            image_depth_array = np.array(image_depth)

            image_depth_array_processed = 255 - image_depth_array

            box_depth_score = []
            box_size = []
            for polygons in polygons_reshaped:
                polygon_area = np.zeros((height, width), dtype=np.uint8)
                for polygon in polygons:
                    pts = np.array(polygon, np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.fillPoly(polygon_area, [pts], 1)

                sam_object_size = int(np.sum(polygon_area != 0))
                depth_map = polygon_area.astype(np.float32) * image_depth_array_processed
                
                depth_score = np.sum(depth_map)
                valid_depth_pixels = np.sum(depth_map != 0)
                
                if valid_depth_pixels != 0:
                    normalized_depth_score = depth_score / valid_depth_pixels
                else:
                    normalized_depth_score = 0.0 
                
                box_depth_score.append(int(normalized_depth_score))
                box_size.append(sam_object_size)
                
            output_data = {
                'image': image,
                'objects': objects,
                'bounding_boxes': bounding_boxes,
                'object_depth': box_depth_score,
                'size': box_size,
                'width': width,
                'height': height
            }
            f_out.write(json.dumps(output_data, cls=NumpyEncoder) + '\n')
