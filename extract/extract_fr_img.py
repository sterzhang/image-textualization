import argparse
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import sys
import json

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

sys.path.insert(0, 'extract/third_party/CenterNet2/projects/CenterNet2/')
from centernet.config import add_centernet_config
from grit.config import add_grit_config

from grit.predictor import VisualizationDemo
import warnings

warnings.filterwarnings("ignore") 
WINDOW_NAME = "Extract objects from image"

"""
input_file ("image")
output_file ("image", "objects", "bounding_boxes")
gpu 12000M
"""


def setup_cfg(args):
    cfg = get_cfg()
    if args.cpu:
        cfg.MODEL.DEVICE="cpu"
    add_centernet_config(cfg)
    add_grit_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    if args.test_task:
        cfg.MODEL.TEST_TASK = args.test_task
    cfg.MODEL.BEAM_SIZE = 1
    cfg.MODEL.ROI_HEADS.SOFT_NMS_ENABLED = False
    cfg.USE_ACT_CHECKPOINT = False
    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config_file",
        default="./extract/configs/GRiT_B_DenseCap_ObjectDet.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--cpu", action='store_true', help="Use CPU only.")
    parser.add_argument(
        "--image_folder",
        type=str,
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--visualize_output",
        type=str,
        default="false",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--test_task",
        type=str,
        default='DenseCap',
        help="Choose a task to have GRiT perform",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--input_file",
        type=str,
    )
    parser.add_argument(
        "--output_file",
        type=str,
    )
    parser.add_argument(
        "--start_line",
        type=int,
    )
    parser.add_argument(
        "--end_line",
        type=int,
    )
    return parser

original_stdout = sys.stdout
sys.stdout = open(os.devnull, 'w')

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)
    sys.stdout = original_stdout
    input_file = args.input_file
    output_file = args.output_file
    
    output_folder = os.path.dirname(output_file)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    total_lines = args.end_line - args.start_line
    if args.image_folder:
        with open(output_file, 'a') as f:
            with open(input_file, 'r') as json_file:
                for i, line in enumerate(tqdm.tqdm(json_file, total=total_lines, desc="Extract objects from the image")):
                    if i < args.start_line:
                        continue
                    if i >= args.end_line:
                        break
                    data = json.loads(line)
                    image_path = data['image']
                    img = read_image(os.path.join(args.image_folder, image_path), format="BGR")
                    predictions, visualized_output = demo.run_on_image(img)

                    # bounding box in image -> details in text format
                    boxes = predictions["instances"].pred_boxes if predictions["instances"].has("pred_boxes") else None
                    object_description = predictions["instances"].pred_object_descriptions.data
                    # create lists to store object descriptions and bounding boxes
                    object_list = []
                    bounding_box_list = []

                    for i in range(len(object_description)):
                        object_list.append(object_description[i])
                        box = [int(a) for a in boxes[i].tensor.cpu().detach().numpy()[0]]
                        bounding_box_list.append(box)

                    # construct data dictionary
                    data = {'image': image_path, 'objects': object_list, 'bounding_boxes': bounding_box_list}
                    f.write(json.dumps(data) + '\n')

                    if args.visualize_output:
                        if args.visualize_output.lower() == 'false':
                            continue  # Skip saving visualizations if output is set to false

                        if not os.path.exists(args.visualize_output):
                            os.mkdir(args.visualize_output)
                        if os.path.isdir(args.visualize_output):
                            assert os.path.isdir(args.visualize_output), args.visualize_output
                            out_filename = os.path.join(args.visualize_output, image_path)
                        visualized_output.save(out_filename)
