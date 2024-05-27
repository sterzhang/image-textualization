# Image Textualization: An Automatic Framework for Generating Rich and Detailed Image Descriptions
<img width="956" alt="image" src="https://github.com/sterzhang/image-textualization/assets/119802220/50dd94f2-ca9b-4aa8-a8b3-5de403111273">


🔥 The data can be downloaded directly from the `image-textualization-data/` folder.

## Contents
- [Install](#install)
- [Datasets](#datasets)
- [Use](#use)
- [Evaluation](#evaluation)
- [Visualization](#visualization)

## Install
See detailed instructions in [Install.md](https://github.com/sterzhang/image-textualization/blob/main/docs/install.md).

## Datasets
### Images
- COCO: Download here [train2017](http://images.cocodataset.org/zips/train2017.zip). 
- SAM: Click here [SAM](https://ai.meta.com/datasets/segment-anything-downloads/) and download sa_000000.tar ~ sa_000024.tar.

After downloading, organize the image datasets as follows in `./dataset/`:
```
├── coco
│   └── train2017
├── sam
    └── images
```
### Descriptions
We open-source 165k detailed descriptions in `image-textualization-data/`:
```
├── image-textualization-data/
    └── image-textualization-coco-50k-gpt4v.jsonl
    └── image-textualization-coco-50k-llava.jsonl
    └── image-textualization-sam-65k.jsonl
```
The format of our jsonl is below:
```json
{"image":"xxx.jpg", "question":"xxxx?", "description":"xxxxxx"}
```

## How To Use
### Extract Objects from Images
```bash
python extract/extract_fr_img.py \
    --test_task DenseCap \
    --config_file ./extract/configs/GRiT_B_DenseCap_ObjectDet.yaml \
    --confidence_threshold 0.55 \
    --image_folder  your-image-path/ \
    --input_file  image.jsonl \
    --output_file  obj_extr_from_img.jsonl \
    --start_line 0 \
    --end_line 999 \
    --visualize_output visualize-output-path \
    --opts MODEL.WEIGHTS ./ckpt/grit_b_densecap_objectdet.pth 
```
### Extract Objects from Descriptions
- chatgpt3.5 version
```bash
python extract/extract_fr_desc.py \
    --input_file_path description.jsonl \
    --output_file_path obj_extr_from_desc.jsonl \
    --api_key_path api_key.txt \
    --start_line 0
```
- llama version
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python ./extract/extract_fr_desc-llama.py \
    --input_file description.jsonl \
    --output_file obj_extr_from_desc.jsonl \
    --stop_tokens "<|eot_id|>" \
    --prompt_structure "<|begin_of_text|><|start_header_id|>user<|end_header_id|>{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>" \
    --start_line 0 \
    --end_line 999
```
### Filter Hallucinations of Obj_extr_from_desc
```bash
python filter/filter_fr_desc.py \
    --model_config ./filter/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py \
    --model_checkpoint ./ckpt/groundingdino_swinb_ogc.pth \
    --box_threshold 0.20 \   
    --text_threshold 0.18 \    
    --input_file obj_extr_from_desc.jsonl \
    --output_file hal_from_desc.jsonl \
    --image_folder your-image-path/ \
    --start_line 0 \
    --end_line 999
```
### Filter Hallucinations of Obj_extr_from_img
```bash
python filter/filter_fr_img.py \
    --model-config ./filter/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py \
    --model-checkpoint ./ckpt/groundingdino_swinb_ogc.pth \
    --box-threshold 0.45 \
    --text-threshold 0.15 \
    --input-file-path obj_extr_from_img.jsonl \
    --output-file-path obj_extr_from_img_wo_hal.jsonl \
    --start_line 0 \
    --end_line 999
```
### Transfer Image to Depth Map
```bash
python ./utils/trans_img2depth.py \
    --input_file image.jsonl \
    --output_folder depth-map-folder/ \
    --image_folder your-image-folder/ \
    --start_line 0 \
    --end_line 999
```

### Fine-grained Annotation
```bash
python fg_annotation/mask_depth.py \
    --input_path image-box-caption.jsonl \
    --output_path fg_anno.jsonl \
    --image_folder your-image-folder\ \
    --image_depth_folder depth-map-folder/ \
    --start_line 0 \
    --end_line 999
```

### Description Refinement
- only for llama version
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python ./refine/add_detail.py \
--input_file fg_anno.jsonl \
--output_file refined_desc.jsonl \
--stop_tokens "<|eot_id|>" \
--prompt_structure "<|begin_of_text|><|start_header_id|>user<|end_header_id|>{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>" \
--start_line 0 \
--end_line 999
```


## Evaluation
### 📊 DenseCap Benchmark
***Benchmark for evaluating details in descriptions.***
Requirements for DenseCap evaluation are in [DenseCap.md](https://github.com/sterzhang/image-textualization/blob/main/docs/DenseCap.md).
1. Inference MLLMs with questions in `benchamrk/DenseCap_Metrics/ques/` and images in ``benchamrk/DenseCap_Metrics/DenCap_image/`, then put your inference results in `benchamrk/DenseCap_Metrics/res/`.
2. Use the transfer-script in `script/trans_to_DenseCap_format.py`, transferring the right format for evaluation.
3. Run `benchamrk/DenseCap_Metrics/eval_DenseCap.py`, remember change path for evaluating gt-gpt4v-prefer or gt-llava-prefer. For example, when evaluating gt-gpt4v-prefer, you need to change to:
```bash
gt_file = './benchmark/DenseCap_Metrics/gt/gt-gpt4v-prefer.json'
res_file = './benchmark/DenseCap_Metrics/res/<your-res-path-after-transfer>.json'
```

### 📙 Linguistic Benchmark
***Benchmark for evaluating linguistic metrics of descriptions.***
1. Inference models finetuned on unmodified dataset and modified dataset respectively, putting results in `unmodi.jsonl` and `modi.jsonl` for example. Each line contains "image" and "description".
2. Use the transfer-script in `script/trans_to_Linguistic_format.py`, transferring the right format for evaluation. Remember put your path here:
```bash
# Define the file paths
modi_file_path = 'modi.jsonl'
unmodi_file_path = 'unmodi.jsonl'
output_file_path = 'linguistic-cmp.jsonl'
```
3. Run `./benchmark/Linguistic/readability.py`.
```bash
python ./benchmark/Linguistic/readability.py --file_path your-inference-result.jsonl --result_file_path ./bench_result/LinBench/LinBench_record.jsonl --start_line 0 --end_line 101
```

### 🔎 POPE Benchmark
***Benchmark for evaluating model's hallucinations.***
Please see the official repo of [POPE](https://github.com/AoiDragon/POPE/).
Here we provide muti-gpu inference scripts (`./benchmark/POPE/eval_pope_muti_gpu.sh`) for quicker inference. Then run `./benchmark/POPE/eval_pope.py` for results.

### 📷 T2IGen Benchmark
***Benchmark for evaluating the image generated by dense description.***
For example, generating images using seed 595:
```bash
bash ./benchmark/t2iGen/eval_t2i.sh cuda:0 595
```

## Visualization
<img width="943" alt="image" src="https://github.com/sterzhang/image-textualization/assets/119802220/4f92f45a-d1c2-4576-b008-641534e0b743">

<img width="1029" alt="image" src="https://github.com/sterzhang/image-textualization/assets/119802220/4048a807-bab8-40dc-959f-dd6ddeb10b7c">