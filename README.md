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
```json
{"image": "xxx.jpg"} 
⬇️
{"image": "xxx.jpg", "extr_obj_fr_img": ["obj1","obj2"], "bounding_boxes": [[206, 137, 426, 364], [418, 119, 639, 388]]}
```

### Extract Objects from Descriptions
```json
{"image": "xxx.jpg", "description": "xxxxxxxx."} 
⬇️
{"image": "xxx.jpg", "extr_obj_fr_desc": ["obj1","obj2"], "description": "xxxxxxxx."}
```


### Filter Hallucinations of extr_obj_fr_desc
```json
{"image": "xxx.jpg", "extr_obj_fr_desc": ["obj1","obj2"], "description": "xxxxxxxx."} 
⬇️
{"image": "xxx.jpg", "del_obj_from_desc": ["hal2"], "description": "xxxxxxxx."}
```

### Filter Hallucinations of extr_obj_fr_img
```json
{"image": "xxx.jpg", "extr_obj_fr_img": ["obj1","obj2"], "bounding_boxes": [[206, 137, 426, 364], [418, 119, 639, 388]]} 
⬇️
{"image": "xxx.jpg", "exist_obj_from_img": ["obj2"], "bounding_boxes": [[418, 119, 639, 388]]}
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

### Fine-grained Annotation for exist_obj_from_img
```json
{"image": "xxx.jpg", "exist_obj_from_img": ["obj2"], "bounding_boxes": [[418, 119, 639, 388]], } 
⬇️
{"image": "xxx.jpg", "exist_obj_from_img": ["obj2"], "bounding_boxes": [[418, 119, 639, 388]], "object_depth": [83], "size": [12428], "width": 640, "height": 480}
```

### Description Refinement
```json
{"image": "xxx.jpg", "del_obj_from_desc": ["hal2"], "exist_obj_from_img": ["obj2"], "bounding_boxes": [[418, 119, 639, 388]], "object_depth": [83], "size": [12428], "width": 640, "height": 480, "description": "xxxxxxxx."}
⬇️
{"image": "xxx.jpg", "original_description": "xxxxx", "modified_description": "xxxxxxxx."}
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