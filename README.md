# Image Textualization: An Automatic Framework for Creating Accurate and Detailed Image Descriptions
![image](https://github.com/sterzhang/image-textualization/assets/119802220/c72ff11a-2b39-4e20-88b5-d3f0d8f9eb42)


🔥 The data can be found in [🤗Huggingface](https://huggingface.co/datasets/Sterzhang/image-textualization/).

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



## Visualization
<img width="833" alt="image" src="https://github.com/sterzhang/image-textualization/assets/119802220/9562860a-96b6-4253-9305-d133161eea70">
