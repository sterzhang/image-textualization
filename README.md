# Image Textualization: An Automatic Framework for Creating Accurate and Detailed Image Descriptions
![image](https://github.com/sterzhang/image-textualization/assets/119802220/c72ff11a-2b39-4e20-88b5-d3f0d8f9eb42)

- [x] Main code for IT framework.
- [ ] Code for evaluation.
- [ ] Release the usage of our IT framework.
- [ ] Data cleaning is on-going. Expect to open-source 170K data before 6/17.

🔥 Now, 165K data can be found in [🤗Huggingface](https://huggingface.co/datasets/Sterzhang/image-textualization/). (Data cleaning...)

## Contents
- [Install](#install)
- [Datasets](#datasets)
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
├── vg
```



## Visualization
<img width="833" alt="image" src="https://github.com/sterzhang/image-textualization/assets/119802220/9562860a-96b6-4253-9305-d133161eea70">


## Acknowledgement

If you find our work useful for your research or applications, please cite using this BibTeX:
```bibtex
@misc{pi2024image,
      title={Image Textualization: An Automatic Framework for Creating Accurate and Detailed Image Descriptions}, 
      author={Renjie Pi and Jianshu Zhang and Jipeng Zhang and Rui Pan and Zhekai Chen and Tong Zhang},
      year={2024},
      eprint={2406.07502},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
