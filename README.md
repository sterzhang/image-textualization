# Image Textualization: An Automatic Framework for Generating Rich and Detailed Image Descriptions
<img width="1029" alt="image" src="https://github.com/sterzhang/image-textualization/assets/119802220/4048a807-bab8-40dc-959f-dd6ddeb10b7c">


🔥 The data can be downloaded directly from the `description/` folder.

## Contents
- [Install](#install)
- [Datasets](#datasets)
- [Evaluation](#evaluation)

## Install
See instructions in [Install.md](https://github.com/sterzhang/image-textualization/blob/main/docs/install.md).

## Datasets

## Evaluation
### DenseCap Benchmark
***Benchmark for evaluating details in descriptions.***
Requirements for DenseCap evaluation are in [DenseCap.md](https://github.com/sterzhang/image-textualization/blob/main/docs/DenseCap.md).
1. Inference MLLMs with questions in `benchamrk/DenseCap_Metrics/ques/` and images in ``benchamrk/DenseCap_Metrics/DenCap_image/`, then put your inference results in `benchamrk/DenseCap_Metrics/res/`.
2. Use the transfer-script in `script/trans_to_DenseCap_format.py`, transferring the right format for evaluation.
3. Run `benchamrk/DenseCap_Metrics/eval_DenseCap.py`, remember change path for evaluating gt-gpt4v-prefer or gt-llava-prefer. For example, when evaluating gt-gpt4v-prefer, you need to change to:
```bash
gt_file = './benchmark/DenseCap_Metrics/gt/gt-gpt4v-prefer.json'
res_file = './benchmark/DenseCap_Metrics/res/<your-res-path-after-transfer>.json'
```

### Linguistic Benchmark
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

### POPE Benchmark
***Benchmark for evaluating model's hallucinations.***
Please see the official repo of [POPE](https://github.com/AoiDragon/POPE/).
Here we provide muti-gpu inference scripts (`./benchmark/POPE/eval_pope_muti_gpu.sh`) for quicker inference. Then run `./benchmark/POPE/eval_pope.py` for results.

### T2IGen Benchmark
***Benchmark for evaluating the image generated by dense description.***
For example, generating images using seed 595:
```bash
bash ./benchmark/t2iGen/eval_t2i.sh cuda:0 595
```
