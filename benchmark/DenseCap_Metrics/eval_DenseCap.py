from pycocoevalcap.eval import eval
import json

gt_file = './benchmark/DenseCap_Metrics/gt/'
res_file = './benchmark/DenseCap_Metrics/res/'

with open(gt_file, 'r') as f: 
    gts = json.load(f)
with open(res_file, 'r') as f:
    res = json.load(f)

if __name__ == '__main__':
    mp = eval(gts, res)
    print(f"{res_file}: {mp}")
