import os
import json
import time
import tqdm
import argparse
parser = argparse.ArgumentParser(description="Parser for Building ChatGPT")
parser.add_argument('--save_path', type=str, default="disease")
parser.add_argument('--load_file', type=str, default="data/disease_knowledges2_en.json")
args=parser.parse_args()


if not os.path.exists(args.save_path):
    os.mkdir(args.save_path)


with open(args.load_file, 'r', encoding='utf-8') as f:
    data = json.load(f)
    bar = tqdm.tqdm(total=len(data))
    for idx,d in enumerate(data):
        name_en=data[d]["cur_disease_title"]
        with open(f'{args.save_path}{name_en}.txt','w',encoding='utf-8') as f:
            for k,v in data[d].items():
                if type(v) is list:
                    s=v[0].replace("\n","").replace("\\n","")
                else:
                    s=v
                f.write(f'{k}: {s}\n')
        bar.update(1)
    bar.close()
