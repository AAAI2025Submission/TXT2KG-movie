from googletrans import Translator
import os
import json
import time
import tqdm
import pickle
import argparse

parser = argparse.ArgumentParser(description="Parser for Building ChatGPT")
parser.add_argument('--save_file', type=str, default="disease/disease_knowledges2.json")
parser.add_argument('--load_file', type=str, default="data/disease_knowledges2_en.json")
args=parser.parse_args()
def translate_text(text, translated_dict):
    translator = Translator()
    if text in translated_dict:
        return translated_dict[text]
    else:
        time.sleep(1)
        result = translator.translate(text, src='zh-CN', dest='en')
        translated_dict[text] = result.text

translated_dict={}
translated_data={}
idx=0
if os.path.exists("checkpoint.pkl"):
    with open("checkpoint.pkl","rb") as f:
        idx,translated_dict,translated_data=pickle.load(f)
try:
    with open(args.load_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        bar = tqdm.tqdm(total=len(data))
        keys=list(data.keys())
        while idx<50:
            translated_data[keys[idx]]={}
            for k,v in data[keys[idx]].items():
                if type(v) is list:
                    s=v[0].replace("\n","").replace("\\n","")
                else:
                    s=v
                if len(k)>0:
                    k_en=translate_text(k, translated_dict)
                if len(s)>0:
                    s_en=translate_text(s, translated_dict)
                translated_data[keys[idx]][k_en]=s_en
            bar.update(1)
            idx+=1
        bar.close()
except Exception as e:
    with open(f"checkpoint.pkl","wb") as f:
        pickle.dump([idx,translated_dict,translated_data],f)

with open(args.save_file, 'w', encoding='utf-8') as f:
    json.dump(translated_data,f,ensure_ascii=False,indent=4)
