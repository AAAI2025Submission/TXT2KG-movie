import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Parser for Building ChatGPT")
parser.add_argument('--load_path', type=str, default="results")
parser.add_argument('--save_path', type=str, default="results")
parser.add_argument('--epoch_num', type=int, default=1)

args=parser.parse_args()

epoch_num=args.epoch_num
accepted_triplet_set=set()
if epoch_num==0:
    with open(f"results/epoch_0/triplets.txt", 'r', encoding='utf-8') as f:
        for line in f:
            accepted_triplet_set.add(line.strip())
# print(f"Epoch 0:")
# print(len(triplet_set))
# print(len(accepted_triplet_set))
# print()
else:
    accepted_entities=[]
    accepted_relations=[]
    with open(f'{args.load_path}/epoch_{epoch_num-1}/hyper_graph/accepted_hyper_entities_for_next_epoch.txt','r',encoding='utf-8') as f:
        for line in f:
            accepted_entities.append(line.strip())
    with open(f'{args.load_path}/epoch_{epoch_num-1}/hyper_graph/accepted_hyper_relations_for_next_epoch.txt','r',encoding='utf-8') as f:
        for line in f:
            accepted_relations.append(line.strip())

    with open(f"{args.load_path}/epoch_{epoch_num-1}/accepted_triplets.txt",'r',encoding='utf-8') as f:
        for line in f:
            accepted_triplet_set.add(line.strip())
    with open(f"{args.load_path}/epoch_{epoch_num}/triplets.txt",'r',encoding='utf-8') as f:
        for line in f:
            h,r,t=line.strip().split('\t')
            if h in accepted_entities or r in accepted_relations or t in accepted_entities:
                accepted_triplet_set.add(line.strip())
with open(f"{args.save_path}/epoch_{epoch_num}/accepted_triplets.txt", 'w', encoding='utf-8') as f:
    for triplet in accepted_triplet_set:
        f.write(f"{triplet}\n")