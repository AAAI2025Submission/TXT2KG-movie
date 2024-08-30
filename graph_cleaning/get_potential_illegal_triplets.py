from collections import defaultdict
import argparse
parser = argparse.ArgumentParser(description="Parser for Building ChatGPT")
parser.add_argument('--save_path', type=str, default="results")
parser.add_argument('--load_path', type=str, default="results")
parser.add_argument('--epoch_num', type=int, default=0)
args=parser.parse_args()

epoch_num=args.epoch_num
relations=set()
with open(f"{args.load_path}/epoch_{epoch_num}/human_feedback/potential_illegal_relations.txt", encoding='utf-8') as f:
    lines=f.readlines()
    for line in lines:
        r=line.rstrip("\t\n").split("\t")
        for ri in r:
            relations.add(ri)

triplets=defaultdict(list)
with open(f"{args.load_path}/epoch_{epoch_num+1}/triplets.txt", encoding='utf-8') as f:
    lines=f.readlines()
    for line in lines:
        h,r,t=line.rstrip("\t\n").split("\t")
        r1 = r.replace("\n", '')
        r1 = r1.replace("_", ' ')
        r1 = r1.replace("+", ' ')
        r1 = r1.replace(",", ' ')
        r1 = r1.lower()
        if r1 in relations:
            triplets[r1].append((h,r1,t))

with open(f"{args.save_path}/epoch_{epoch_num+1}/potential_illegal_triplets.txt", 'w', encoding='utf-8') as f:
    for r in relations:
        for h,r,t in triplets[r]:
            f.write(f"{h}\t{r}\t{t}\n")


