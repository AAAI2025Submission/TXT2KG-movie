import argparse
parser = argparse.ArgumentParser(description="Parser for Building ChatGPT")
parser.add_argument('--save_path', type=str, default="results")
parser.add_argument('--load_path', type=str, default="results")
parser.add_argument('--epoch_num', type=int, default=0)
args=parser.parse_args()
epoch_num=args.epoch_num
entities=set()
relations=set()

with open(f"{args.load_path}/epoch_{epoch_num}/triplets.txt", encoding='utf-8') as f:
    triplets = f.readlines()
    for tri in triplets:
        h,r,t=tri.rstrip("\t\n").split("\t")
        entities.add(h)
        entities.add(t)
        relations.add(r)
# print(entities)
# print(relations)

with open(f'{args.save_path}/epoch_{epoch_num}/entities.txt', 'w', encoding='utf-8') as f:
    for entity in entities:
        f.write(f"{entity}\n")
with open(f'{args.save_path}/epoch_{epoch_num}/relations.txt', 'w', encoding='utf-8') as f:
    for relation in relations:
        f.write(f"{relation}\n")

