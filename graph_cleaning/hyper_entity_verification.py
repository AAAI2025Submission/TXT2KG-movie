import numpy as np
from pykeen.models import TuckER
from pykeen.triples import TriplesFactory
import torch
import os
import argparse

import pickle
import torch
from tqdm import tqdm
import shutil

parser = argparse.ArgumentParser(description="Parser for Building ChatGPT")
parser.add_argument('--save_path', type=str, default="results")
parser.add_argument('--load_path', type=str, default="results")
parser.add_argument('--xi', type=float, default=0.7)
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--beta', type=float, default=-0.1)
parser.add_argument('--epoch_num', type=int, default=0)

args=parser.parse_args()
epoch_num=args.epoch_num

def conf(avg_score,min_score):
    return args.alpha*avg_score+args.beta*min_score

sum_hyper=0
sum_hyper_candidate=0
for epoch_num in range(0,21):
    triplets=TriplesFactory.from_path(f'{args.load_path}/epoch_{epoch_num}/triplets.txt')

    model = TuckER(triples_factory=triplets,random_seed=1234)
    model.load_state_dict(torch.load(f'{args.load_path}/epoch_{epoch_num}/model.pkl'))
    model.eval()

    entity2id=triplets.entity_to_id
    relation2id=triplets.relation_to_id
    id2entity=triplets.entity_id_to_label
    id2relation=triplets.relation_id_to_label

    entity_embedding = model.relation_representations[0]
    relation_embedding= model.relation_representations[0]
    with open(f'{args.load_path}/epoch_{epoch_num}/entity_mapping.pkl', 'rb') as f:
        entity_mapping,entity_mapping_reverse = pickle.load(f)

    clusters = []
    with open(f"{args.load_path}/epoch_{epoch_num}/GPT_feedback/filtered_entity_clusters.csv", encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip(",\n")
            cluster = line.split(",")
            mapped_cluster = []
            for i in range(len(cluster)):
                if cluster[i] not in entity_mapping_reverse:
                    print(f"Warning: Entity {cluster[i]} not found in the mapping")
                    continue
                mapped_cluster.append(entity2id[entity_mapping_reverse[cluster[i]]])
            clusters.append(mapped_cluster)

    entity_related_tails = []
    entity_related_heads=[]
    triplets_numpy=triplets.mapped_triples
    for c in clusters:
        matched_tails = [triplets_numpy[triplets_numpy[:, 0] == e][:, 1:] for e in c]
        matched_heads = [triplets_numpy[triplets_numpy[:, 2] == e][:, :2] for e in c]
        if len(matched_tails)>0:
            entity_related_tails.append(np.concatenate(matched_tails))
        else:
            entity_related_tails.append(np.array([]))
        if len(matched_heads)>0:
            entity_related_heads.append(np.concatenate(matched_heads))
        else:
            entity_related_heads.append(np.array([]))
    hyper_head_triplets=[]
    hyper_tail_triplets=[]
    for i in range(len(clusters)):
        hyper_entity="+".join([id2entity[c] for c in clusters[i]])
        triplet_group_head=[]
        triplet_group_tail=[]
        contain_hyper_head=False
        contain_hyper_tail=False
        for e in entity_related_tails[i]:
            # if "+" in id2relation[e[0]] or "+" in id2entity[e[1]]:
            #     contain_hyper_head=True
            #     break
            triplet=(hyper_entity,id2relation[e[0]],id2entity[e[1]])
            if triplet not in triplet_group_head:
                triplet_group_head.append(triplet)
        if not contain_hyper_head:
            hyper_head_triplets.append(triplet_group_head)

        for e in entity_related_heads[i]:
            # if "+" in id2entity[e[0]] or "+" in id2relation[e[1]]:
            #     contain_hyper_tail=True
            #     break
            triplet=(id2entity[e[0]],id2relation[e[1]],hyper_entity)
            if triplet not in triplet_group_tail:
                triplet_group_tail.append(triplet)
        if not contain_hyper_tail:
            hyper_tail_triplets.append(triplet_group_tail)

    test_triplets=[]
    for i in range(len(clusters)):
        heads_repeated=np.repeat(clusters[i],len(entity_related_tails[i]),axis=0)
        relations_and_tails=np.tile(entity_related_tails[i],(len(clusters[i]),1))
        tails_repeated=np.tile(clusters[i],len(entity_related_heads[i]))
        heads_and_relations=np.tile(entity_related_heads[i],(len(clusters[i]),1))
        corresponds_triplets = np.concatenate([np.column_stack((heads_repeated, relations_and_tails)),
                                               np.column_stack((heads_and_relations, tails_repeated))])
        test_triplets.append(corresponds_triplets)

    all_scores=[]
    for tris in tqdm(test_triplets):
        if len(tris)==0:
            all_scores.append(np.array([]))
            continue
        scores=model.score_hrt(torch.tensor(tris))
        all_scores.append(scores.detach().cpu().numpy())
    if not os.path.exists(f'{args.load_path}/epoch_{epoch_num}/hyper_graph'):
        os.makedirs(f'{args.load_path}/epoch_{epoch_num}/hyper_graph')

    assert len(all_scores)==len(hyper_head_triplets)==len(hyper_tail_triplets)
    hyper_candidate_num=0
    hyper_num=0
    accepted_hyper_entities=[]
    with open(f'{args.load_path}/epoch_{epoch_num}/hyper_graph/hyper_entity_scores.txt','w',encoding='utf-8') as f:
        for i in range(len(all_scores)):
            if len(all_scores[i])==0:
                continue
            max_score = np.max(all_scores[i])
            avg_score = np.mean(all_scores[i])
            min_score = np.min(all_scores[i])
            for triplet in hyper_head_triplets[i]+hyper_tail_triplets[i]:
                f.write(f"{triplet[0]},{triplet[1]},{triplet[2]}\n")
            hyper_candidate_num+=len(all_scores[i])
            if conf(avg_score,min_score)>args.xi:
                hyper_num+=len(all_scores[i])
                f.write("Hyper triplet\n")
                accepted_hyper_entities.append("+".join([id2entity[c] for c in clusters[i]]))
            else:
                f.write("Candidate triplet\n")
            f.write(f"{max_score},{avg_score},{min_score}\n\n")

    sum_hyper+=hyper_num
    sum_hyper_candidate+=hyper_candidate_num
    if epoch_num % 5 == 0:
        print(f"Epoch {epoch_num}: Hyper num: {sum_hyper}, Hyper candidate num: {sum_hyper_candidate}")
    with open(f'{args.load_path}/epoch_{epoch_num}/hyper_graph/hyper_head_triplets.csv','w',encoding='utf-8') as f:
        for triplets in hyper_head_triplets:
            for triplet in triplets:
                f.write(f"{triplet[0]},{triplet[1]},{triplet[2]}\n")
            if len(triplets)>0:
                f.write("\n")
    with open(f'{args.load_path}/epoch_{epoch_num}/hyper_graph/hyper_tail_triplets.csv','w',encoding='utf-8') as f:
        for triplets in hyper_tail_triplets:
            for triplet in triplets:
                f.write(f"{triplet[0]},{triplet[1]},{triplet[2]}\n")
            if len(triplets)>0:
                f.write("\n")

    if os.path.exists(f'{args.load_path}/epoch_{epoch_num-1}/hyper_graph/accepted_hyper_entities_for_next_epoch.txt'):
        shutil.copyfile(f'{args.load_path}/epoch_{epoch_num-1}/hyper_graph/accepted_hyper_entities_for_next_epoch.txt',f'{args.load_path}/epoch_{epoch_num}/hyper_graph/accepted_hyper_entities_for_next_epoch.txt')
        with open(f'{args.load_path}/epoch_{epoch_num}/hyper_graph/accepted_hyper_entities_for_next_epoch.txt','a',encoding='utf-8') as f:
            for entity in accepted_hyper_entities:
                f.write(f"{entity}\n")
    else:
        with open(f'{args.load_path}/epoch_{epoch_num}/hyper_graph/accepted_hyper_entities_for_next_epoch.txt','w',encoding='utf-8') as f:
            for entity in accepted_hyper_entities:
                f.write(f"{entity}\n")


