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
parser.add_argument('--xi', type=float, default=1)
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
    with open(f'{args.load_path}/epoch_{epoch_num}/relation_mapping.pkl', 'rb') as f:
        relation_mapping,relation_mapping_reverse = pickle.load(f)

    clusters = []
    with open(f"{args.load_path}/epoch_{epoch_num}/GPT_feedback/filtered_relation_clusters.csv", encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip(",\n")
            cluster = line.split(",")
            mapped_cluster = []
            for i in range(len(cluster)):
                if cluster[i] not in relation_mapping_reverse:
                    print(f"Warning: Relation {cluster[i]} not found in the mapping")
                    continue
                mapped_cluster.append(relation2id[relation_mapping_reverse[cluster[i]]])
            clusters.append(mapped_cluster)

    related_triplets = []
    triplets_numpy=triplets.mapped_triples
    for c in clusters:
        matched_heads_and_tails = [triplets_numpy[triplets_numpy[:, 1] == e][:, [0,2]] for e in c]
        if len(matched_heads_and_tails)>0:
            related_triplets.append(np.concatenate(matched_heads_and_tails))
        else:
            related_triplets.append(np.array([]))

    hyper_triplets=[]
    accepted_hyper_relations=[]
    for i in range(len(clusters)):
        hyper_relation ="+".join([id2relation[c] for c in clusters[i]])
        triplet_group=[]
        contain_hyper_entity=False
        for j in range(len(related_triplets[i])):
            # if "+" in id2entity[related_triplets[i][j][0]] or "+" in id2entity[related_triplets[i][j][1]]:
            #     contain_hyper_entity=True
            #     break
            triplet=(id2entity[related_triplets[i][j][0]],hyper_relation,id2entity[related_triplets[i][j][1]])
            if triplet not in triplet_group:
                triplet_group.append(triplet)
        if not contain_hyper_entity:
            hyper_triplets.append(triplet_group)

    test_triplets=[]
    for i in range(len(clusters)):
        relations_repeated=np.repeat(clusters[i],len(related_triplets[i]),axis=0)
        heas_and_tails=np.tile(related_triplets[i],(len(clusters[i]),1))
        if len(heas_and_tails)==0:
            test_triplets.append(np.array([]))
            continue
        corresponds_triplets = np.column_stack((heas_and_tails[:,0], relations_repeated, heas_and_tails[:,1]))
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

    hyper_candidate_num=0
    hyper_num=0
    with open(f'{args.load_path}/epoch_{epoch_num}/hyper_graph/hyper_relation_scores.txt','w',encoding='utf-8') as f:
        assert len(hyper_triplets) == len(all_scores)
        for i in range(len(hyper_triplets)):
            score=all_scores[i]
            triplets=hyper_triplets[i]
            if len(score)==0:
                continue
            max_score = np.max(score)
            avg_score = np.mean(score)
            min_score = np.min(score)
            for triplet in triplets:
                f.write(f"{triplet[0]},{triplet[1]},{triplet[2]}\n")

            hyper_candidate_num+=len(score)
            if conf(avg_score,min_score)>args.xi:

                hyper_num+=len(score)
                accepted_hyper_relations.append("+".join([id2relation[c] for c in clusters[i]]))
                f.write("Hyper relation\n")
            else:
                f.write("Candidate relation\n")
            f.write(f"{max_score},{avg_score},{min_score}\n\n")
    sum_hyper+=hyper_num
    sum_hyper_candidate+=hyper_candidate_num
    if epoch_num%5==0:
        print(f"Epoch {epoch_num}: Hyper num: {sum_hyper}, Hyper candidate num: {sum_hyper_candidate}")
    # print(f"Hyper candidate num: {hyper_candidate_num}")
    # print(f"Hyper num: {hyper_num}")

    with open(f'{args.load_path}/epoch_{epoch_num}/hyper_graph/hyper_relation_triplets.csv','w',encoding='utf-8') as f:
        for triplets in hyper_triplets:
            for triplet in triplets:
                f.write(f"{triplet[0]},{triplet[1]},{triplet[2]}\n")
            f.write("\n")

    if os.path.exists(f'{args.load_path}/epoch_{epoch_num-1}/hyper_graph/accepted_hyper_relations_for_next_epoch.txt'):
        shutil.copyfile(f'{args.load_path}/epoch_{epoch_num-1}/hyper_graph/accepted_hyper_relations_for_next_epoch.txt',
                        f'{args.load_path}/epoch_{epoch_num}/hyper_graph/accepted_hyper_relations_for_next_epoch.txt')
        with open(f'{args.load_path}/epoch_{epoch_num}/hyper_graph/accepted_hyper_relations_for_next_epoch.txt','a',encoding='utf-8') as f:
            for relation in accepted_hyper_relations:
                f.write(f"{relation}\n")
    else:
        with open(f'{args.load_path}/epoch_{epoch_num}/hyper_graph/accepted_hyper_relations_for_next_epoch.txt','w',encoding='utf-8') as f:
            for relation in accepted_hyper_relations:
                f.write(f"{relation}\n")

