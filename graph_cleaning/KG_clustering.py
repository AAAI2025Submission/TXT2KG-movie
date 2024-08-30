import numpy as np
from pykeen.models import TuckER
from pykeen.triples import TriplesFactory
import torch

from sklearn.cluster import DBSCAN, KMeans,HDBSCAN
import argparse

from sklearn.metrics import silhouette_score
from tqdm import tqdm

import random
import pickle

import time
random.seed(time.time())

parser = argparse.ArgumentParser(description="Parser for Building ChatGPT")
parser.add_argument('--save_path', type=str, default="results")
parser.add_argument('--load_path', type=str, default="results")
parser.add_argument('--epoch_num', type=int, default=0)
parser.add_argument('--enable_entity_cluster', type=bool, default=True)
parser.add_argument('--enable_relation_cluster', type=bool, default=True)
parser.add_argument('--entity_cluster_method', type=str, default='dbscan')
parser.add_argument('--relation_cluster_method', type=str, default='dbscan')
parser.add_argument('--kmeans_entity_cluster_density', type=int, default=10)
parser.add_argument('--kmeans_relation_cluster_density', type=int, default=10)
parser.add_argument('--dbscan_entity_cluster_eps', type=int, default=6)
parser.add_argument('--dbscan_relation_cluster_eps', type=int, default=6)
parser.add_argument('--dbscan_entity_cluster_min_samples', type=int, default=2)
parser.add_argument('--dbscan_relation_cluster_min_samples', type=int, default=2)
parser.add_argument('--kmeans_max_iter', type=int, default=30)
parser.add_argument('--auto', type=bool, default=True)
parser.add_argument('--entity_sample_num', type=int, default=-1)
parser.add_argument('--relation_sample_num', type=int, default=-1)

args=parser.parse_args()

epoch_num=args.epoch_num

with open(f'{args.load_path}/epoch_{epoch_num}/entities.txt','r',encoding='utf-8') as f:
    entities_raw = f.readlines()
    entities = []
    entity_mapping = {}
    entity_mapping_reverse = {}
    for e in entities_raw:
        e1=e.replace("\n",'')
        entity=e1.replace("_",' ')
        entity=entity.replace("+",' ')
        entity = entity.replace(",", ' ')
        entity=entity.lower()
        entities.append(entity)
        entity_mapping[e1]=entity
        entity_mapping_reverse[entity]=e1
    with open(f'{args.save_path}/epoch_{epoch_num}/entity_mapping.pkl','wb') as f:
        f.write(pickle.dumps([entity_mapping,entity_mapping_reverse]))

with open(f'{args.load_path}/epoch_{epoch_num}/relations.txt','r',encoding='utf-8') as f:
    relations_raw=f.readlines()
    relations=[]
    relation_mapping={}
    relation_mapping_reverse={}
    for r in relations_raw:
        r1=r.replace("\n",'')
        relation=r1.replace("_",' ')
        relation=relation.replace("+",' ')
        relation = relation.replace(",", ' ')
        relation=relation.lower()
        relations.append(relation)
        relation_mapping[r1]=relation
        relation_mapping_reverse[relation]=r1
    with open(f'{args.save_path}/epoch_{epoch_num}/relation_mapping.pkl','wb') as f:
        f.write(pickle.dumps([relation_mapping,relation_mapping_reverse]))

def generate_random_float_values(n, start, end):
    return [random.uniform(start, end) for _ in range(n)]

triplets=TriplesFactory.from_path(f'{args.load_path}/epoch_{epoch_num}/triplets.txt')

model = TuckER(triples_factory=triplets,random_seed=1234)
model.load_state_dict(torch.load(f'{args.load_path}/epoch_{epoch_num}/model.pkl'))



id2entity=triplets.entity_id_to_label
id2relation=triplets.relation_id_to_label

entity_embedding = model.entity_representations[0]
relation_embedding= model.relation_representations[0]
entity_numpy_array = list(entity_embedding.parameters())[0].detach().numpy()
relation_numpy_array = list(relation_embedding.parameters())[0].detach().numpy()

if args.entity_sample_num>0:
    sample_idxs=random.sample(range(len(entities)),args.entity_sample_num)
    id2entity=[id2entity[i] for i in sample_idxs]
    entity_numpy_array=entity_numpy_array[np.array(sample_idxs)]
if args.relation_sample_num>0:
    sample_idxs=random.sample(range(len(relations)),args.relation_sample_num)
    id2relation=[id2relation[i] for i in  sample_idxs]
    relation_numpy_array=relation_numpy_array[np.array(sample_idxs)]



if args.enable_entity_cluster:
    if args.entity_cluster_method == 'dbscan':
        if args.auto == True:
            max_clusters = -1
            best_eps = 0
            # Find the best configuration for DBSCAN using grid search with Silhouette Score
            for eps in tqdm(generate_random_float_values(10, 0.1, 10)):
                entity_dbscan = DBSCAN(eps=eps, min_samples=args.dbscan_entity_cluster_min_samples).fit(entity_numpy_array)

                # Calculate the Silhouette Score
                labels = entity_dbscan.labels_

                # Update the best score and parameters if this score is better
                if len(set(labels)) > max_clusters:
                    max_clusters = len(set(labels))
                    best_eps = eps
            if max_clusters == -1:
                raise ValueError("No valid configuration found for DBSCAN. Please try a different method.")
        else:
            best_eps = args.dbscan_entity_cluster_eps
        entity_dbscan=DBSCAN(eps=best_eps,min_samples=args.dbscan_entity_cluster_min_samples).fit(entity_numpy_array)
        entity_labels=entity_dbscan.labels_
        num_entity_labels=len(set(entity_labels))
        entity_label_dict = {i: [] for i in set(entity_labels)}

    elif args.entity_cluster_method == 'hdbscan':
        if args.auto == True:
            max_clusters = -1
            best_eps = 0
            # Find the best configuration for DBSCAN using grid search with Silhouette Score
            for eps in tqdm(generate_random_float_values(50, 0.1, 10)):
                entity_dbscan = HDBSCAN(cluster_selection_epsilon=eps, min_cluster_size=args.dbscan_entity_cluster_min_samples).fit(entity_numpy_array)

                # Calculate the Silhouette Score
                labels = entity_dbscan.labels_

                # Update the best score and parameters if this score is better
                if len(set(labels)) > max_clusters:
                    max_clusters = len(set(labels))
                    best_eps = eps
            if max_clusters == -1:
                raise ValueError("No valid configuration found for DBSCAN. Please try a different method.")
        else:
            best_eps = args.dbscan_entity_cluster_eps
        entity_dbscan=HDBSCAN(cluster_selection_epsilon=best_eps,min_cluster_size=args.dbscan_entity_cluster_min_samples).fit(entity_numpy_array)
        entity_labels=entity_dbscan.labels_
        num_entity_labels=len(set(entity_labels))
        entity_label_dict = {i: [] for i in set(entity_labels)}
    elif args.entity_cluster_method == 'kmeans':
        if args.auto == True:
            best_score = -1
            best_dense = 0
            # Find the best configuration for KMeans using grid search with Silhouette Score
            for k in tqdm(range(2, 20)):
                entity_kmeans = KMeans(n_clusters=len(id2entity)//k, random_state=0, max_iter=args.kmeans_max_iter).fit(entity_numpy_array)

                # Calculate the Silhouette Score
                labels = entity_kmeans.labels_

                if len(set(labels)) <=1:
                    continue
                score = silhouette_score(entity_numpy_array, labels)

                # Update the best score and parameters if this score is better
                if score > best_score:
                    best_score = score
                    best_dense = k

            if best_score == -1:
                raise ValueError("No valid configuration found for DBSCAN. Please try a different method.")
        else:
            best_dense = args.entity_cluster_density

        entity_kmeans= KMeans(n_clusters=len(id2entity)//best_dense, random_state=0,max_iter=args.kmeans_max_iter).fit(entity_numpy_array)
        entity_labels=entity_kmeans.labels_
        entity_label_dict = {i: [] for i in range(len(set(entity_labels)))}
    else:
        raise ValueError("Invalid entity cluster method. Please choose either dbscan or kmeans.")
    # Iterate over the labels and add the corresponding node to the appropriate list in the dictionary
    for i, label in enumerate(entity_labels):
        entity_label_dict[label].append(id2entity[i])

    # Iterate over the dictionary and print each label and its corresponding nodes
    with open(f'{args.save_path}/epoch_{epoch_num}/entity_KG_clusters.txt','w',encoding='utf-8') as f:
        for label, nodes in entity_label_dict.items():
            # print(f"Entity Label {label}: Nodes {nodes}")
            f.write(f"{label}:\t{nodes}\n")

    with open(f'{args.save_path}/epoch_{epoch_num}/entity_KG_clusters.csv','w',encoding='utf-8') as f:
        for label, nodes in entity_label_dict.items():
            if len(nodes) > 1:
                for node in nodes:
                    n1 = node.replace("_", ' ')
                    n1 = n1.replace("+", ' ')
                    n1 = n1.replace(",", ' ')
                    n1 = n1.lower()
                    f.write(f"{n1},")
                f.write('\n')


if args.enable_relation_cluster:


    if args.relation_cluster_method == 'dbscan':
        if args.auto == True:
            max_clusters = -1
            best_eps = 0
            # Find the best configuration for DBSCAN using grid search with Silhouette Score
            for eps in tqdm(generate_random_float_values(50, 0.1, 10)):
                dbscan = DBSCAN(eps=eps, min_samples=args.dbscan_relation_cluster_min_samples).fit(relation_numpy_array)

                # Calculate the Silhouette Score
                labels = dbscan.labels_
                # Update the best score and parameters if this score is better
                if len(set(labels)) > max_clusters:
                    max_clusters = len(set(labels))
                    best_eps = eps
            if max_clusters == -1:
                raise ValueError("No valid configuration found for DBSCAN. Please try a different method.")
        else:
            best_eps = args.dbscan_relation_cluster_eps
        relation_dbscan=DBSCAN(eps=best_eps,min_samples=args.dbscan_relation_cluster_min_samples).fit(relation_numpy_array)
        relation_labels=relation_dbscan.labels_
        num_relation_labels=len(set(relation_labels))
        relation_label_dict = {i: [] for i in set(relation_labels)}

    elif args.relation_cluster_method == 'hdbscan':
        if args.auto == True:
            max_clusters = -1
            best_eps = 0
            # Find the best configuration for DBSCAN using grid search with Silhouette Score
            for eps in tqdm(generate_random_float_values(50, 0.1, 10)):
                dbscan = HDBSCAN(cluster_selection_epsilon=eps, min_cluster_size=args.dbscan_relation_cluster_min_samples).fit(relation_numpy_array)

                # Calculate the Silhouette Score
                labels = dbscan.labels_
                # Update the best score and parameters if this score is better
                if len(set(labels)) > max_clusters:
                    max_clusters = len(set(labels))
                    best_eps = eps
            if max_clusters == -1:
                raise ValueError("No valid configuration found for DBSCAN. Please try a different method.")
        else:
            best_eps = args.dbscan_relation_cluster_eps
        relation_dbscan=HDBSCAN(cluster_selection_epsilon=best_eps,min_cluster_size=args.dbscan_relation_cluster_min_samples).fit(relation_numpy_array)
        relation_labels=relation_dbscan.labels_
        num_relation_labels=len(set(relation_labels))
        relation_label_dict = {i: [] for i in set(relation_labels)}
    elif args.relation_cluster_method == 'kmeans':
        if args.auto == True:
            max_clusters = -1
            best_dense = 0
            # Find the best configuration for KMeans using grid search with Silhouette Score
            for k in tqdm(range(2, 20)):
                kmeans = KMeans(n_clusters=len(id2relation)//k, random_state=0, max_iter=args.kmeans_max_iter).fit(relation_numpy_array)

                # Calculate the Silhouette Score
                labels = kmeans.labels_
                if len(set(labels)) <=1:
                    continue
                score = silhouette_score(relation_numpy_array, labels)

                # Update the best score and parameters if this score is better
                if score > max_clusters:
                    max_clusters = score
                    best_dense = k

            if max_clusters == -1:
                raise ValueError("No valid configuration found for DBSCAN. Please try a different method.")
        else:
            best_dense = args.relation_cluster_density
        relation_kmeans= KMeans(n_clusters=len(id2relation)//best_dense, random_state=0,max_iter=args.kmeans_max_iter).fit(relation_numpy_array)
        relation_labels=relation_kmeans.labels_
        relation_label_dict = {i: [] for i in range(len(set(relation_labels)))}
    else:
        raise ValueError("Invalid relation cluster method. Please choose either dbscan or kmeans.")

    # Iterate over the labels and add the corresponding node to the appropriate list in the dictionary
    for i, label in enumerate(relation_labels):
        relation_label_dict[label].append(id2relation[i].lower())

    # Iterate over the dictionary and print each label and its corresponding nodes
    with open(f'{args.save_path}/epoch_{epoch_num}/relation_KG_clusters.txt','w',encoding='utf-8') as f:
        for label, nodes in relation_label_dict.items():
            # print(f"Relation Label {label}: Nodes {nodes}")
            f.write(f"{label}:\t{nodes}\n")

    with open(f'{args.save_path}/epoch_{epoch_num}/relation_KG_clusters.csv','w',encoding='utf-8') as f:
        for label, nodes in relation_label_dict.items():
            if len(nodes) > 1:
                for node in nodes:
                    r1 = node.replace("_", ' ')
                    r1 = r1.replace("+", ' ')
                    r1 = r1.replace(",", ' ')
                    r1 = r1.lower()
                    f.write(f"{r1},")
                f.write('\n')