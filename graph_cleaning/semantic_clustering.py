from sklearn.metrics import silhouette_score
from transformers import AutoTokenizer, AutoModel
import torch
import random
from sklearn.cluster import DBSCAN, KMeans,HDBSCAN
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import pickle
import argparse
import time

random.seed(time.time())

parser = argparse.ArgumentParser(description="Parser for Building ChatGPT")
parser.add_argument('--save_path', type=str, default="results")
parser.add_argument('--load_path', type=str, default="results")
parser.add_argument('--epoch_num', type=int, default=0)
parser.add_argument('--model_name', type=str, default='bert-base-uncased')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--max_length', type=int, default=128)
parser.add_argument('--enable_entity_cluster', type=bool, default=True)
parser.add_argument('--enable_relation_cluster', type=bool, default=True)
parser.add_argument('--entity_cluster_method', type=str, default='dbscan')
parser.add_argument('--relation_cluster_method', type=str, default='dbscan')
parser.add_argument('--kmeans_entity_cluster_density', type=int, default=10)
parser.add_argument('--kmeans_relation_cluster_density', type=int, default=10)
parser.add_argument('--kmeans_max_iter', type=int, default=30)


parser.add_argument('--dbscan_entity_cluster_eps', type=int, default=6)
parser.add_argument('--dbscan_relation_cluster_eps', type=int, default=6)
parser.add_argument('--dbscan_entity_cluster_min_samples', type=int, default=2)
parser.add_argument('--dbscan_relation_cluster_min_samples', type=int, default=2)

parser.add_argument('--auto', type=bool, default=True)
parser.add_argument('--entity_sample_num', type=int, default=-1)
parser.add_argument('--relation_sample_num', type=int, default=-1)

args=parser.parse_args()

epoch_num=args.epoch_num

def generate_random_float_values(n, start, end):
    return [random.uniform(start, end) for _ in range(n)]

class NameDataset(Dataset):
    def __init__(self, names):
        self.names = names

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        return self.names[idx]

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










# Load pre-trained model and tokenizer
model_name = args.model_name
device='cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

if args.entity_sample_num>0:
    entities=random.sample(entities,args.entity_sample_num)
if args.relation_sample_num>0:
    relations=random.sample(relations,args.relation_sample_num)

entity_dataset = NameDataset(entities)
relation_dataset = NameDataset(relations)
entity_dataloader=DataLoader(entity_dataset,batch_size=args.batch_size,shuffle=False)
relation_dataloader=DataLoader(relation_dataset,batch_size=args.batch_size,shuffle=False)

if args.enable_entity_cluster:
    # Encode entities
    entity_embeddings=[]
    for e in tqdm(entity_dataloader):
        entity_input = tokenizer(e, padding=True, truncation=True, max_length=args.max_length, return_tensors='pt').to(device)

        # Get embeddings
        with torch.no_grad():
            model_output = model(**entity_input)
            entity_embeddings.append(model_output.last_hidden_state[:, 0, :])


    entity_embeddings=torch.cat(entity_embeddings)
    # print(entity_embeddings)


    entity_numpy_array=entity_embeddings.cpu().detach().numpy()
    if args.entity_cluster_method == 'dbscan':
        if args.auto == True:
            max_clusters = -1
            best_eps = 0
            # Find the best configuration for DBSCAN using grid search with Silhouette Score
            for eps in tqdm(generate_random_float_values(10, 0.1, 10)):
                entity_dbscan = DBSCAN(eps=eps, min_samples=args.dbscan_entity_cluster_min_samples).fit(
                    entity_numpy_array)

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
        entity_dbscan = DBSCAN(eps=best_eps, min_samples=args.dbscan_entity_cluster_min_samples).fit(entity_numpy_array)
        entity_labels = entity_dbscan.labels_
        num_entity_labels = len(set(entity_labels))
        entity_label_dict = {i: [] for i in set(entity_labels)}
    elif args.entity_cluster_method == 'hdbscan':
        if args.auto == True:
            max_clusters = -1
            best_eps = 0
            # Find the best configuration for DBSCAN using grid search with Silhouette Score
            for eps in tqdm(generate_random_float_values(10, 0.1, 10)):
                entity_dbscan = HDBSCAN(cluster_selection_epsilon=eps, min_cluster_size=args.dbscan_entity_cluster_min_samples).fit(
                    entity_numpy_array)

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
        entity_dbscan = HDBSCAN(cluster_selection_epsilon=best_eps, min_cluster_size=args.dbscan_entity_cluster_min_samples).fit(entity_numpy_array)
        entity_labels = entity_dbscan.labels_
        num_entity_labels = len(set(entity_labels))
        entity_label_dict = {i: [] for i in set(entity_labels)}
    elif args.entity_cluster_method == 'kmeans':
        if args.auto == True:
            best_score = -1
            best_dense = 0
            # Find the best configuration for KMeans using grid search with Silhouette Score
            for k in tqdm(range(2, 20)):
                entity_kmeans = KMeans(n_clusters=len(entities) // k, random_state=0,
                                       max_iter=args.kmeans_max_iter).fit(entity_numpy_array)

                # Calculate the Silhouette Score
                labels = entity_kmeans.labels_

                if len(set(labels)) <= 1:
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

        entity_kmeans = KMeans(n_clusters=len(entities) // best_dense, random_state=0,
                               max_iter=args.kmeans_max_iter).fit(entity_numpy_array)
        entity_labels = entity_kmeans.labels_
        entity_label_dict = {i: [] for i in range(len(set(entity_labels)))}
    else:
        raise ValueError("Invalid entity cluster method. Please choose either dbscan or kmeans.")

    # Iterate over the labels and add the corresponding node to the appropriate list in the dictionary
    for i, label in enumerate(entity_labels):
        entity_label_dict[label].append(entities[i])

    # Iterate over the dictionary and print each label and its corresponding nodes
    with open(f'{args.save_path}/epoch_{epoch_num}/entity_semantic_clusters.txt','w',encoding='utf-8') as f:
        for label, nodes in entity_label_dict.items():
            # print(f"Entity Label {label}: Nodes {nodes}")
            f.write(f"{label}:\t{nodes}\n")

    with open(f'{args.save_path}/epoch_{epoch_num}/entity_semantic_clusters.csv','w',encoding='utf-8') as f:
        for label, nodes in entity_label_dict.items():
            if len(nodes)>1:
                f.write(",".join(nodes)+"\n")

if args.enable_relation_cluster:
    # Encode relations
    relation_embeddings=[]
    for e in tqdm(relation_dataloader):
        relation_input = tokenizer(e, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)

        # Get embeddings
        with torch.no_grad():
            model_output = model(**relation_input)
            relation_embeddings.append(model_output.last_hidden_state[:, 0, :])

    relation_embeddings=torch.cat(relation_embeddings)



    relation_numpy_array=relation_embeddings.cpu().detach().numpy()

    if args.relation_cluster_method == 'dbscan':
        if args.auto == True:
            max_clusters = -1
            best_eps = 0
            # Find the best configuration for DBSCAN using grid search with Silhouette Score
            for eps in tqdm(generate_random_float_values(10, 0.1, 10)):
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
        relation_dbscan = DBSCAN(eps=best_eps, min_samples=args.dbscan_relation_cluster_min_samples).fit(
            relation_numpy_array)
        relation_labels = relation_dbscan.labels_
        num_relation_labels = len(set(relation_labels))
        relation_label_dict = {i: [] for i in set(relation_labels)}
    elif args.relation_cluster_method == 'hdbscan':
        if args.auto == True:
            max_clusters = -1
            best_eps = 0
            # Find the best configuration for DBSCAN using grid search with Silhouette Score
            for eps in tqdm(generate_random_float_values(10, 0.1, 10)):
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
        relation_dbscan = HDBSCAN(cluster_selection_epsilon=best_eps, min_cluster_size=args.dbscan_relation_cluster_min_samples).fit(
            relation_numpy_array)
        relation_labels = relation_dbscan.labels_
        num_relation_labels = len(set(relation_labels))
        relation_label_dict = {i: [] for i in set(relation_labels)}
    elif args.relation_cluster_method == 'kmeans':
        if args.auto == True:
            max_clusters = -1
            best_dense = 0
            # Find the best configuration for KMeans using grid search with Silhouette Score
            for k in tqdm(range(2, 20)):
                kmeans = KMeans(n_clusters=len(relations) // k, random_state=0, max_iter=args.kmeans_max_iter).fit(
                    relation_numpy_array)

                # Calculate the Silhouette Score
                labels = kmeans.labels_
                if len(set(labels)) <= 1:
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
        relation_kmeans = KMeans(n_clusters=len(relations) // best_dense, random_state=0,
                                 max_iter=args.kmeans_max_iter).fit(relation_numpy_array)
        relation_labels = relation_kmeans.labels_
        relation_label_dict = {i: [] for i in range(len(set(relation_labels)))}
    else:
        raise ValueError("Invalid relation cluster method. Please choose either dbscan or kmeans.")

    # Iterate over the labels and add the corresponding node to the appropriate list in the dictionary

    for i, label in enumerate(relation_labels):
        relation_label_dict[label].append(relations[i])

    # Iterate over the dictionary and print each label and its corresponding nodes
    with open(f'{args.save_path}/epoch_{epoch_num}/relation_semantic_clusters.txt','w',encoding='utf-8') as f:
        for label, nodes in relation_label_dict.items():
            # print(f"Relation Label {label}: Nodes {nodes}")
            f.write(f"{label}:\t{nodes}\n")

    with open(f'{args.save_path}/epoch_{epoch_num}/relation_semantic_clusters.csv','w',encoding='utf-8') as f:
        for label, nodes in relation_label_dict.items():
            if len(nodes)>1:
                f.write(",".join(nodes)+"\n")