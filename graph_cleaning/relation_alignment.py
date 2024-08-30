import os
import pickle
import argparse
import json

parser = argparse.ArgumentParser(description="Parser for Building ChatGPT")
parser.add_argument('--url', type=str, default="bolt://localhost")
parser.add_argument('--username', type=str, default="[YOUR_NEO4J_USERNAME]")
parser.add_argument('--password', type=str, default="[YOUR_NEO4J_PASSWORD]")
parser.add_argument('--port', type=int, default=7687)
parser.add_argument('--epoch_num', type=int, default=0)
parser.add_argument('--load_path', type=str, default="results")

args=parser.parse_args()

epoch_num = args.epoch_num
relation_mapping, relation_mapping_reverse = pickle.load(open(f"{args.load_path}/epoch_{epoch_num}/relation_mapping.pkl", "rb"))
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
            mapped_cluster.append(relation_mapping_reverse[cluster[i]])
        clusters.append(mapped_cluster)
def custom_serializer(obj):
    return None

from neo4j import GraphDatabase

url = f"{args.url}:{args.port}"
username =args.username
password = args.password
driver = GraphDatabase.driver(url, auth=(username, password))  # replace with your actual username and password

for cluster in clusters:
    with driver.session() as session:
        new_props = {}
        nodes_a = []
        nodes_b = []
        for c in cluster:
            # Retrieve all nodes connected by the old edge type
            result = session.run(f"MATCH (a)-[r:`{c}`]->(b) RETURN a, b, properties(r) as props")
            for record in result:
                old_props = record["props"]
                a=record["a"]
                b=record["b"]
                nodes_a.append(a)
                nodes_b.append(b)

                # Create a new edge with the unified name and copy the properties from the old edge with a prefix
                for key, value in old_props.items():
                    new_props[f"{c}_{key}"] = value

                # Delete the old edge
                session.run(
                    f"MATCH (a)-[r:`{c}`]->(b) WHERE id(a) = {a.id} AND id(b) = {b.id} DELETE r")
        for a, b in zip(nodes_a, nodes_b):
            session.run(f"""
                MATCH (a) WHERE id(a) = {a.id}
                MATCH (b) WHERE id(b) = {b.id}
                MERGE (a)-[r:`{"+".join(cluster)}`]->(b)
                SET r+=$new_props
            """, new_props=new_props)