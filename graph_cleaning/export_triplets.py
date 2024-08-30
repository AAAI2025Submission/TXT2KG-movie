import os
import argparse

from py2neo import Graph

parser = argparse.ArgumentParser(description="Parser for Building ChatGPT")
parser.add_argument('--url', type=str, default="bolt://localhost")
parser.add_argument('--username', type=str, default="[YOUR_NEO4J_USERNAME]")
parser.add_argument('--password', type=str, default="[YOUR_NEO4J_PASSWORD]")
parser.add_argument('--port', type=int, default=7687)
parser.add_argument('--epoch_num', type=int, default=0)
parser.add_argument('--save_path', type=str, default="results")

args=parser.parse_args()

epoch_num = args.epoch_num

url = f"{args.url}:{args.port}"
username =args.username
password = args.password

graph = Graph(url, auth=(username, password))

# Write a Cypher query to retrieve all relationships in the graph
query = """
MATCH (h)-[r]->(t)
RETURN h.id AS head, type(r) AS relation, t.id AS tail
"""

# Execute the query and fetch the results
results = graph.run(query).data()

# Format the results into the desired (h,r,t) format
triplets = [(result['head'], result['relation'], result['tail']) for result in results]

# Export the triplets to a file
if not os.path.exists(f'{args.save_path}/epoch_{epoch_num}'):
    os.makedirs(f'{args.save_path}/epoch_{epoch_num}')
with open(f'{args.save_path}/epoch_{epoch_num}/triplets.txt', 'w', encoding='utf-8') as f:
    for triplet in triplets:
        f.write(f"{triplet[0]}\t{triplet[1]}\t{triplet[2]}\n")