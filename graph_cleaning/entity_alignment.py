import os
import pickle
import argparse

parser = argparse.ArgumentParser(description="Parser for Building ChatGPT")
parser.add_argument('--url', type=str, default="bolt://localhost")
parser.add_argument('--username', type=str, default="[YOUR_NEO4J_USERNAME]")
parser.add_argument('--password', type=str, default="[YOUR_NEO4J_PASSWORD]")
parser.add_argument('--port', type=int, default=7687)
parser.add_argument('--epoch_num', type=int, default=0)
parser.add_argument('--load_path', type=str, default="results")

args=parser.parse_args()

epoch_num = args.epoch_num
clusters = []
entity_mapping, entity_mapping_reverse = pickle.load(open(f"{args.load_path}/epoch_{epoch_num}/entity_mapping.pkl", "rb"))

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
            mapped_cluster.append(entity_mapping_reverse[cluster[i]])
        clusters.append(mapped_cluster)



from neo4j import GraphDatabase

url = f"{args.url}:{args.port}"
username = args.username
password = args.password
driver = GraphDatabase.driver(url, auth=(username, password))  # replace with your actual username and password

for cluster in clusters:
    with driver.session() as session:
        new_node_id = "+".join(cluster)
        session.run("CREATE (n:Node {id: $new_node_id})", new_node_id=new_node_id)

        for node_id in cluster:
            new_props = {}
            # Copy properties from the old node to the new node
            result = session.run("MATCH (n {id: $node_id}) RETURN properties(n) as props", node_id=node_id)
            for record in result:
                old_props = record["props"]
                for key, value in old_props.items():
                    new_props[f"{node_id}_{key}"] = value
            # add properties to the new node
            session.run("MATCH (n {id: $new_node_id}) SET n += $new_props", new_node_id=new_node_id, new_props=new_props)

            # Update relationships to point to the new node
            session.run("""
                MATCH (n {id: $old_node_id})-[r]->(m),(new_node {id: $new_node_id})
                WITH TYPE(r) as type, PROPERTIES(r) as props, r, m, new_node
                CALL apoc.create.relationship(new_node, type, props, m) YIELD rel AS new_r
                DELETE r
            """, old_node_id=node_id, new_node_id=new_node_id)

            session.run("""
                            MATCH (m)-[r]->(n {id: $old_node_id}),(new_node {id: $new_node_id})
                            WITH TYPE(r) as type, PROPERTIES(r) as props, r, m, new_node
                            CALL apoc.create.relationship(m, type, props, new_node) YIELD rel AS new_r
                            DELETE r
                        """, old_node_id=node_id, new_node_id=new_node_id)

            # Delete the old node
            session.run("MATCH (n {id: $old_node_id}) DELETE n", old_node_id=node_id)
