from neo4j import GraphDatabase
import argparse

parser = argparse.ArgumentParser(description="Parser for Building ChatGPT")
parser.add_argument('--url', type=str, default="bolt://localhost")
parser.add_argument('--port', type=int, default=7687)
parser.add_argument('--username', type=str, default="[YOUR_NEO4J_USERNAME]")
parser.add_argument('--password', type=str, default="[YOUR_NEO4J_PASSWORD]")
args=parser.parse_args()
url = f"{args.url}:{args.port}"
username =args.username
password = args.password
driver = GraphDatabase.driver(url, auth=(username, password))  # replace with your actual username and password

def list_relationship_types(tx):
    result = tx.run("CALL db.relationshipTypes()")
    return [record[0] for record in result]

with driver.session() as session:
    relationship_types = session.read_transaction(list_relationship_types)

# print(relationship_types)