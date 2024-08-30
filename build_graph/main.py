#This script is modified from https://bratanic-tomaz.medium.com/constructing-knowledge-graphs-from-text-using-openai-functions-096a6d010c17

from langchain_community.graphs import Neo4jGraph
from langchain_core.load import Serializable
import argparse

parser = argparse.ArgumentParser(description="Parser for Building ChatGPT")
parser.add_argument('--url', type=str, default="bolt://localhost")
parser.add_argument('--port', type=int, default=7687)
parser.add_argument('--username', type=str, default="[YOUR_NEO4J_USERNAME]")
parser.add_argument('--password', type=str, default="[YOUR_NEO4J_PASSWORD]")
parser.add_argument('--load_path', type=str, default="movies")
parser.add_argument('--openai_api_key', type=str, default="[YOUR_OPENAI_API_KEY]")
parser.add_argument('--openai_model_name', type=str, default="gpt-3.5-turbo")
parser.add_argument('--temperature', type=float, default=0)
parser.add_argument('--max_repeat_num', type=int, default=10)

args=parser.parse_args()
url =f'{args.url}:{args.port}'
username =args.username
password = args.password
graph = Neo4jGraph(
    url=url,
    username=username,
    password=password
)

from langchain_community.graphs.graph_document import (
    Node as BaseNode,
    Relationship as BaseRelationship,
    GraphDocument,
)
from langchain.schema import Document
from typing import List, Dict, Any, Optional
from langchain.pydantic_v1 import Field, BaseModel

class Property(BaseModel):
  """A single property consisting of key and value"""
  key: str = Field(..., description="key")
  value: str = Field(..., description="value")

class Node(BaseNode):
    properties: Optional[List[Property]] = Field(
        None, description="List of node properties")

class Relationship(BaseRelationship):
    properties: Optional[List[Property]] = Field(
        None, description="List of relationship properties"
    )

class KnowledgeGraph(BaseModel):
    """Generate a knowledge graph with entities and relationships."""
    nodes: List[Node] = Field(
        ..., description="List of nodes in the knowledge graph")
    rels: List[Relationship] = Field(
        ..., description="List of relationships in the knowledge graph"
    )

def format_property_key(s: str) -> str:
    words = s.split()
    if not words:
        return s
    first_word = words[0].lower()
    capitalized_words = [word.capitalize() for word in words[1:]]
    return "".join([first_word] + capitalized_words)

def props_to_dict(props) -> dict:
    """Convert properties to a dictionary."""
    properties = {}
    if not props:
      return properties
    for p in props:
        properties[format_property_key(p.key)] = p.value
    return properties


def check_node_exists(node_id):
    node = graph.evaluate(f"MATCH (n) WHERE n.id = '{node_id}' RETURN n")
    return node is not None
def map_to_base_node(node: Node,filename:str) -> BaseNode:
    """Map the KnowledgeGraph Node to the base Node."""
    properties = props_to_dict(node.properties) if node.properties else {}
    # Add name property for better Cypher statement generation
    properties["name"] = node.id.title()
    properties["filename"]=filename

    return BaseNode(
        id=node.id.title(), type=node.type.capitalize(), properties=properties
    )


def map_to_base_relationship(rel: Relationship,filename:str) -> BaseRelationship:
    """Map the KnowledgeGraph Relationship to the base Relationship."""
    source = map_to_base_node(rel.source,filename)
    target = map_to_base_node(rel.target,filename)
    properties = props_to_dict(rel.properties) if rel.properties else {}
    properties["filename"]=filename
    return BaseRelationship(
        source=source, target=target, type=rel.type, properties=properties
    )

import os
from langchain.chains.openai_functions import (
    create_openai_fn_chain,
    create_structured_output_chain,
)
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

os.environ["OPENAI_API_KEY"] = args.openai_api_key
llm = ChatOpenAI(model=args.openai_model_name, temperature=args.temperature)

def get_extraction_chain(
    allowed_nodes: Optional[List[str]] = None,
    allowed_rels: Optional[List[str]] = None
    ):
    prompt = ChatPromptTemplate.from_messages(
        [(
          "system",
          f"""# Movie Knowledge Graph Instructions for GPT-4
## 1. Overview
You are a top-tier algorithm designed for extracting movie information, converting them into structured formats to build a movie knowledge graph. Try your best to structure it into nodes and relationships (Please use node properties as few as possible).
- **Nodes** represent entities and concepts. They're akin to Wikipedia nodes.
- **Relationships** are connections between nodes. Please prioritize relations over properties.
- The aim is to achieve simplicity and clarity in the knowledge graph, making it accessible for a vast audience.
## 2. Labeling Nodes
- **Consistency**: Ensure you use basic or elementary types for node labels.
  - For example, when you identify an entity representing a person, always label it as **"person"**. Avoid using more specific terms like "mathematician" or "scientist".
- **Node IDs**: Never utilize integers as node IDs. Node IDs should be names or human-readable identifiers found in the text.
{'- **Allowed Node Labels:**' + ", ".join(allowed_nodes) if allowed_nodes else ""}
{'- **Allowed Relationship Types**:' + ", ".join(allowed_rels) if allowed_rels else ""}
## 3. Handling Numerical Data and Dates
- Numerical data, like age or other related information, should be incorporated as attributes or properties of the respective nodes.
- **No Separate Nodes for Dates/Numbers**: Do not create separate nodes for dates or numerical values. Always attach them as attributes or properties of nodes.
- **Property Format**: Properties must be in a key-value format.
- **Quotation Marks**: Never use escaped single or double quotes within property values.
- **Naming Convention**: Use camelCase for property keys, e.g., `birthDate`.
## 4. Coreference Resolution
- **Maintain Entity Consistency**: When extracting entities, it's vital to ensure consistency.
If an entity, such as "John Doe", is mentioned multiple times in the text but is referred to by different names or pronouns (e.g., "Joe", "he"),
always use the most complete identifier for that entity throughout the knowledge graph. In this example, use "John Doe" as the entity ID.
Remember, the knowledge graph should be coherent and easily understandable, so maintaining consistency in entity references is crucial.
## 5. Strict Compliance
Adhere to the rules strictly. Non-compliance will result in termination.
          """),
            ("human", "Use the given format to extract information from the following input: {input}"),
            ("human", "Tip: Make sure to answer in the correct format"),
        ])
    return create_structured_output_chain(KnowledgeGraph, llm, prompt, verbose=False)

def extract_and_store_graph(
    document: Document,
    filename:str,
    nodes:Optional[List[str]] = None,
    rels:Optional[List[str]]=None) -> None:
    # Extract graph data using OpenAI functions
    extract_chain = get_extraction_chain(nodes, rels)
    data = extract_chain.invoke(document.page_content)['function']
    # Construct a graph document
    graph_document = GraphDocument(
      nodes = [map_to_base_node(node,filename) for node in data.nodes],
      relationships = [map_to_base_relationship(rel,filename) for rel in data.rels],
      source = document
    )
    # Store information into a graph
    graph.add_graph_documents([graph_document])

from langchain_community.document_loaders import WikipediaLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_community.document_loaders import TextLoader

# Read the wikipedia article
# raw_documents = WikipediaLoader(query="Walt Disney",load_max_docs=3).load()
from tqdm import tqdm
initial=0
dirs=os.listdir(args.load_path)
for idx,p in tqdm(enumerate(dirs),total=len(dirs),initial=initial):
    raw_documents=TextLoader(os.path.join(args.load_path, p), encoding='utf-8').load()
    # # # Define chunking strategy
    text_splitter = TokenTextSplitter(chunk_size=2048, chunk_overlap=24)
    # # Only take the first the raw_documents
    documents = text_splitter.split_documents(raw_documents)



    # Specify which node labels should be extracted by the LLM
    # allowed_nodes = ["Person", "Company", "Location", "Event", "Movie", "Service", "Award"]

    for i, d in enumerate(documents):
        for repeat in range(args.max_repeat_num):
            try:
                extract_and_store_graph(d,p)
                break
            except Exception as e:
                print(e)
                continue