import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Parser for Building ChatGPT")
parser.add_argument('--load_path', type=str, default="results")
parser.add_argument('--save_path', type=str, default="results")
parser.add_argument('--epoch_num', type=int, default=0)
parser.add_argument('--openai_api_key', type=str, default="[YOUR_OPENAI_API_KEY]")
parser.add_argument('--model', type=str, default="gpt-3.5-turbo")
parser.add_argument('--temperature', type=float, default=0)
parser.add_argument('--use_entity_KG_clusters', type=bool, default=True)
parser.add_argument('--use_relation_KG_clusters', type=bool, default=True)
parser.add_argument('--use_entity_semantic_clusters', type=bool, default=True)
parser.add_argument('--use_relation_semantic_clusters', type=bool, default=True)
parser.add_argument('--max_cluster_size', type=int, default=20)


args=parser.parse_args()

from typing import Optional, List

from langchain.chains.openai_functions import create_structured_output_chain
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from langchain_core.pydantic_v1 import BaseModel, Field

from langchain.chains.openai_functions import (
    create_openai_fn_chain,
    create_structured_output_chain,
)

# Initialize the OpenAI LLM
os.environ["OPENAI_API_KEY"] = args.openai_api_key
llm = ChatOpenAI(model=args.model,temperature=args.temperature,api_key=os.environ['OPENAI_API_KEY'])

class Cluster(BaseModel):
    items: List[str] = Field(
        None, description="List of the cluster items")
def set_prompt(input_type):
    if input_type=="Entity":

        prompt = ChatPromptTemplate.from_messages(
                [(
                  "system",
                  f"""# Entity Cluster Verification
        ## 1. Overview
        You are a a top-tier algorithm designed for merging EXACTLY THE SAME entity in a movie knowledge graph. Your tasks are:
        - To identify entity with EXACTLY THE SAME meanings from a provided list of entity names.
        - To return a list of EXACTLY THE SAME meaning entity.
        - Ensure that the output entity names are EXACTLY THE SAME as the input names (case-sensitive and with no alterations).
        
        ## 2. Input: 
        A list of entity names (may contain space).
        
        ## 3. Output:
        If no entity names are EXACTLY THE SAME, return 'None'.
        A list of entity names (may contain space) that are EXACTLY THE SAME.
        
        
        ## 4. Instructions:
        Analyze the provided list of entity names based on the specified criteria.
        Remove any entity that do not meet the criteria.
        Return a list of EXACTLY THE SAME entity names, ensuring the names are exactly as they appear in the input.
        Only output the list, do not include any additional information.
        
        ## 5. Examples:
        - Example 1:
        -- Input: [ 'Iron Man','Harry Potter and the Sorcerer's Stone', 'Captain America: The First Avenger']
        -- Output: ['Iron Man', 'Captain America: The First Avenger']
        
        - Example 2:
        -- Input: ['Harry Potter and the Sorcerer's Stone', 'Harry Potter and the Goblet of Fire', 'The Two Towers']
        -- Output: ['Harry Potter and the Sorcerer's Stone', 'Harry Potter and the Goblet of Fire']
        
        - Example 4:
        -- Input: ['Inception', 'The Matrix']
        -- Output: ['None']
        -- Explanation: There are no EXACTLY THE SAME entity names in the input list. Because they are related but not exactly the same.
    
    
        
        ## 6. Strict Compliance
        Adhere to the rules strictly. Non-compliance will result in termination.
                      """),
                    ("human", "Use the given format to extract information from the following input: {input}"),
                    ("human", "Tip: Make sure to answer in the correct format"),
                ])
    elif input_type=="Relation":

        prompt = ChatPromptTemplate.from_messages(
            [(
                "system",
                f"""# Relation Cluster Verification
                ## 1. Overview
                You are a a top-tier algorithm designed for merging EXACTLY THE SAME relation in a movie knowledge graph. Your tasks are:
                - To identify relation with EXACTLY THE SAME meanings from a provided list of relation names.
                - To return a list of EXACTLY THE SAME meaning relation.
                - Ensure that the output relation names are EXACTLY THE SAME as the input names (case-sensitive and with no alterations).

                ## 2. Input: 
                A list of relation names (may contain space).

                ## 3. Output:
                A list of relation names (may contain space) that are EXACTLY THE SAME.
                If no relation names are EXACTLY THE SAME, return 'None'.


                ## 4. Instructions:
                Analyze the provided list of relation names based on the specified criteria.
                Remove any relation that do not meet the criteria.
                Return a list of EXACTLY THE SAME relation names, ensuring the names are exactly as they appear in the input.
                Only output the list, do not include any additional information.

                ## 5. Examples:
                - Example 1:
                -- Input: ['HasSymptom', 'SymptomOf', 'Date']
                -- Output: ['HasSymptom','SymptomOf']
                
                - Example 2:
                -- Input: ['has risk factors', 'has subtypes', 'Date']
                -- Output: ['None']

                

                ## 6. Strict Compliance
                Adhere to the rules strictly. Non-compliance will result in termination.
                              """),
                ("human", "Use the given format to extract information from the following input: {input}"),
                ("human", "Tip: Make sure to answer in the correct format"),
            ])

    else:
        raise ValueError("Invalid input type. Please choose either Entity or Relation.")
    return prompt

entity_clusters = []
if args.use_entity_KG_clusters:
    with open(f"{args.load_path}/epoch_{args.epoch_num}/entity_KG_clusters.csv",'r', encoding='utf-8') as f:
        lines=f.readlines()
        for line in lines:
            c=line.rstrip(",\n").split(",")
            if len(c)<=args.max_cluster_size:
                entity_clusters.append(c)
if args.use_entity_semantic_clusters:
    with open(f"{args.load_path}/epoch_{args.epoch_num}/entity_semantic_clusters.csv",'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            c = line.rstrip(",\n").split(",")
            if len(c) <= args.max_cluster_size:
                entity_clusters.append(c)

filtered_entity_clusters = []
if len(entity_clusters)>0:
    chain = create_structured_output_chain(Cluster, llm, set_prompt("Entity"))
    for cluster in tqdm(entity_clusters):
        # use the runnable to get the output
        if len(cluster)<args.max_cluster_size:
            output_list = chain.run(str(cluster)).items
            # convert the output to a list
            if len(output_list)>1:
                filtered_entity_clusters.append(output_list)


relation_clusters = []
if args.use_relation_KG_clusters:
    with open(f"{args.load_path}/epoch_{args.epoch_num}/relation_KG_clusters.csv",'r', encoding='utf-8') as f:
        lines=f.readlines()
        for line in lines:
            c=line.rstrip(",\n").split(",")
            if len(c)<=args.max_cluster_size:
                relation_clusters.append(c)

if args.use_relation_semantic_clusters:
    with open(f"{args.load_path}/epoch_{args.epoch_num}/relation_semantic_clusters.csv",'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            c = line.rstrip(",\n").split(",")
            if len(c) <= args.max_cluster_size:
                relation_clusters.append(c)

filtered_relation_clusters = []
if len(relation_clusters)>0:
    chain = create_structured_output_chain(Cluster,llm, set_prompt("Relation"))
    for cluster in tqdm(relation_clusters):
        # use the runnable to get the output
        if len(cluster) < args.max_cluster_size:
            output_list = chain.run(str(cluster)).items
            # convert the output to a list
            if len(output_list)>1:
                filtered_relation_clusters.append(output_list)

if not os.path.exists(f"{args.save_path}/epoch_{args.epoch_num}/GPT_feedback"):
    os.makedirs(f"{args.save_path}/epoch_{args.epoch_num}/GPT_feedback")
with open(f"{args.save_path}/epoch_{args.epoch_num}/GPT_feedback/filtered_entity_clusters.csv",'w', encoding='utf-8') as f:
    for cluster in filtered_entity_clusters:
        f.write(",".join(cluster)+"\n")

with open(f"{args.save_path}/epoch_{args.epoch_num}/GPT_feedback/filtered_relation_clusters.csv",'w', encoding='utf-8') as f:
    for cluster in filtered_relation_clusters:
        f.write(",".join(cluster)+"\n")

