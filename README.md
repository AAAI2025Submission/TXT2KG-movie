## TXT2KG: Automated Construction of Hyper-Knowledge Graphs from Unstructured Text

**[Note: This repository is for the movie domain. For the movie domain, please refer to [this repository](https://github.com/AAAI2025Submission/TXT2KG).]**

This is the repository for AAAI anonymous submission **TXT2KG: Automated Construction of Hyper-Knowledge Graphs from Unstructured Text**.

In this paper, we propose TXT2KG, a comprehensive automated approach for constructing KGs that incorporates ontology-level knowledge. 
TXT2KG utilizes LLMs to extract both triples and properties from unstructured texts via carefully crafted prompts. 
Subsequently, TXT2KG performs knowledge deduplication and ontology structuring through clustering, leveraging semantic and KG neighbor information. 
After that, to further enhance the quality of these clusters, LLMs are applied for filtering. 
Finally, TXT2KG generate Hyper-Knowledge Graphs with validated hyper-triples.

![fig](https://github.com/AAAI2025Submission/TXT2KG/blob/master/fig/Arc_v2.png)

## Requirements

- googletrans==3.1.0a0
- langchain==0.2.14
- langchain_community==0.2.12
- langchain_core==0.2.35
- langchain_openai==0.1.22
- matplotlib==3.3.4
- neo4j==5.17.0
- numpy==1.21.5
- py2neo==2021.2.4
- pykeen==1.9.0
- scikit_learn==1.3.2
- torch==1.13.0+cu116
- tqdm==4.64.0
- transformers==4.18.0



## Quick Start
### Neo4j Configuration

1. Download Neo4j desktop [here](https://neo4j.com/download/).
2. Create a Neo4j database.
3. Enable APOC.

### Dataset

- We show some input examples in folder `movies` and `people`. The overall input unstructured data is available [here](https://entuedu-my.sharepoint.com/:f:/g/personal/zhixiang002_e_ntu_edu_sg/EkaSljUEEnJJpZVu66N3SMkBGPLNu6cetE3ausjzdIrh9w?e=HgvWaB).
- The generated property KG and hyper KG are available [here](https://entuedu-my.sharepoint.com/:f:/g/personal/zhixiang002_e_ntu_edu_sg/EiMeyt-8jRNHhuUynR3OVEwBh-rQUZu4oWQBlrBXBFqndA?e=5l4bpH).

These KGs and hyper KGs are dump files for Neo4j. Please load them into Neo4j following [this guide](https://neo4j.com/docs/desktop-manual/current/operations/create-from-dump/).

## TXT2KG Implementation

### Property KG Construction

1. Create a Neo4j database.
2. Enable APOC.
3. Generate an OpenAI API key for LLM usage [here](https://openai.com/index/openai-api/).
4. Set your `openai_api_key`, `username`, and `password` in `build_from_files.sh`.

```shell
./build_from_files.sh
```

### Hyper-KG Construction
1. TXT2KG directly modifies the Neo4j database. If you want to save the initial Property KG, dump and back up the dataset following [this guide](https://neo4j.com/docs/desktop-manual/current/operations/create-dump/).
2. Set your 'openai_api_key' ,`username` and `password` in `hyper.sh`.
3. Set the number of epochs you want.

```shell
./hyper.sh
```

## Citations

Currently not available.

## Q&A

For any questions, feel free to leave an issue.
Thank you very much for your attention and further contribution :)
