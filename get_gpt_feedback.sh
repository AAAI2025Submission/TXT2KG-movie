set -v
epoch_num=1

python graph_cleaning/GPT_cluster_feedback.py --load_path results \
                                              --save_path results \
                                              --epoch_num $epoch_num \
                                              --openai_api_key [YOUR_OPENAI_API_KEY] \
                                              --model gpt-3.5-turbo \
                                              --temperature 0 \
                                              --use_entity_KG_clusters True \
                                              --use_relation_KG_clusters True \
                                              --use_entity_semantic_clusters True \
                                              --use_relation_semantic_clusters True \
                                              --max_cluster_size 20
