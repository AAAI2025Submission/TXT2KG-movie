set -v
epoch_num=0
port=7687



python graph_cleaning/export_triplets.py --url bolt://localhost \
                                         --username [YOUR_NEO4J_USERNAME] \
                                         --password [YOUR_NEO4J_PASSWORD] \
                                          --port $port \
                                         --epoch_num $epoch_num \
                                         --save_path results



python graph_cleaning/KG_embeddings.py --save_path results \
                                       --load_path results \
                                       --epoch_num $epoch_num \
                                       --KG_train_epoch_num 100 \
                                       --seed 1234


python graph_cleaning/KG_clustering.py --save_path results \
                                       --load_path results \
                                       --epoch_num $epoch_num \
                                       --enable_entity_cluster True \
                                      --enable_relation_cluster True \
                                      --entity_cluster_method dbscan \
                                      --relation_cluster_method dbscan \
                                      --auto True \
                                      --dbscan_entity_cluster_min_samples 2 \
                                      --dbscan_relation_cluster_min_samples 2

python graph_cleaning/get_entity_and_relation.py --save_path results \
                                             --load_path results \
                                             --epoch_num $epoch_num

python graph_cleaning/semantic_clustering.py --save_path results \
                                             --load_path results \
                                             --epoch_num $epoch_num \
                                             --batch_size 64 \
                                              --max_length 128 \
                                             --enable_entity_cluster True \
                                             --enable_relation_cluster True \
                                             --entity_cluster_method dbscan \
                                             --relation_cluster_method dbscan \
                                             --auto True \
                                             --dbscan_entity_cluster_min_samples 2 \
                                             --dbscan_relation_cluster_min_samples 2