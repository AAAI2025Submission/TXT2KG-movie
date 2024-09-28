set -e
port=7687
username=[YOUR_NEO4J_USERNAME]
password=[YOUR_NEO4J_PASSWORD]
openai_api_key=[YOUR_OPENAI_API_KEY]
echo ==================================================
for epoch_num in $(seq 0 20)
do
echo ==================================================
echo Epoch number: $epoch_num
echo -------------------------
echo export_triplets
python graph_cleaning/export_triplets.py --url bolt://localhost \
                                         --username $username \
                                         --password $password \
                                          --port $port \
                                         --epoch_num $epoch_num \
                                         --save_path results

echo -------------------------
echo KG_embeddings
python graph_cleaning/KG_embeddings.py --save_path results \
                                       --load_path results \
                                       --epoch_num $epoch_num \
                                       --KG_train_epoch_num 100 \
                                       --seed 1234 \
                                       --model TuckER
echo -------------------------
echo get_entity_and_relation
python graph_cleaning/get_entity_and_relation.py --save_path results \
                                             --load_path results \
                                             --epoch_num $epoch_num
echo -------------------------
echo KG_clustering
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



echo -------------------------
echo semantic_clustering
python graph_cleaning/semantic_clustering.py --save_path results \
                                             --load_path results \
                                             --epoch_num $epoch_num \
                                             --model_name bert-base-uncased \
                                             --batch_size 64 \
                                              --max_length 128 \
                                             --enable_entity_cluster True \
                                             --enable_relation_cluster True \
                                             --entity_cluster_method dbscan \
                                             --relation_cluster_method dbscan \
                                             --auto True \
                                             --dbscan_entity_cluster_min_samples 2 \
                                             --dbscan_relation_cluster_min_samples 2
echo -------------------------
echo GPT_cluster_feedback
python graph_cleaning/GPT_cluster_feedback.py --load_path results \
                                              --save_path results \
                                              --epoch_num $epoch_num \
                                              --openai_api_key $openai_api_key \
                                              --model gpt-3.5-turbo \
                                              --temperature 0 \
                                              --use_entity_KG_clusters True \
                                              --use_relation_KG_clusters True \
                                              --use_entity_semantic_clusters True \
                                              --use_relation_semantic_clusters True \
                                              --max_cluster_size 20

echo -------------------------
echo relation_alignment

python graph_cleaning/relation_alignment.py --url bolt://localhost \
                                            --username $username \
                                            --password $password \
                                            --port $port \
                                            --epoch_num $epoch_num \
                                            --load_path results

echo -------------------------
echo entity_alignment
python graph_cleaning/entity_alignment.py --url bolt://localhost \
                                            --username $username \
                                            --password $password \
                                            --port $port \
                                            --epoch_num $epoch_num \
                                            --load_path results

echo -------------------------
echo hyper_entity_verification
python graph_cleaning/hyper_entity_verification.py --load_path results \
                                            --save_path results \
                                            --epoch_num $epoch_num \
                                            --xi 0.7 \
                                            --alpha 0.5 \
                                            --beta -0.1


echo -------------------------
echo hyper_relation_verification
python graph_cleaning/hyper_relation_verification.py --load_path results \
                                            --save_path results \
                                            --epoch_num $epoch_num \
                                            --xi 0.7 \
                                            --alpha 0.5 \
                                            --beta -0.1

echo -------------------------
echo cal_hyper_triplets
python graph_cleaning/cal_hyper_triplets.py --save_path results \
                                            --epoch_num $epoch_num \
                                            --load_path results

echo echo Epoch $epoch_num finished
echo ++++++++++++++++++++++++++++++++++++++++++++++++

done