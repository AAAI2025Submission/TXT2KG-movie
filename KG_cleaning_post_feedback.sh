set -v
epoch_num=0
port=7687



python graph_cleaning/relation_alignment.py --url bolt://localhost \
                                            --username [YOUR_NEO4J_USERNAME] \
                                            --password [YOUR_NEO4J_PASSWORD] \
                                            --port $port \
                                            --epoch_num $epoch_num \
                                            --load_path results



python graph_cleaning/entity_alignment.py --url bolt://localhost \
                                            --username [YOUR_NEO4J_USERNAME] \
                                            --password [YOUR_NEO4J_PASSWORD] \
                                            --port $port \
                                            --epoch_num $epoch_num \
                                            --load_path results
