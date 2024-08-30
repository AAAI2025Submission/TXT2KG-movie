python build_graph/translate.py --load_file disease/disease_knowledges2.json --save_file disease/disease_knowledges2_en.json
python build_graph/process_json.py --save_path disease/ --load_file disease/disease_knowledges2_en.json
python build_graph/main.py --url bolt://localhost \
                            --username [YOUR_NEO4J_USERNAME] \
                            --password [YOUR_NEO4J_PASSWORD] \
                            --port 7687 \
                            --load_path disease \
                            --openai_api_key [YOUR_OPENAI_API_KEY] \
                            --openai_model_name gpt-3.5-turbo \
                            --temerature 0