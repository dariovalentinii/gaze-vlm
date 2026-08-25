import json
import sys
import spacy
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(_PROJECT_ROOT))
from src.data.constants import ROOT

nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('all-mpnet-base-v2')


def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line.strip())
            data.append(json_obj)
    return data

def entity_recall(entity_list, description, threshold=0.5, is_print=False):
    
    doc = nlp(description)
    desc_noun_list = list(set([token.norm_ for token in doc if token.pos_ == "NOUN" or token.pos_ == "PROPN"]))
    
    if not entity_list:
        return 0.0, 0, 0
    if not desc_noun_list:
        return 0.0, 0, len(entity_list)
    
    # if is_print:
    #     print("Entities: ", entity_list)
    #     print("Nouns in Description: ",desc_noun_list)

    #     print("Num. of Entities:             ", len(entity_list))
    #     print("Num. of Nouns in Description: ", len(desc_noun_list))
    
    entity_embeddings = model.encode(entity_list, convert_to_tensor=False)
    desc_noun_embeddings = model.encode(desc_noun_list, convert_to_tensor=False)

    sim_matrix = cosine_similarity(entity_embeddings, desc_noun_embeddings, dense_output=True)

    max_values = np.amax(sim_matrix, axis=1, keepdims=True)
    max_sim_matrix = np.where(sim_matrix == max_values, sim_matrix, 0)
    # if is_print:
    #     print(max_sim_matrix)
    
    exist_matrix = np.where(max_sim_matrix > threshold, max_sim_matrix, 0)
    non_zero_count = np.count_nonzero(exist_matrix)
    # nonzero_indices = np.where(~np.all(exist_matrix == 0, axis=1))[0]
    # non_zero_count = len(nonzero_indices)

    recall = non_zero_count/len(entity_list)
    hitted_num = non_zero_count
    entity_num = len(entity_list)

    if is_print:
        print("Hitted Num.: ", hitted_num, "/", entity_num)
    return recall, hitted_num, entity_num

def evaluator(description_json_file_path, model_output_file_path, threshold, is_print):

    with open(description_json_file_path, 'r') as json_file:
        data = json.load(json_file)
    
    model_output = read_jsonl(model_output_file_path)
    
    # Extract model name from metadata
    model_name = None
    model_method = None
    prompt_version = None
    for i in model_output:
        if 'model' in i:
            model_name = i.get('model', 'unknown')
            model_method = i.get('method', None)
            prompt_version = i.get('prompt_version', None)
            break

    recall_scores = {}
    total_hitted_num = 0
    total_entity_num = 0
    count = 0
    for i in model_output:
        # Skip first row with model info
        if 'filename' not in i:
            continue
            
        count += 1
        file_name = i['filename']
        model_description = i.get('entities_output', '')
        
        # Skip if entities_output is None or empty
        if not model_description:
            continue
            
        img_id = file_name.split('.')[0]
        if is_print:
            print(f"\nImage: {img_id}, {count}/{len(model_output)-1}")
                
        ann_entities = data[img_id]['Entities']
        r, hitted_num, entity_num = entity_recall(ann_entities, model_description, threshold, is_print)
        recall_scores[img_id] = r
        total_hitted_num += hitted_num
        total_entity_num += entity_num

    macro_avg_recall = sum(recall_scores.values())/len(recall_scores)
    micro_avg_recall = total_hitted_num/total_entity_num
    print("\nTotal entity num: ", total_entity_num)
    print("Total hitted num: ", total_hitted_num)

    return macro_avg_recall, micro_avg_recall, model_name, model_method, prompt_version

def main(cogbench_description_file_path, model_output_file_path, scores_output_dir):
    macro_avg_recall, micro_avg_recall, model_name, model_method, prompt_version = evaluator(cogbench_description_file_path, model_output_file_path, threshold=0.6, is_print=False)
    
    print("Macro Avg. Recall: ", macro_avg_recall)
    print("Micro Avg. Recall: ", micro_avg_recall)
    
    if model_name:
        if scores_output_dir:
            results_dir = Path(scores_output_dir)
        else:
            results_dir = Path(model_output_file_path).parent
        results_dir.mkdir(parents=True, exist_ok=True)
        
        if prompt_version:
            scores_file = results_dir / f"scores_{prompt_version}.json"
        else:
            scores_file = results_dir / "scores.json"
        
        scores_data = {}
        if scores_file.exists():
            with open(scores_file, 'r') as f:
                scores_data = json.load(f)
        
        scores_data["model"] = model_name
        if model_method is not None:
            scores_data["method"] = model_method
        if prompt_version is not None:
            scores_data["prompt_version"] = prompt_version
        scores_data["recognition"] = {
            "macro_avg_recall": float(macro_avg_recall),
            "micro_avg_recall": float(micro_avg_recall)
        }
        
        with open(scores_file, 'w') as f:
            json.dump(scores_data, f, indent=2)
        

        print(f"\nRecognition scores saved to {scores_file}")
        return scores_file
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cogbench_description_file_path",
        type=str,
        default=str(ROOT / "data" / "cogbench_v1-1" / "cogbench_v1_description.json"),
        help="Download cogbench.zip and `unzip cogbench.zip` and change the path here",
    )
    parser.add_argument(
        "--model_output_file_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--scores_output_dir",
        type=str,
        default=None,
        help="Optional output directory for scores.json. If omitted, uses results/<method>/<model>/.",
    )
    args = parser.parse_args()

    scores_file = main(args.cogbench_description_file_path, args.model_output_file_path, args.scores_output_dir)
