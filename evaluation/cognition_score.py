import json
import argparse
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(_PROJECT_ROOT))
from src.data.constants import ROOT

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line.strip())
            data.append(json_obj)
    return data

def find_null_scores(model_eval_result_list):
    """
    Return a list of (img_name, reasoning_type, conclusion_text) for any null/None score.
    """
    hits = []
    for sample in model_eval_result_list:
        # Skip metadata row
        if "Image Name" not in sample:
            continue

        img_name = sample.get("Image Name", "<unknown>")

        for k, v in sample.items():
            if "Reasoning" not in k:
                continue

            # Expected: v is a dict {conclusion: 0/1}
            if isinstance(v, dict):
                for conclusion, score in v.items():
                    if score is None:
                        hits.append((img_name, k, conclusion))
            else:
                # Unexpected format; still catch None
                if v is None:
                    hits.append((img_name, k, "<non-dict reasoning field>"))
    return hits


def assert_no_null_scores(model_eval_result_list):
    hits = find_null_scores(model_eval_result_list)
    if hits:
        print(f"[ERROR] Found {len(hits)} null/None score(s) in evaluation output. Aborting.")
        for img_name, rtype, concl in hits:
            print(f"  - Image: {img_name} | Reasoning: {rtype} | Conclusion: {concl}")
        raise ValueError("Null/None scores detected in evaluation file. Fix/rerun evaluation.")


def cognition_score(model_eval_result_list, scores_file=None, model_name=None, model_scenario=None, prompt_version=None):

    assert_no_null_scores(model_eval_result_list)

    reasoning_types = ['Special Time Reasoning', 'Location Reasoning', 'Character Reasoning', 'Character Relationship Reasoning', 
                  'Event Reasoning', 'Event Relationship Reasoning', 'Next Moment Event Reasoning', 'Mental State Reasoning']

    sample_scores_dict = {}
    for sample in model_eval_result_list:
        # Skip first row with model info
        if 'Image Name' not in sample:
            continue
            
        img_name = sample['Image Name']
        sample_scores = {}
        for k, v in sample.items():
            if 'Reasoning' in k:
                reasoning_type = k
                reasoning_scores = list(v.values())
                sample_scores[reasoning_type] = reasoning_scores
        sample_scores_dict[img_name] = sample_scores

    reasoning_scores_dict = {}
    for reasoning_type in reasoning_types:
        reasoning_scores_dict[reasoning_type] = []
        for img_name in sample_scores_dict:
            if reasoning_type in sample_scores_dict[img_name]:
                reasoning_scores_dict[reasoning_type] += sample_scores_dict[img_name][reasoning_type]

    for reasoning_type in reasoning_types:
        print(reasoning_type + ": ", '%.3f' % (sum(reasoning_scores_dict[reasoning_type])/len(reasoning_scores_dict[reasoning_type])))
    
    overall_score = sum([sum(reasoning_scores_dict[reasoning_type]) for reasoning_type in reasoning_types])/sum([len(reasoning_scores_dict[reasoning_type]) for reasoning_type in reasoning_types])
    print("Overall: ", '%.3f' % overall_score)
    
    # Save to ROOT/results/{model_scenario}/{model_name}/scores_{prompt_version}.json
    if model_name:
        if scores_file is not None:
            scores_file = Path(scores_file)
            results_dir = scores_file.parent
        else:
            model_dir_name = model_name.split("/")[-1] if "/" in model_name else model_name
            if model_scenario:
                if model_scenario in [0, "0"]:
                    scenario_dir_name = "baseline"
                elif model_scenario in [2, "2"]:
                    scenario_dir_name = "lgg"
                elif model_scenario in [3, "3"]:
                    scenario_dir_name = "dual_encoding"
                else:
                    scenario_dir_name = model_scenario
            else:
                scenario_dir_name = "unknown_scenario"

            results_dir = Path(ROOT) / "results" / scenario_dir_name / model_dir_name
            results_dir.mkdir(parents=True, exist_ok=True)
            if prompt_version:
                scores_file = results_dir / f"scores_{prompt_version}.json"
            else:
                scores_file = results_dir / "scores.json"
        
        scores_data = {}
        if scores_file.exists():
            with open(scores_file, 'r') as f:
                scores_data = json.load(f)
        
        if "model" in scores_data and scores_data["model"] != model_name:
            print(f"Warning: model name in scores file ({scores_data['model']}) does not match current model name ({model_name}). Overwriting with current model name. Exiting without saving scores.")
            return  # Do not save scores if model name doesn't match

        if model_scenario is None or "scenario" not in scores_data:
            print(f"Warning: scenario information is missing. Scores will be saved without scenario info.")
            scores_data["scenario"] = "unknown"
             
        if scores_data['scenario'] != model_scenario:
            print(f"Warning: scenario in scores file ({scores_data['scenario']}) does not match current scenario ({model_scenario}). Overwriting with current scenario. Exiting without saving scores.")
            return  # Do not save scores if scenario doesn't match
        
        if "prompt_version" in scores_data and scores_data["prompt_version"] != prompt_version:
            print(f"Warning: prompt version in scores file ({scores_data['prompt_version']}) does not match current prompt version ({prompt_version}). Overwriting with current prompt version. Exiting without saving scores.")
            return  # Do not save scores if prompt version doesn't match
        cognition_scores = {}
        for reasoning_type in reasoning_types:
            cognition_scores[reasoning_type] = float(sum(reasoning_scores_dict[reasoning_type])/len(reasoning_scores_dict[reasoning_type]))
        cognition_scores["overall"] = float(overall_score)
        
        scores_data["cognition"] = cognition_scores
        
        with open(scores_file, 'w') as f:
            json.dump(scores_data, f, indent=2)
        
        print(f"\nResults saved to {scores_file}")

def main(file_path, scores_file=None):
    if file_path.endswith('.jsonl'):
        model_eval_result_list = read_jsonl(file_path)
    elif file_path.endswith('.json'):
        with open(file_path) as f:
            model_eval_result_list = json.load(f)
    else:
        raise ValueError("File type not supported")
    
    # Extract model info from first row
    model_name = None
    model_scenario = None
    prompt_version = None
    if model_eval_result_list and 'model' in model_eval_result_list[0]:
        model_name = model_eval_result_list[0].get('model')
        model_scenario = model_eval_result_list[0].get('scenario')
        prompt_version = model_eval_result_list[0].get('prompt_version')
    
    cognition_score(model_eval_result_list, scores_file=scores_file, model_name=model_name, model_scenario=model_scenario, prompt_version=prompt_version)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_output_file_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--scores_file",
        type=str,
        default=None,
        help="File path to save scores.json. If not provided, saves to the same directory as the eval output file."
    )
    args = parser.parse_args()

    main(args.eval_output_file_path, scores_file=args.scores_file)
