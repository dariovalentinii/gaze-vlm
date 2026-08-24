"""
    GPT-4 Eval.
"""

import re
import copy
import json
import argparse
import time
from pathlib import Path
import sys
from tqdm import tqdm
# from openai import OpenAI
from google import genai
import datetime

    
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(_PROJECT_ROOT))
from src.data.constants import ROOT, GEMINI_API_KEY

# Custom exceptions for Gemini API error handling
class GeminiAPIError(RuntimeError):
    """Request failed before producing a valid response object."""

class GeminiBlockedResponse(RuntimeError):
    """Request succeeded but model returned no text (e.g., blocked)."""

class GeminiEmptyTextResponse(RuntimeError):
    """Request succeeded and response exists, but response.text is missing/empty."""


def _is_retryable_unavailable_error(err: Exception) -> bool:
    """Return True for transient Gemini service-unavailable errors (e.g., 503 UNAVAILABLE)."""
    msg = str(err).upper()
    return (
        "503" in msg
        and "UNAVAILABLE" in msg
    )

# Global counter for API calls
count = 0
##############################

def _is_empty_model_output(value):
        if value is None:
            return True
        if isinstance(value, str):
            return value.strip() in {"", "None"}
        return False

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line.strip())
            data.append(json_obj)
    return data

def output_parse(model_output):

    # Check if the input string is just a single digit
    if re.match(r'^\d$', model_output):
        output = {"1": int(model_output)}
    # Check if the input string contains a single output in the format "[1]"
    elif re.match(r'\[\d\]', model_output):
        match = re.search(r'\[(\d)\]', model_output)
        output = {"1": int(match.group(1))}
    else:
        matches = re.findall(r'\d+\.\s\[(\d+)\]', model_output)
        output = {str(i + 1): int(match) for i, match in enumerate(matches)}

    return output

def gt_result_merge(gt, model_output):
    """
    Merge model outputs into data dict.
    Expects model_output entries with filename and separate output fields for each reasoning type.
    """
    gt_with_output = copy.deepcopy(gt)
    reasoning_types = ['Special Time Reasoning', 'Location Reasoning', 'Character Reasoning', 
                      'Character Relationship Reasoning', 'Event Reasoning', 
                      'Event Relationship Reasoning', 'Next Moment Event Reasoning', 'Mental State Reasoning']
    
    for i in gt:
        img_name = gt_with_output[i]['Image Name']
        model_entry = next(item for item in model_output if item['filename'] == img_name)
        
        # Map each reasoning type to its corresponding output field
        # Expected field names: special_time_reasoning_output, location_reasoning_output, etc.
        for reasoning_type in reasoning_types:
            # Convert reasoning type to snake_case field name
            field_name = reasoning_type.lower().replace(' ', '_') + '_output'
            if field_name in model_entry:
                gt_with_output[i][f'{reasoning_type}_output'] = model_entry[field_name]
            else:
                # Fallback to single model_output field if per-type outputs not available
                if 'model_output' in model_entry:
                    gt_with_output[i][f'{reasoning_type}_output'] = model_entry['model_output']
    
    return gt_with_output

def evaluation_data_format(gt_with_result):
    simplified_gt_with_result = {}
    for i in gt_with_result:
        # drop GT fields with "None"
        none_dropped = {k: v for k, v in gt_with_result[i].items() if k.endswith('_output') or (v[0]!='None' and not k.endswith('_output'))}
        
        # extraction conclusion from Reasonings
        conclusion_extracted = {}
        for k,v in none_dropped.items():
            if 'Reasoning' in k and k!="Event Relationship Reasoning" and not k.endswith('_output'):
                # Apply list comprehension only to GT reasoning fields, not model outputs
                conclusion_extracted[k] = [i.split('->')[-1].strip() for i in v]
            else:
                conclusion_extracted[k] = v
        simplified_gt_with_result[i] = conclusion_extracted    

    return simplified_gt_with_result

def gpt_eval_user_input(reasoning_type, reasoning_conclusions, model_output, debug=False):
    """
        GPT Evaluation for a specific reasoning type (w/o Event Relationship Reasoning)
    """
    reasoning = ""
    c = 0
    reasoning_score_dict = {reasoning_type: {}}

    for conclusion in reasoning_conclusions:
        c += 1
        conclusion_text = "{}. {}\n".format(c, conclusion)
        reasoning_score_dict[reasoning_type][conclusion_text.strip()] = []
        reasoning += conclusion_text

    if _is_empty_model_output(model_output):
        prompt = f"[SKIPPING API CALL] no model output for {reasoning_type}, directly assign score 0 for all conclusions."
        if debug:
            print(prompt)
    else:
        user_template = "<DESCRIPTION>: \n{}\n\n<KEY POINT>: \n{}\n".format(model_output, reasoning)
        gpt_output_template = "Please write your answers in '[ ]' with 0 or 1 in the following format (number + square brackets):\n1. [1]  2. [0]\nYour answers to the {} <KEY POINT>(s) above:\n".format(c)
        
        for i in range(1,c+1):
            gpt_output_template += "{}. [ ]  ".format(i)
            
        prompt = user_template + gpt_output_template
        
    return prompt, reasoning_score_dict

def gpt_eval_er_user_input(reasoning_conclusions, model_output, debug=False):
    """
        GPT Evaluation for Event Relationship Reasoning
    """
    c = 0
    reasoning = ""
    reasoning_score_dict = {'Event Relationship Reasoning':{}}
    
    for conclusion in reasoning_conclusions:
        c += 1
        conclusion_text = "{}. {}\n".format(c, conclusion)  
        reasoning_score_dict['Event Relationship Reasoning'][conclusion_text.strip()] = []
        reasoning += conclusion_text

    if _is_empty_model_output(model_output):
        prompt = f"[SKIPPING API CALL] no model output for Event Relationship Reasoning, directly assign score 0 for all conclusions."
        if debug:
            print(prompt)
    else:
        user_template = "<DESCRIPTION>: \n{}\n\n<EVENT RELATIONSHIP>: \n{}\n".format(model_output, reasoning)
        gpt_output_template = "Please write your answers in '[ ]' with 0 or 1 in the following format (number + square brackets):\n1. [1]  2. [0]\nYour answers to the {} <EVENT RELATIONSHIP>(s) above:\n".format(c)

        for i in range(1,c+1):
            gpt_output_template += "{}. [ ]  ".format(i)

        prompt = user_template + gpt_output_template

    return prompt, reasoning_score_dict

def chat_gpt_evaluation(model_config, system_prompt, user_input):
    
    global count
    try:
        client = genai.Client(api_key=model_config['key'])
        
        response = client.models.generate_content(
            model=model_config['name'],
            config=genai.types.GenerateContentConfig(
                system_instruction=system_prompt
            ),
            contents=user_input
        )
    except Exception as e:
        msg = (
            f"Gemini API call failed (exception raised).\n"
            f"error={type(e).__name__}: {e}"
        )
        raise GeminiAPIError(msg) from e

    if response is None:
        raise GeminiEmptyTextResponse(
            f"Gemini returned None response object.\nmodel={model_config.get('name')!r}"
        )

    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        count += 1
        return text.strip()
        
    block_reason = getattr(getattr(response, "prompt_feedback", None), "block_reason", None)
    if block_reason is not None:
        raise GeminiBlockedResponse(
            "Gemini returned no text because the response was blocked.\n"
            f"block_reason={block_reason!r}\n"
        )

    # Otherwise: not explicitly blocked, but still no usable text
    raise GeminiEmptyTextResponse(
        "Gemini returned a response object but no usable response.text.\n"
        f"raw_response={response!r}\n"
        f"raw_text_field={text!r}"
    )
        

def evaluator(eval_model, eval_system, eval_er_system, gt_with_output, output_file, debug=False):
    reasoning_types = ['Special Time Reasoning', 'Location Reasoning', 'Character Reasoning', 
                      'Character Relationship Reasoning', 'Event Reasoning', 
                      'Event Relationship Reasoning', 'Next Moment Event Reasoning', 'Mental State Reasoning']
    
    for fn, sample in tqdm(gt_with_output.items()): 
        merged_score_dict = {}
        img_name = sample['Image Name']
        
        if debug:
            print(f"\n=== Evaluating {img_name} ===")
        
        # Evaluate each reasoning type independently with its corresponding output
        for reasoning_type in reasoning_types:
            if reasoning_type not in sample:
                continue
            
            # Get the model output corresponding to this reasoning type
            output_field = f'{reasoning_type}_output'
            if output_field not in sample:
                print(f"Warning: {output_field} not found in sample for {img_name}")
                continue
            
            model_output_for_reasoning = sample[output_field]
            reasoning_conclusions = sample[reasoning_type]
            
            if not reasoning_conclusions:  # Skip if no conclusions
                continue
            
            # Special handling for Event Relationship Reasoning
            if reasoning_type == 'Event Relationship Reasoning':
                prompt, score_dict = gpt_eval_er_user_input(reasoning_conclusions, model_output_for_reasoning, debug=debug)
            else:
                prompt, score_dict = gpt_eval_user_input(reasoning_type, reasoning_conclusions, model_output_for_reasoning, debug=debug)

            if debug:
                print(f"\n[DEBUG] {reasoning_type} model output for {img_name}: {model_output_for_reasoning}")
                print(f"\n[DEBUG] {reasoning_type} conclusions for {img_name}: {reasoning_conclusions}")
                print(f"\n[DEBUG] {reasoning_type} prompt for {img_name}: {prompt}")
                print(f"\n[DEBUG] {reasoning_type} initial score_dict for {img_name}: {score_dict}")

            # if model output is empty or None, skip API call and directly assign score 0 for all conclusions
            if _is_empty_model_output(model_output_for_reasoning):
                for i in score_dict:
                    for j in score_dict[i]:
                        score_dict[i][j] = 0
                merged_score_dict.update(score_dict)
                continue
            
            max_attempts = 5
            attempts = 0
            
            while attempts < max_attempts:
                try:
                    if reasoning_type == 'Event Relationship Reasoning':
                        eval_output = chat_gpt_evaluation(eval_model, eval_er_system, prompt)
                    else:
                        eval_output = chat_gpt_evaluation(eval_model, eval_system, prompt)
                except GeminiAPIError as e:
                    if _is_retryable_unavailable_error(e):
                        attempts += 1
                        if attempts == max_attempts:
                            print(f"Max attempts reached for {img_name}, {reasoning_type}. Error: {e}")
                            for i in score_dict:
                                for j in score_dict[i]:
                                    score_dict[i][j] = 0
                            eval_output = None
                            break
                        print(f"Transient API unavailability on attempt {attempts} for {img_name} in {reasoning_type}: {e}")
                        print("Waiting 1 second and retrying...")
                        time.sleep(1)
                        continue

                    print(f"\n[API ERROR] {img_name}, {reasoning_type}: {repr(e)}")
                    print("Assigning score 0 for all conclusions due to API error.")

                    for i in score_dict:
                        for j in score_dict[i]:
                            score_dict[i][j] = 0
                    eval_output = None
                    break
                except (GeminiBlockedResponse, GeminiEmptyTextResponse) as e:
                    print(f"\n[API ERROR] {img_name}, {reasoning_type}: {repr(e)}")
                    print("Assigning score 0 for all conclusions due to API error.")

                    for i in score_dict:
                        for j in score_dict[i]:
                            score_dict[i][j] = 0
                    eval_output = None
                    break
                
                try:
                    output_dict = output_parse(eval_output)
                    for i in score_dict:
                        for j in score_dict[i]:
                            score_dict[i][j] = output_dict[j.split('.')[0].strip()]
                    break  
                except Exception as e:
                    attempts += 1
                    if attempts == max_attempts:
                        print(f"Max attempts reached for {img_name}, {reasoning_type}. Error: {e}")
                        break
                    print(f"Parsing error on attempt {attempts} for {img_name} in {reasoning_type}, error: {e}.")
                    print(f"gemini output: {eval_output}, parsed: {output_dict}")
                    print("Retrying...")
            
            merged_score_dict.update(score_dict)
        
        # Remove the number prefix from keys
        for reasoning_type in merged_score_dict:
            new_dict = {}
            for key, value in merged_score_dict[reasoning_type].items():
                new_key = key[3:].strip() if key[0].isdigit() and key[1] == '.' else key
                new_dict[new_key] = value
            merged_score_dict[reasoning_type] = new_dict
        
        merged_score_dict['Image Name'] = img_name

        with open(output_file, "a+") as f:
            json.dump(merged_score_dict, f)
            f.write('\n')


def main(ann_json_file_path, model_output_file_path, eval_output_dir, model_name, debug=False):
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is required for cognition evaluation.")

    # chatgpt/gemini config
    model = {"name": model_name,"key": GEMINI_API_KEY}

    with open(ann_json_file_path, 'r') as json_file:
        ann = json.load(json_file)
    
    # system instruction
    system_dir = Path(__file__).resolve().parent / "system"
    eval_system = (system_dir / "eval_system_prompt_v2.txt").read_text(encoding="utf-8")
    eval_er_system = (system_dir / "eval_system_prompt_er_v2.txt").read_text(encoding="utf-8")

    model_output = read_jsonl(model_output_file_path)
    
    # Extract model info and determine output file path
    model_name = None
    model_scenario = None
    prompt_version = None
    if model_output and 'model' in model_output[0]:
        model_name = model_output[0]['model']
        model_scenario = model_output[0].get('scenario')
        prompt_version = model_output[0].get('prompt_version')
            
        if eval_output_dir:
            results_dir = Path(eval_output_dir)
        else:
            results_dir = Path(model_output_file_path).parent
        results_dir.mkdir(parents=True, exist_ok=True)
        if prompt_version:
            eval_output_file = str(results_dir / f"cognition_gpt_eval_{prompt_version}.jsonl")
        else:
            eval_output_file = str(results_dir / "cognition_gpt_eval.jsonl")
        
        # Write metadata to output file
        with open(eval_output_file, "w") as f:
            json.dump({"model": model_name, "scenario": model_scenario, "prompt_version": prompt_version}, f)
            f.write('\n')
        
        # Remove the metadata row from model_output for processing
        model_output = model_output[1:]
    
        gt_with_result = gt_result_merge(ann, model_output)
        simplified_gt_with_result = evaluation_data_format(gt_with_result)

        print("Starting evaluation at {}".format(datetime.datetime.now()))
        evaluator(model, eval_system, eval_er_system, simplified_gt_with_result, eval_output_file, debug=debug)
        print("Evaluation completed at {}".format(datetime.datetime.now()))
        print("Total API calls made: {}".format(count))
        print("Results saved to {}".format(eval_output_file))
        return eval_output_file
    else:
        print("No valid model output")
        print(f"Model output: {json.dumps(model_output, indent=2, ensure_ascii=False)}")
        sys.exit(1)


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
        "--eval_output_dir",
        type=str,
        default=None,
        help="Output directory. If default value is used, will auto-save to results/{model_scenario}/{model_name}/cognition_gpt_eval.jsonl",
    )
    parser.add_argument(
        "--gemini_name",
        type=str,
        default="gemini-2.5-flash",
        help="Gemini model name",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode with prints"
    )
    args = parser.parse_args()

    eval_output_file = main(args.cogbench_description_file_path, args.model_output_file_path, args.eval_output_dir, args.gemini_name, args.debug)
