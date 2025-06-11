"""
Generate preference dataset from cycle consistency scores.
"""
import os
import json
import argparse
from tqdm import tqdm
import re


def cleanup_text(text):
        return re.sub(r"\s+", " ", text).strip().replace("**", "").replace('"', '')

def dump_json(data, save_path):
    with open(save_path, 'w') as f:
        json.dump(data, f)

def split_and_save_data(dataset, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    train_data = dataset[:int(len(dataset)*0.9)]
    valid_data = dataset[int(len(dataset)*0.9):int(len(dataset)*0.95)]
    test_data = dataset[int(len(dataset)*0.95):]
    print("train: ", len(train_data))
    print("valid: ", len(valid_data))
    print("test: ", len(test_data))

    dump_json(train_data, os.path.join(save_path, "train_rankings.json"))
    dump_json(valid_data, os.path.join(save_path, "valid_rankings.json"))
    dump_json(test_data, os.path.join(save_path, "test_rankings.json"))
    print(f"Data saved to {save_path}")
    return 

def make_preference_data(all_data_files, cycle, score_key="dreamsim"):
    # create a list of dictionaries with keys "input", "generations", and "scores"
    # "input" is the input image or text
    # "generations" is a list of generations
    # "scores" is a list of cycle consistency scores 
    list_of_dicts = []
    num_data = len(all_data_files[0])

    if cycle == "i2t2i":
        for i in range(num_data): 
            generations = []
            scores = []
            image = all_data_files[0][i]["real_image"]
            for data_file in all_data_files:
                cleaned_text = cleanup_text(data_file[i]["fake_text"])
                score = data_file[i][score_key]  
                generations.append(cleaned_text)
                scores.append(score)

            list_of_dicts.append({
                "input": image,
                "generations": generations,
                "scores": scores
            })
    elif cycle == "t2i2t":
        for i in range(num_data): 
            generations = []
            scores = []
            prompt = all_data_files[0][i]["real_text"]
            for data_file in all_data_files:
                img_path = data_file[i]["fake_image"]
                score = data_file[i][score_key] 
                generations.append(img_path)
                scores.append(score)

            list_of_dicts.append({
                "input": prompt,
                "generations": generations,
                "scores": scores
            })
    else:
        raise NotImplementedError

    return list_of_dicts

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default=None, help="Path to the output folder containing cycle consistency generation results.")
    parser.add_argument("--dataset", type=str, default="DCI", help='Source of input data')
    parser.add_argument("--cycle", type=str, default=None, choices=['i2t2i', 't2i2t'], help="Which cycle to run?")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save the preference dataset.")
    args = parser.parse_args()

    # gather all generation results
    all_data_files = []
    output_path = os.path.join(args.output_path, args.dataset)
    for model_combo in os.listdir(output_path):
        filename = os.path.join(output_path, model_combo, args.cycle, "metrics.json")
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                data = json.load(f)
            all_data_files.append(data)

    # generate comparison pairs
    score_key = "dreamsim" if args.cycle == "i2t2i" else "sbert_score"
    preference_data = make_preference_data(all_data_files, args.cycle, score_key=score_key)

    # split train, valid, test and save dataset
    split_and_save_data(preference_data, args.save_path)