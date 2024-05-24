#!/usr/bin/env python3

import json
import math
import re
import argparse
import os
from tqdm import tqdm

def chars(text):
    return len(re.sub(r'\s+', '', text))

def sentences(text):
    return len(re.split(r'[.?!]', text)) - 1

def words(text):
    words_array = re.findall(r'\b\w+\b', text)
    return len(words_array)

def syllables(word):
    word = word.lower()
    word = re.sub(r'[^a-z]', '', word)
    syllable_count = len(re.findall(r'[aeiouy]+', word))
    if word.endswith('e'):
        syllable_count -= 1
    if syllable_count == 0:
        syllable_count = 1
    return syllable_count

def total_syllables(text):
    words_array = re.findall(r'\b\w+\b', text)
    count_syllables = sum(syllables(word) for word in words_array)
    return count_syllables

def polysyllables(text):
    words_array = re.findall(r'\b\w+\b', text)
    count_polysyllables = sum(1 for word in words_array if syllables(word) > 2)
    return count_polysyllables

def ari(text):
    try:
        return 4.71 * (chars(text) / words(text)) + 0.5 * (words(text) / sentences(text)) - 21.43
    except ZeroDivisionError:
        return None

def fk(text):
    try:
        return 0.39 * (words(text) / sentences(text)) + 11.8 * (total_syllables(text) / words(text)) - 15.59
    except ZeroDivisionError:
        return None

def smog(text):
    try:
        return 1.043 * math.sqrt(polysyllables(text) * (30 / sentences(text))) + 3.1291
    except ZeroDivisionError:
        return None

def cl(text):
    try:
        return 0.0588 * (chars(text) / words(text) * 100) - 0.296 * (sentences(text) / words(text) * 100) - 15.8
    except ZeroDivisionError:
        return None

def calculate_readability_scores(text):
    return {
        "ARI": ari(text),
        "FK": fk(text),
        "SMOG": smog(text),
        "CL": cl(text)
    }

def main():
    parser = argparse.ArgumentParser(description='Calculate readability scores for JSONL file.')
    parser.add_argument('--file_path', type=str, help='Path to the input JSONL file', required=True)
    parser.add_argument('--result_file_path', type=str, help='Path to the output result file', required=True)
    parser.add_argument('--start_line', type=int, default=0, help='Start line number', required=True)
    parser.add_argument('--end_line', type=int, default=0, help='End line number', required=True)
    args = parser.parse_args()

    if os.path.exists(args.result_file_path):
        with open(args.result_file_path, 'r', encoding='utf-8') as result_file:
            results = json.load(result_file)
            original_scores = results['original_scores']
            modified_scores = results['modified_scores']
    else:
        original_scores = {"ARI": [], "FK": [], "SMOG": [], "CL": []}
        modified_scores = {"ARI": [], "FK": [], "SMOG": [], "CL": []}

        with open(args.file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            selected_lines = lines[args.start_line:args.end_line + 1]
            for line in tqdm(selected_lines, desc="Processing", unit="line"):
                data = json.loads(line)
                original_description = data['unmodi_desc']
                modified_description = data['modi_desc']

                original_readability = calculate_readability_scores(original_description)
                modified_readability = calculate_readability_scores(modified_description)

                for key in original_scores:
                    if original_readability[key] is not None:
                        original_scores[key].append(original_readability[key])
                    if modified_readability[key] is not None:
                        modified_scores[key].append(modified_readability[key])

        results = {
            "original_scores": original_scores,
            "modified_scores": modified_scores
        }

        with open(args.result_file_path, 'w', encoding='utf-8') as result_file:
            json.dump(results, result_file, ensure_ascii=False, indent=4)


    def average_scores(scores):
        return {key: sum(values) / len(values) for key, values in scores.items() if values}


    original_avg_scores = average_scores(original_scores)
    modified_avg_scores = average_scores(modified_scores)


    print("Original Descriptions Average Readability Scores:")
    for key, value in original_avg_scores.items():
        print(f"{key}: {value:.2f}")

    print("\nModified Descriptions Average Readability Scores:")
    for key, value in modified_avg_scores.items():
        print(f"{key}: {value:.2f}")

if __name__ == "__main__":
    main()

"""
python /home/zhangjianshu/Mercury/benchmark/Linguistic/readability.py --file_path /home/zhangjianshu/Mercury/bench_result/LinBench_gpt-gt_101_llava.jsonl --result_file_path /home/zhangjianshu/Mercury/bench_result/LinBench_record.jsonl --start_line 0 --end_line 101
"""
