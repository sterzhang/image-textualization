import spacy
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import os

def count_pos(text, nlp):
    doc = nlp(text)
    pos_counts = {}
    for token in doc:
        pos = token.pos_
        if pos in pos_counts:
            pos_counts[pos] += 1
        else:
            pos_counts[pos] = 1
    return pos_counts

def process_file(input_file):
    original_data = []
    modified_data = []
    nlp = spacy.load("en_core_web_sm")

    with open(input_file, 'r', encoding='utf-8') as file:
        for line in tqdm(file, desc="Processing"):
            data = json.loads(line)
            original_description = data['original_description']
            modified_description = data['modified_description']
            original_counts = count_pos(original_description, nlp)
            modified_counts = count_pos(modified_description, nlp)

            original_counts_list.append(original_counts)
            modified_counts_list.append(modified_counts)

            for pos, count in original_counts.items():
                if pos in {'ADP', 'DET', 'NOUN', 'PUNCT', 'ADJ', 'VERB', 'PRON', 'CCONJ', 'AUX', 'ADV'}:
                    original_data.append({'POS': pos, 'Count': count, 'Type': 'Original'})
            for pos, count in modified_counts.items():
                if pos in {'ADP', 'DET', 'NOUN', 'PUNCT', 'ADJ', 'VERB', 'PRON', 'CCONJ', 'AUX', 'ADV'}:
                    modified_data.append({'POS': pos, 'Count': count, 'Type': 'Modified'})

    return original_data, modified_data

def save_cache(data, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(data, f)

def load_cache(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data


def plot_violinplot(original_data, modified_data, output_file):
    for item in original_data:
        if item['Type'] == 'Original':
            item['Type'] = 'MLLM'
    
    for item in modified_data:
        if item['Type'] == 'Modified':
            item['Type'] = 'IT'

    data = original_data + modified_data
    df = pd.DataFrame(data)
    pos_order = ['ADV','CCONJ','PRON','AUX','ADJ','VERB','PUNCT','ADP','DET','NOUN']

    plt.figure(figsize=(15, 13))
    
    sns.set_context("notebook", font_scale=2)
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.linewidth'] = 2.5  
    plt.rcParams['legend.edgecolor'] = 'black'  
    plt.rcParams['legend.frameon'] = True  
    plt.rcParams['legend.framealpha'] = 1  
    plt.rcParams['legend.fancybox'] = False  
    plt.rcParams['xtick.major.size'] = 10 
    plt.rcParams['xtick.major.width'] = 2  
    plt.rcParams['ytick.major.size'] = 10 
    plt.rcParams['ytick.major.width'] = 2  

    
    sns.violinplot(x='Count', y='POS', hue='Type', data=df, split=True, inner="quart", scale="count", palette={"MLLM": "#A0BDD5", "IT": "#CA8BA8"}, order=pos_order, linewidth=2.5)  
    
    # Add dashed lines
    for x in range(0, 100, 20):
        plt.axvline(x=x, color='gray', linestyle='--', linewidth=0.5)
    
    # Set x-axis limit
    plt.xlim(0, 105)

    plt.xlabel('Count', fontsize=27, fontweight='bold')
    plt.ylabel('POS Tag', fontsize=27, fontweight='bold')
    
    plt.legend(title='', title_fontsize='26', fontsize='26', frameon=True, edgecolor='black', framealpha=1, fancybox=False, borderpad=1, handletextpad=0.5, loc='upper right')
    legend = plt.legend(title='', title_fontsize='26', fontsize='26', frameon=True, edgecolor='black', framealpha=1, fancybox=False, borderpad=1, handletextpad=0.5, loc='upper right')

    legend.get_frame().set_linewidth(2.5)
    plt.savefig(output_file)
    plt.close()




def average_counts(counts_list):
    avg_counts = {}
    total_counts = len(counts_list)
    for counts in counts_list:
        for pos, count in counts.items():
            if pos in avg_counts:
                avg_counts[pos] += count
            else:
                avg_counts[pos] = count
    for pos in avg_counts:
        avg_counts[pos] /= total_counts
    return avg_counts

def detailed_comparison(original_data, modified_data):
    original_counts_list = []
    modified_counts_list = []

    for data in original_data:
        pos_counts = {}
        pos_counts[data['POS']] = data['Count']
        original_counts_list.append(pos_counts)
    
    for data in modified_data:
        pos_counts = {}
        pos_counts[data['POS']] = data['Count']
        modified_counts_list.append(pos_counts)
    
    avg_original_counts = average_counts(original_counts_list)
    avg_modified_counts = average_counts(modified_counts_list)

    print("\nDetailed Comparison:")
    print(f"{'POS':<15}{'Before':<20}{'After':<20}{'Change':<20}")
    
    pos_tags = set(avg_original_counts.keys()).union(set(avg_modified_counts.keys()))
    for pos in sorted(pos_tags):
        before = avg_original_counts.get(pos, 0)
        after = avg_modified_counts.get(pos, 0)
        change = after - before
        print(f"{pos:<15}{before:<20.2f}{after:<20.2f}{change:<20.2f}")

def display_comparison(before_counts, after_counts):
    # Collect all POS tags present in both before and after counts
    pos_to_compare = set(before_counts.keys()).union(set(after_counts.keys()))
    
    print("\nDetailed Comparison:")
    print(f"{'POS':<15}{'Before':<20}{'After':<20}{'Change':<20}")
    
    for pos in pos_to_compare:
        before = before_counts.get(pos, 0)
        after = after_counts.get(pos, 0)
        change = after - before
        print(f"{pos:<15}{before:<20.2f}{after:<20.2f}{change:<20.2f}")

if __name__ == '__main__':
    input_file = './jianshu/it-data/it-sam-65k.jsonl'
    original_cache_file = './benchmark/Count/cache/sam-ori.json'
    modified_cache_file = './benchmark/Count/cache/sam-modi.json'
    output_file = './bench_result/it-sam.png'

    original_counts_list = []
    modified_counts_list = []

    if os.path.exists(original_cache_file) and os.path.exists(modified_cache_file):
        original_data = load_cache(original_cache_file)
        modified_data = load_cache(modified_cache_file)
    else:
        original_data, modified_data = process_file(input_file)
        save_cache(original_data, original_cache_file)
        save_cache(modified_data, modified_cache_file)

    plot_violinplot(original_data, modified_data, output_file)
    
    avg_original_counts = average_counts(original_counts_list)
    avg_modified_counts = average_counts(modified_counts_list)
    
    print("\nAverage Part-of-Speech Counts (Before):")
    for pos, count in avg_original_counts.items():
        print(f"{pos}: {count:.2f}")

    print("\nAverage Part-of-Speech Counts (After):")
    for pos, count in avg_modified_counts.items():
        print(f"{pos}: {count:.2f}")

    display_comparison(avg_original_counts, avg_modified_counts)