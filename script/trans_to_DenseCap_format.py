import json

def convert_jsonl_to_json(jsonl_file, output_file):
    data_dict = {}
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line.strip())
            image_id = entry["image"].split('.')[0]  
            description = entry["description"].replace('\n', '') 
            if image_id not in data_dict:
                data_dict[image_id] = [description]
            else:
                data_dict[image_id].append(description)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, ensure_ascii=False)


jsonl_file = ""
output_file = ""
convert_jsonl_to_json(jsonl_file, output_file)
