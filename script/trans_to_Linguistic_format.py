import json

# Define the file paths
modi_file_path = ''
unmodi_file_path = ''
output_file_path = ''


with open(modi_file_path, 'r', encoding='utf-8') as modi_file:
    modi_lines = [json.loads(line) for line in modi_file]

with open(unmodi_file_path, 'r', encoding='utf-8') as unmodi_file:
    unmodi_lines = [json.loads(line) for line in unmodi_file]

if len(modi_lines) != len(unmodi_lines):
    raise ValueError("The number of lines in modi_gpt.jsonl and unmodi_gpt.jsonl do not match.")

output_lines = []
for modi_line, unmodi_line in zip(modi_lines, unmodi_lines):
    output_lines.append({
        "modi_desc": modi_line.get('description'),
        "unmodi_desc": unmodi_line.get('description')
    })

with open(output_file_path, 'w', encoding='utf-8') as output_file:
    for line in output_lines:
        output_file.write(json.dumps(line) + '\n')


