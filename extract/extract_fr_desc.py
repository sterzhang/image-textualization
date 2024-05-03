import argparse
import json
import re
import sys
import time
from tqdm import tqdm
import openai
import requests as requests

"""
input_file ("image", "description")
output_file  
{
    "image": image_name,
    "extr_obj": extr_obj,
    "description": description
}
"""

example1_description = """description: This image captures a cozy Christmas scene. At the center of the image is a bed adorned with a **red satin bedspread** with **gold tassels** and a **gold pillow**. The bed is nestled within **red curtains** that have **gold trim**, adding to the festive atmosphere.\n\nOn the bed, there are **four teddy bears**, two of which are brown and the other two are white. They are arranged in a pile, creating a sense of warmth and comfort. Adding to this feeling is a **white cat** that is peacefully sleeping next to the teddy bears.\n\nThe bed is further decorated with **red and gold Christmas ornaments**, enhancing the holiday spirit. To the right of the bed stands a **Christmas tree** adorned with **gold ornaments**, while on the left side of the bed, there are **two red candlesticks**.\n\nOverall, this image exudes a warm and festive atmosphere, perfect for the holiday season."""
example1_response = "response: red satin bedspread. gold tassels. gold pillow. red curtains. gold trim. four teddy bears. white cat. red and gold Christmas ornaments. Christmas tree. gold ornaments. two red candlesticks."

example2_description = """description: The image depicts a lively scene in an ornately decorated room. At the center of the image stands a man with a long white beard, wearing a tall, cylindrical black hat with a flat top. He is dressed in a white shirt paired with a green tie and carries a black backpack. 
The room is bustling with people, some of whom are wearing green shirts, adding to the vibrant atmosphere. The room itself is richly adorned, featuring a gold ceiling and red columns that exude an air of grandeur. 
The man, despite being in the midst of the crowd, stands out due to his unique attire and commanding presence. The people are visible on all sides of him, suggesting that he is at the heart of this gathering."""
example2_response = "response: man with a long white beard. tall cylindrical black hat. white shirt. green tie. black backpack. people wearing green shirts. gold ceiling. red columns."

example3_description = """description: This image captures a charming scene in a bedroom. The main subject is a **black cat** standing on its hind legs on a **gray nightstand**. The cat, full of curiosity, is reaching its paw into a **white cup** adorned with blue and green designs.\n\nThe nightstand hosts a few other items: a **beige lamp**, a **white tissue box** with blue flowers, and a **pink jar of Vaseline**. Each item is neatly arranged on the nightstand, creating a harmonious tableau.\n\nIn the background, you can see a bed covered with a **white comforter** and adorned with a **yellow pillow**, adding a pop of color to the scene. The precise location of these objects and their relative positions contribute to the overall composition of the image, creating a snapshot of everyday life with a hint of feline mischief."""
example3_response = "response: black cat. gray nightstand. white cup. beige lamp. white tissue box. blue flowers. pink jar of Vaseline. white comforter. yellow pillow."


def load_api_keys_from_txt(key_path):
    key_list = []
    with open(key_path, 'r') as file:
        for line in file:
            matches = re.findall(r'sk-\w+', line)
            key_list += matches
    return key_list

def query_ChatGPT(input_data, count, code_api_list):
    response = None
    received = False
    while not received:
        try:
            openai.api_key = code_api_list[count % len(code_api_list)]

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0613",
                messages=input_data,
                temperature=1.0,
            )
            received = True
        except openai.error.InvalidRequestError as e:
            print(f"InvalidRequestError\nPrompt passed in:\n\n{input_data}\n\n")
            raise e
        except openai.error.APIError as e:
            print("API error:", e)
            print(code_api_list[count % len(code_api_list)])
            count += 1
            # time.sleep(2)
    return response, count

def query_ChatGPT_free(question):
    # put your key at here
    api_key = ""
    headers = {
        "Authorization": 'Bearer ' + api_key,
    }
    params = {
        "messages": [
            {
                "role": 'user',
                "content": question
            }
        ],
        "model": 'gpt-3.5-turbo'
    }
    response = requests.post(
        "https://aigptx.top/v1/chat/completions",
        headers=headers,
        json=params,
        stream=False
    )
    res = response.json()
    res_content = res['choices'][0]['message']['content']
    return res


def main(args):
    total_lines = sum(1 for _ in open(args.input_file_path))

    with open(args.api_key_path, 'r') as key_file:
        code_api_list = load_api_keys_from_txt(args.api_key_path)

    with open(args.input_file_path, 'r') as input_file:
        with open(args.output_file_path, 'a') as output_file:
            for i, line in enumerate(tqdm(input_file, total=total_lines, desc="Stage1.2 - extract objects from the description")):
                if i < args.start_line:
                    continue  # Skip lines until the start_line is reached
                json_obj = json.loads(line)
                image_name = json_obj.get("image")
                description = json_obj.get("description")

                extr_prompt = f"""\
                Your are a helpful entities extractor! Please help me to extract common entities in the given description of an image. Remember, you should only help me extract common entities that truly correspond to the objects in the image. Avoid extracting abstract or non-specific entities (such as "cozy atmosphere", "excitement", "sky view")!!! Your response should strictly follow below format:

                obj1. obj2. obj3 ….

                Here are some examples for your reference.
            
                example1:
                {example1_description}
                {example1_response}
                ---

                example2:
                {example2_description}
                {example2_response}
                ---

                example3:
                {example3_description}
                {example3_response}
                ---

                description: {description}
                response:"""

                response = query_ChatGPT_free(extr_prompt)
                time.sleep(0.5)

                extr_obj = response.split(". ")

                output_data = {
                    "image": image_name,
                    "extr_obj": extr_obj,
                    "description": description
                }
                output_file.write(json.dumps(output_data) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract objects from image descriptions using OpenAI ChatGPT.")
    parser.add_argument("--input_file_path", type=str, default="./annotation/tmp.jsonl", help="Path to input file containing image descriptions.")
    parser.add_argument("--output_file_path", type=str, default="./annotation/stage1/obj_extr_from_desc.jsonl", help="Path to output file to store extracted objects.")
    parser.add_argument("--api_key_path", type=str, default="api_key.txt", help="Path to file containing OpenAI API keys.")
    parser.add_argument("--start_line", type=int, default=0, help="Start reading the input file from this line number (if fail in line x, then set this value as x to continue generate).")

    args = parser.parse_args()
    main(args)


"""
python stage1/obj_extract_from_desc.py --input_file_path /home/zhangjianshu/Mercury/llava/llava_stage0_1.jsonl \
    --output_file_path /home/zhangjianshu/Mercury/llava/llava_stage1_desc.jsonl \
    --api_key_path api_key.txt \
    --start_line 0
"""