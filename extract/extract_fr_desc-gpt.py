import argparse
import json
import re
import sys
import time
from tqdm import tqdm
import openai
import requests as requests


example1_description = """%%%DESCRIPTION%%%: This image captures a cozy Christmas scene. At the center of the image is a bed adorned with a **red satin bedspread** with **gold tassels** and a **gold pillow**. The bed is nestled within **red curtains** that have **gold trim**, adding to the festive atmosphere.\n\nOn the bed, there are **four teddy bears**, two of which are brown and the other two are white. They are arranged in a pile, creating a sense of warmth and comfort. Adding to this feeling is a **white cat** that is peacefully sleeping next to the teddy bears.\n\nThe bed is further decorated with **red and gold Christmas ornaments**, enhancing the holiday spirit. To the right of the bed stands a **Christmas tree** adorned with **gold ornaments**, while on the left side of the bed, there are **two red candlesticks**.\n\nOverall, this image exudes a warm and festive atmosphere, perfect for the holiday season."""
example1_response = "%%%RESPONSE%%%: red satin bedspread. gold tassels. gold pillow. red curtains. gold trim. four teddy bears. white cat. red and gold Christmas ornaments. Christmas tree. gold ornaments. two red candlesticks."

example2_description = """%%%DESCRIPTION%%%: The image depicts a lively scene in an ornately decorated room. At the center of the image stands a man with a long white beard, wearing a tall, cylindrical black hat with a flat top. He is dressed in a white shirt paired with a green tie and carries a black backpack. 
The room is bustling with people, some of whom are wearing green shirts, adding to the vibrant atmosphere. The room itself is richly adorned, featuring a gold ceiling and red columns that exude an air of grandeur. 
The man, despite being in the midst of the crowd, stands out due to his unique attire and commanding presence. The people are visible on all sides of him, suggesting that he is at the heart of this gathering."""
example2_response = "%%%RESPONSE%%%: man with a long white beard. tall cylindrical black hat. white shirt. green tie. black backpack. people wearing green shirts. gold ceiling. red columns."

example3_description = """%%%DESCRIPTION%%%: This image captures a charming scene in a bedroom. The main subject is a **black cat** standing on its hind legs on a **gray nightstand**. The cat, full of curiosity, is reaching its paw into a **white cup** adorned with blue and green designs.\n\nThe nightstand hosts a few other items: a **beige lamp**, a **white tissue box** with blue flowers, and a **pink jar of Vaseline**. Each item is neatly arranged on the nightstand, creating a harmonious tableau.\n\nIn the background, you can see a bed covered with a **white comforter** and adorned with a **yellow pillow**, adding a pop of color to the scene. The precise location of these objects and their relative positions contribute to the overall composition of the image, creating a snapshot of everyday life with a hint of feline mischief."""
example3_response = "%%%RESPONSE%%%: black cat. gray nightstand. white cup. beige lamp. white tissue box. blue flowers. pink jar of Vaseline. white comforter. yellow pillow."



def query_ChatGPT(question):
    """
    complete this part.
    """
    return res


def main(args):
    total_lines = sum(1 for _ in open(args.input_file_path))


    with open(args.input_file_path, 'r') as input_file:
        with open(args.output_file_path, 'a') as output_file:
            for i, line in enumerate(tqdm(input_file, total=total_lines, desc="Extract objects from the description")):
                if i < args.start_line:
                    continue
                if i >= args.end_line:
                    break
                json_obj = json.loads(line)
                image_name = json_obj.get("image")
                description = json_obj.get("description")

                extr_prompt = f"""\
                ###TASK DESCRIPTION###
                Your are a helpful entities extractor. Please help me to extract the OBJECTS mentioned in a description about an image. 
                ###ATTENTION###
                1. Only extract the descriptions of objects that are described with certainty. For example, in the sentence "there's a white car parked, perhaps belonging to one of the hotel guests," the "hotel guests" part is included within "perhaps," indicating uncertainty. Therefore, you only need extract "a white car" that is described with certainty. 
                2. Avoid extracting abstract or non-specific entities (such as "cozy atmosphere", "excitement", "sky view")!!! 
                3. Your response should strictly start with "%%%RESPONSE%%%:" and follow this format: "%%%RESPONSE%%%: obj1. obj2. obj3...."

                ###IN-CONTEXT EXAMPLES###
                Here are some examples for your reference.
            
                Example1:
                {example1_description}
                {example1_response}
                ---

                Example2:
                {example2_description}
                {example2_response}
                ---

                Example3:
                {example3_description}
                {example3_response}
                ---

                description: {description}
                response:"""

                response = query_ChatGPT(extr_prompt)

                extr_obj = response.split(". ")

                output_data = {
                    "image": image_name,
                    "extr_obj_fr_desc": extr_obj,
                    "description": description
                }
                output_file.write(json.dumps(output_data) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract objects from image descriptions using OpenAI ChatGPT.")
    parser.add_argument("--input_file_path", type=str, default="./annotation/tmp.jsonl", help="Path to input file containing image descriptions.")
    parser.add_argument("--output_file_path", type=str, default="./annotation/stage1/obj_extr_from_desc.jsonl", help="Path to output file to store extracted objects.")
    parser.add_argument("--start_line", type=int, default=0)
    parser.add_argument("--end_line", type=int, default=100000)
    args = parser.parse_args()
    main(args)

