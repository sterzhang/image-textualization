import argparse
import json
import re
import sys
import time
from tqdm import tqdm
import openai
import requests as requests


additional_info = """You are a helpful language and visual assitant. Imagine visualizing an image based on its written description.  Now, your task is to add some missing objects to this description to make it more detailed. Objects will be given below. Your must integrate all these objects by using the additional information we provide and your logical reasoning ability. The additional information contains each object’s relative postion and depth of focus, which can help you appropriately add objects to the right place in the description. Below are the detailed meaning of relative postion and depth of focus:

1. relative position: Relative position within an image utilizes a normalized coordinate system, where both the x (horizontal) and y (vertical) axes range from 0 to 1. This system defines the corners of a rectangle within the image by specifying the coordinates of its upper-left and lower-right vertices in the format **`[x1, y1, x2, y2]`**.For example, **`[0.00, 0.00, 0.50, 0.50]`** indicates the upper-left quarter of the image, spanning from the left to the middle horizontally and from the top to the middle vertically. Similarly, **`[0.50, 0.00, 1.00, 0.50]`**, **`[0.00, 0.50, 0.50, 1.00]`**, and **`[0.50, 0.50, 1.00, 1.00]`** denote the upper-right, lower-left, and lower-right quarters, respectively, allowing for precise object placement based on their coordinates. Additionally, you can tell the relation of postion between two objects if their values of depth of focus are close, like [0.20, 0.20, 0.40, 0.40] is in the left of [0.30, 0.30, 0.50, 0.50]. 
2. depth of focus: the distance that the object is from the camera. The larger the value, the closer it is to the camera. If two objects have the same depth value, it means they are in the same depth within the image. If the depth of focus is close to 0, it means the corresponding objects is in the relative background of the image, probably out of focus. On the contrary, if the depth of focus is close to 1, it means the corresponding object is right in the focus. Additionally, if two objects’ values are pretty close, for example, one is 0.12 and the other is 0.02, then it means they might on the same layer, both in the background. Moreover, if one relative position’s space fully covers the other relative position’s space and their depth of focus are close, it probably means these two objects have some relationships. For example, if there is a baseball player in blue helmet with relative position [0.42, 0.38, 0.59, 0.87] and depth of focus is 0.63, and a white baseball glove with relative position [0.56, 0.54, 0.58, 0.58] and depth of focus is 0.58, it probably means this baseball player in blue helmet is wearing a white baseball glove.

The original image description provides an overall view of the image, but may miss some details. If the object already correctly exists in the description, then do not modify it; if the object is missing, insert it into the appropriate part of the description using your logical reasoning ability. You need to mention all the given objects in your modified image description. Remember the additonal information of the relative postion and the depth of focus should not directly mentioned in the description. Again, the addition information of the relative postion and depth of focus should not be directly mentioned in the description!!! For example, your modified response shouldn't contain 'The clock, with a depth of focus of 0.89' or 'including a brown cow laying on the beach [0.0, 0.26, 0.16, 0.58]' such sentences.

Here are some examples for you to reference:
object1: a wooden chair with rods
relative position: [0.25, 0.85, 0.49, 1.0]
depth of focus: 1.0

object2: white refrigerator in the kitchen
relative position: [0.67, 0.36, 0.83, 0.65]
depth of focus: 0.29

object3: a white microwave oven
relative position: [0.59, 0.47, 0.67, 0.54]
depth of focus: 0.03

object4: a yellow bottle of mustard
relative position: [0.72, 0.59, 0.74, 0.66]
depth of focus: 1.0

object5: a clear plastic soap dispenser
relative position: [0.68, 0.58, 0.71, 0.65]
depth of focus: 0.64

object6: a brown ceramic cup
relative position: [0.46, 0.57, 0.53, 0.66]
depth of focus: 0.97

object7: a white coffee mug
relative position: [0.27, 0.48, 0.31, 0.52]
depth of focus: 0.29

object8: a small green plant
relative position: [0.09, 0.53, 0.14, 0.65]
depth of focus: 0.67

object9: blue pot on stove
relative position: [0.32, 0.54, 0.36, 0.59]
depth of focus: 0.67

object10: a white coffee mug
relative position: [0.3, 0.48, 0.34, 0.51]
depth of focus: 0.21

object11: a brown wooden table
relative position: [0.0, 0.83, 0.24, 1.0]
depth of focus: 0.95

object12: a jar of yellow mustard
relative position: [0.62, 0.34, 0.63, 0.37]
depth of focus: 0.0

The original image description:  This is a well-organized kitchen with a clean, modern aesthetic. The kitchen features a white countertop against a white wall, creating a bright and airy atmosphere.

On the countertop, you can see a variety of appliances and items. There's a sleek coffee machine, ready to brew a fresh cup. Next to it is a paper towel holder, standing tall and at the ready for any spills. A vase adds a touch of elegance to the space, while a blender suggests the possibility of smoothies or soups being made here. Various bottles and jars are also present, perhaps containing spices or cooking ingredients.

The objects are mostly colored in shades of white, black, and silver, complementing the modern look of the kitchen. However, there are also pops of color with some yellow and blue objects, adding a bit of cheerfulness to the space.

Above the countertop, shelves hold additional items. The arrangement is neat and everything appears to have its own place.

In the foreground of the image is a wooden chair, perhaps providing a spot to sit while waiting for that coffee to brew or the blender to finish its job. In the background, there's a window letting in natural light.

Overall, this kitchen is not only functional with its various appliances and ample storage, but also stylish with its color scheme and neat arrangement.

Modified image description: This is a well-organized kitchen with a clean, modern aesthetic. The kitchen features a white countertop against a white wall, creating a bright and airy atmosphere.

On the countertop, you can see a variety of appliances and items. There's a sleek coffee machine, ready to brew a fresh cup, placed beside a paper towel holder, standing tall and at the ready for any spills. A vase adds a touch of elegance to the space, while a blender suggests the possibility of smoothies or soups being made here. Various bottles and jars are also present, perhaps containing spices or cooking ingredients. Among these, a yellow bottle of mustard and a jar of yellow mustard catch the eye, adding pops of color to the predominantly white and silver theme.

Next to the countertop, there's a white refrigerator in the kitchen, occupying a significant portion of the space. It stands out with its pristine white finish, serving as a focal point in the room. Nearby, a white microwave oven sits, ready for quick heating or cooking tasks.

On the stovetop, a blue pot catches the light, hinting at something delicious simmering inside. Nearby, a small green plant adds a touch of nature to the scene, providing a refreshing contrast to the sterile kitchen environment.

Moving towards the foreground, a wooden chair with rods stands, providing a cozy spot to sit and enjoy the culinary creations from this well-appointed kitchen. Beside it, a brown ceramic cup and a white coffee mug are placed, suggesting a recent pause for a hot beverage. Additionally, another white coffee mug is seen nearby, indicating the presence of multiple occupants or frequent coffee breaks.

In the background, a window bathes the room in natural light, enhancing the overall ambiance of the space.

Overall, this kitchen seamlessly blends functionality with style, offering a visually pleasing and efficient environment for culinary endeavors.

----
objects 1: a light green toilet
relative position: [51, 175, 158, 325]
depth of focus: 0.52

objects 2: white toilet in bathroom
relative position: [307, 213, 411, 332]
depth of focus: 0.98

objects 3: a clear jar with a lid
relative position: [318, 123, 338, 161]
depth of focus: 0.69

objects 4: a towel on a rack
relative position: [0, 151, 34, 281]
depth of focus: 0.67

objects 5: jar with white napkins
relative position: [341, 122, 360, 157]
depth of focus: 0.63

objects 6: dark brown cabinet on the wall
relative position: [281, 0, 390, 123]
depth of focus: 0.00

objects 7: silver metal bathroom trash can
relative position: [283, 290, 324, 332]
depth of focus: 0.83

objects 8: small white towel on a towel rack
relative position: [253, 177, 298, 290]
depth of focus: 0.91

objects 9: metal art on the wall
relative position: [408, 0, 449, 259]
depth of focus: 0.46

objects 10: a silver trash can
relative position: [0, 301, 34, 332]
depth of focus: 1.00

objects 11: a light blue toilet seat
relative position: [87, 233, 158, 276]
depth of focus: 0.67

objects 12:  on the shelf
relative position: [298, 142, 314, 160]
depth of focu: 0.74

The original image description: The image presents a striking contrast between two distinct bathroom designs. On the left, we see a bathroom that exudes a cool, calming aura with its blue upper wall and green lower wall. The toilet, in harmony with the lower wall, is also green. A white towel, neatly hung on the wall, adds a touch of cleanliness and order to the space.\n\nOn the right, we are introduced to a bathroom that embraces a warmer, more vibrant color scheme. The wall is painted a lively shade of orange and is adorned with a tree decal, adding an element of nature to the indoor space. A black cabinet and a white toilet form the main fixtures in this bathroom. Above the cabinet, a black shelf holds jars and a plant, further enhancing the natural theme. A window opens up to a view of trees, connecting the indoors with the outdoors and creating an open, airy atmosphere.\n\nIn both bathrooms, despite their contrasting designs and color schemes, there is an evident emphasis on cleanliness, orderliness, and harmony with nature. Each design offers its own unique interpretation of comfort and style.

Modified image description: The image presents a striking contrast between two distinct bathroom designs, each embodying a unique aesthetic and color palette that offers its own interpretation of comfort and style, meticulously detailed with various objects that add depth and character to the spaces.

On the left, the bathroom exudes a cool, calming aura with its blue upper wall and green lower wall, complemented by a light green toilet that harmonizes with the lower wall. A white towel, neatly hung on a towel rack, and a silver trash can positioned discreetly near the toilet, add touches of cleanliness and order to the space. Additionally, a light blue toilet seat accentuates the toilet's design, further integrating the bathroom's color scheme.

On the right, the bathroom embraces a warmer, more vibrant theme with its lively orange wall, decorated with a tree decal that introduces an element of nature. The main fixtures, a white toilet and a dark brown cabinet on the wall, anchor the design. Above the cabinet, a black shelf hosts an assortment of objects, including a clear jar with a lid, a jar with white napkins, and a candle, enhancing the natural and cozy theme. A small white towel on a towel rack, a silver metal bathroom trash can beneath it, and metal art on the wall contribute to the aesthetic, while a window offers views of the outdoors, bridging the indoor space with nature.

In both bathrooms, the emphasis on cleanliness, orderliness, and a harmonious relationship with nature is evident. Each design, with its contrasting color schemes and thoughtful placement of everyday items, reflects a unique perspective on creating a comfortable and stylish bathroom environment.
"""


def load_api_keys_from_txt(key_path):
    key_list = []
    with open(key_path, 'r') as file:
        for line in file:
            matches = re.findall(r'sk-\w+', line)
            key_list += matches
    return key_list

def query_ChatGPT_free(question):
    api_key = "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"
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
    return res

def main(args):
    total_lines = sum(1 for _ in open(args.input_file, 'r'))

    with open(args.input_file, 'r') as input_file, open(args.output_file, 'a') as output_file:
        for i, line in enumerate(tqdm(input_file, total=total_lines, desc="Adding objects to the description")):
            if i < args.start_line:
                continue
            output_str = ""  
            json_obj = json.loads(line)
            objects = json_obj.get("objects", [])
            boxes = json_obj.get("bounding_boxes")
            obj_depths = json_obj.get("object_depth")
            sizes = json_obj.get("size")
            width = json_obj.get("width")
            height = json_obj.get("height")
            description = json_obj.get("description")

            add_obj_num = len(objects)
            del_obj_num = len(del_obj_from_desc)

            if add_obj_num < 4:
                modified_description = description
            else:
                norm_boxes = []
                min_depth = float('inf')  
                max_depth = float('-inf')  
                for i in range(add_obj_num):
                    x1_norm = round(boxes[i][0] / width, 2)
                    y1_norm = round(boxes[i][1] / height, 2)
                    x2_norm = round(boxes[i][2] / width, 2)
                    y2_norm = round(boxes[i][3] / height, 2)
                    norm_boxes.append((x1_norm, y1_norm, x2_norm, y2_norm))

                    # Update min_depth and max_depth
                    if obj_depths[i] < min_depth:
                        min_depth = obj_depths[i]
                    if obj_depths[i] > max_depth:
                        max_depth = obj_depths[i]

                # Normalize depth
                for i in range(add_obj_num):
                    depth_norm = (obj_depths[i] - min_depth) / (max_depth - min_depth)
                    output_str += f"object{i+1}: {objects[i]}\n"
                    output_str += f"relative position: {[norm_boxes[i][0], norm_boxes[i][1], norm_boxes[i][2], norm_boxes[i][3]]}\n"
                    output_str += f"depth of focus: {round(depth_norm,2)}\n\n"
                
                output_str += "The original image description: " + description + "\n\n"
                output_str += "Modified image description:\n"

                prompt = additional_info + "\n\n" + output_str
                
                response = query_ChatGPT_free(prompt)
                time.sleep(0.3)
                modified_description = response["choices"][0]["message"]["content"]
            
            # Write to output file
            json_obj["original_description"] = description
            json_obj["modified_description"] = modified_description
            output_file.write(json.dumps(json_obj) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process image descriptions.")
    parser.add_argument("--input_file", type=str, help="Path to the input JSON file.")
    parser.add_argument("--output_file", type=str, help="Path to the output JSONL file.")
    parser.add_argument("--start_line", type=int, default=0, help="Start line in the input file.")
    args = parser.parse_args()
    main(args)

"""
python modify/add_detail.py --input_file /home/zhangjianshu/Mercury/llava/llava_stage3.jsonl \
--output_file /home/zhangjianshu/Mercury/llava/llava_stage4.jsonl \
--start_line 0
"""