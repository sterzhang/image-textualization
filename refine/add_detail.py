from vllm import LLM, SamplingParams
import argparse
import json
import re
import sys
import time
from tqdm import tqdm


additional_info = """
###TASK Description###
You are a helpful language assitant. Imagine visualizing an image based on its description. Now, your task is to make the Original Description more detailed. You'll need to use the subsequent provided Objects, along with the corresponding extra information of Relative Spatial Positioning, Relative Distance from the Lens, and Relative Size Proportion in Images (Percentage), to better assist you in adding these Objects to the original description. 

###Extra Information Explanation###
1. Relative Spatial Positioning: 
It uses a normalized coordinate system where both x (horizontal) and y (vertical) axes range from 0 to 1. The x-coordinate starts at 0 on the image's left edge and increases to 1 towards the right edge. Similarly, the y-coordinate starts at 0 at the top edge and increases to 1 towards the bottom. This system uses four coordinates to define the corners of a rectangle within the image: [x1, y1, x2, y2], representing the top-left and bottom-right corners of the rectangle, respectively.
For instance, a positioning of [0.00, 0.00, 0.50, 0.50] means the object's top-left corner is at (0.00, 0.00) and its bottom-right corner is at (0.50, 0.50), placing the object in the upper left quarter of the image. Similarly, [0.50, 0.00, 1.00, 0.50] positions the object in the upper right quarter, with corners at (0.50, 0.00) and (1.00, 0.50). A positioning of [0.00, 0.50, 0.50, 1.00] places the object in the bottom left quarter, with corners at (0.00, 0.50) and (0.50, 1.00), while [0.50, 0.50, 1.00, 1.00] positions it in the bottom right quarter, with corners at (0.50, 0.50) and (1.00, 1.00).
Moreover, by comparing these coordinates, you can determine the relative positions of objects. For example, an object with positioning [0.20, 0.20, 0.40, 0.40] is to the left of another with [0.30, 0.30, 0.50, 0.50].

2. Relative Distance from the Lens: 
It measures how far objects are from the camera within the image. The closer the value is to 1, the nearer the object is to the camera, placing it in the foreground. Conversely, a value close to 0 indicates that the object is far from the camera, situated in the background.
If two objects share the same Object Distance from the Lens value, it suggests they are at the same depth or layer within the image. For instance, if one object has a value of 0.12 and another is at 0.05, these small differences suggest that both objects are likely in the background, even if not exactly at the same depth but relatively close. This helps in understanding the spatial arrangement of elements in the image and how they relate to the viewer's perspective.

3. Relative Size Proportion in Images (Percentage): 
It can tell you the approximate proportion of the object within the image. If the proportion is particularly small or large, it should be emphasized. If the proportion is moderate, it does not need to be highlighted.

###Remember###
1. Through the extra information of different Objects, some Objects may represent the same thing. When adding Objects to the Original Description, it is important to avoid duplication.
2. The photographic characteristics mentioned in the Original Description should be preserved, such as sizes, locations, camera angles, depths;
3. The Relative Spatial Positioning, Relative Distance from the Lens, and Relative Size Proportion in Images of each Object need to be emphasized. However, these details must be expressed without directly using specific values of the extra information, you must convey this information naturally through logical reasoning.

###In-context Examples###
[Chain of thought is placed within a pair of "@@@"  (remember only in the Examples will you be provided with a chain of thoughts to help you understand; in the actual task, these will not be given to you.)]

$$$Example 1$$$
%%%The Original Description:%%% In the center of the image, a vibrant blue lunch tray holds four containers, each brimming with a variety of food items. The containers, two in pink and two in yellow, are arranged in a 2x2 grid.\n\nIn the top left pink container, a slice of bread rests, lightly spread with butter and sprinkled with a handful of almonds. The bread is cut into a rectangle, and the almonds are scattered across its buttery surface.\n\nAdjacent to it in the top right corner, another pink container houses a mix of fruit. Sliced apples with their fresh white interiors exposed share the space with juicy chunks of pineapple. The colors of the apple slices and pineapple chunks contrast beautifully against the pink container.\n\nBelow these, in the bottom left corner of the tray, a yellow container holds a single meatball alongside some broccoli. The meatball, round and browned, sits next to the vibrant green broccoli florets.\n\nFinally, in the bottom right yellow container, there's a sweet treat - a chocolate chip cookie. 

Object1: pink container with orange wedges
Relative Spatial Positioning: [0.48, 0.0, 0.98, 0.5]
Object Distance from the Lens: 0.08
Relative Size Proportion of Objects in Images (Percentage): 9.43
@@@
1. From the values of Object Relative Spatial Positioning [0.48, 0.0, 0.98, 0.5], we can determine that the object is located in the top right corner of the image, corresponding to "Adjacent to it in the top right corner, another pink container houses a mix of fruit. Sliced apples with their fresh white interiors exposed share the space with juicy chunks of pineapple. The colors of the apple slices and pineapple chunks contrast beautifully against the pink container" in the description. 
2. The description mentions "pink container houses a mix of fruit," aligning with our provided Object1 Caption "pink container with orange wedges". However, "Adjacent to it in the top right corner, another pink container houses a mix of fruit. Sliced apples with their fresh white interiors exposed share the space with juicy chunks of pineapple. The colors of the apple slices and pineapple chunks contrast beautifully against the pink container" does not include "orange wedges," so we need to incorporate them for additional detail. 
3. Additionally, with an Object Distance from the Lens of 0.08, we can infer that the object is relatively distant from the camera. 
4. Therefore, the original description can be revised to: "Adjacent to it in the top right corner, away from the camera side, another pink container houses a mix of fruit. Sliced apples with their fresh white interiors exposed share the space with juicy chunks of pineapple and orange wedges. The colors of the apple slices, pineapple chunks and orange wedges contrast beautifully against the pink container."
@@@

Object2 Caption: yellow food tray in food
Relative Spatial Positioning: [0.0, 0.4, 0.94, 1.0]
Object Distance from the Lens: 0.95
Relative Size Proportion of Objects in Images (Percentage): 13.93
@@@
1. From the values of Object Relative Spatial Positioning [0.0, 0.4, 0.94, 1.0], we can determine that the object is located in the bottom left of the image, corresponding to "Below these, in the bottom left corner of the tray, a yellow container holds a single meatball alongside some broccoli. The meatball, round and browned, sits next to the vibrant green broccoli florets" in the description.
2. The description mentions "a yellow container holds a single meatball alongside some broccoli" aligning with our provided Object2 Caption "yellow food tray in food". Since the original description "a yellow container holds a single meatball alongside some broccoli. The meatball, round and browned, sits next to the vibrant green broccoli florets." is already very clear, we do not need to add the "yellow food tray in food" into it. 
3. Additionally, with an Object Distance from the Lens of 0.95, we can infer that the object is very close to the camera. 
4. Therefore, the original description can be revised to: "Below these, in the bottom left corner of the tray, close to the camera, a yellow container holds a single meatball alongside some broccoli. The meatball, round and browned, sits next to the vibrant green broccoli florets"
@@@

Object3 Caption: broccoli in a yellow container
Relative Spatial Positioning: [0.4, 0.45, 0.91, 1.0]
Object Distance from the Lens: 1.0
Relative Size Proportion of Objects in Images (Percentage): 15.58
@@@
1. From the values of Object Relative Spatial Positioning [0.4, 0.45, 0.91, 1.0], we can determine that the object is located in the bottom left of the image, also corresponding to "Below these, in the bottom left corner of the tray, a yellow container holds a single meatball alongside some broccoli. The meatball, round and browned, sits next to the vibrant green broccoli florets" in the description.
2. The description mentions "a yellow container holds a single meatball alongside some broccoli" aligning with our provided Object3 Caption "broccoli in a yellow container". Since the original description "a yellow container holds a single meatball alongside some broccoli. The meatball, round and browned, sits next to the vibrant green broccoli florets." is already very clear, we do not need to add the "broccoli in a yellow container" into it. 
3. Additionally, with an Object Distance from the Lens of 1.0, we can infer that the object is also very close to the camera. 
4. Therefore, the original description can be revised to: "Below these, in the bottom left corner of the tray, close to the camera, a yellow container holds a single meatball alongside some broccoli. The meatball, round and browned, sits next to the vibrant green broccoli florets"
@@@

Object4 Caption: a pink container with food
Relative Spatial Positioning: [0.0, 0.03, 0.68, 0.78]
Object Distance from the Lens: 0.79
Relative Size Proportion of Objects in Images (Percentage): 5.85
@@@
1. From the values of Object Relative Spatial Positioning [0.0, 0.03, 0.68, 0.78], we can determine that the object is located in the upper left of the image, also corresponding to "In the top left pink container, a slice of bread rests, lightly spread with butter and sprinkled with a handful of almonds. The bread is cut into a rectangle, and the almonds are scattered across its buttery surface." in the description.
2. The description mentions "In the top left pink container, a slice of bread rests, lightly spread with butter and sprinkled with a handful of almonds. The bread is cut into a rectangle, and the almonds are scattered across its buttery surface." aligning with our provided Object4 Caption "a pink container with food". Since the original description is already very clear, we do not need to add the "a pink container with food" into it. 
3. Additionally, Object Distance from the Lens of 0.79 indicates that it is prominently displayed but not dominating the foreground. This suggests it should be described clearly but doesn't require exaggerated focus relative to other elements.
4. Therefore, the original description don't need to modify.
@@@

Object5 Caption: a slice of an orange
Relative Spatial Positioning: [0.72, 0.08, 0.82, 0.19]
Object Distance from the Lens: 0.0
Relative Size Proportion of Objects in Images (Percentage): 0.58
@@@
1. Because the positions of Object1, captioned as "pink container with orange wedges," with Object Relative Spatial Positioning: [0.48, 0.0, 0.98, 0.5], and Object5, captioned as "a slice of an orange," with Object Relative Spatial Positioning: [0.72, 0.08, 0.82, 0.19], are both in the top right corner, and Object1, whether in terms of its position [0.48, 0.0, 0.98, 0.5], or in terms of the meaning of "orange wedges," encompasses Object4, this can be disregarded.
@@@


%%%Your Modified Description:%%% In the center of the image, a vibrant blue lunch tray holds four containers, each brimming with a variety of food items. The containers, two in pink and two in yellow, are arranged in a 2x2 grid.\n\nIn the top left pink container, a slice of bread rests, lightly spread with butter and sprinkled with a handful of almonds. The bread is cut into a rectangle, and the almonds are scattered across its buttery surface.\n\nAdjacent to it in the top right corner, away from the camera side, another pink container houses a mix of fruit. Sliced apples with their fresh white interiors exposed share the space with juicy chunks of pineapple and orange wedges. The colors of the apple slices, pineapple chunks and orange wedges contrast beautifully against the pink container.\n\nBelow these, in the bottom left corner of the tray, close to the camera, a yellow container holds a single meatball alongside some broccoli. The meatball, round and browned, sits next to the vibrant green broccoli florets.\n\nFinally, in the bottom right yellow container, there's a sweet treat - a chocolate chip cookie. 



-----
# Example 2:
%%%The Original Description:%%% At the center of the frame is a **black Rolex clock** mounted on a **black pole**. The clock has a **white face** with **black hands**, indicating the time.\n\nBehind the clock, there's a **brown tree trunk** with a rough texture, adding a touch of nature to the scene. Further back, there's a **white building** with a **red tile roof**, possibly a hotel as indicated by the sign that reads "**HOTEL**".\n\nIn front of the building, there's a **white car** parked, perhaps belonging to one of the hotel guests. Beyond this immediate scene, there's a street with a **white crosswalk**, suggesting that this location is pedestrian-friendly.\n\n

Object1: the clock face is white
Relative Spatial Positioning: [0.23, 0.06, 0.55, 0.31]
Distance from the Lens: 1.0
Relative Size Proportion in Images (Percentage): 5.58
@@@
1. The original description already mentions "The clock has a **white face** with **black hands**". Since the object detail is already included, no modification is necessary for this object.
@@@

Object2: a parked silver car
Relative Spatial Positioning: [0.1, 0.6, 0.47, 0.84]
Distance from the Lens: 0.82
Relative Size Proportion in Images (Percentage): 4.37
@@@
1. From the values of Object Relative Spatial Positioning [0.1, 0.6, 0.47, 0.84], we can determine that the object's position is towards the lower middle of the left side of the image.
2. Object Distance from the Lens of 0.82, we can infer that it is relatively close to the camera, but since the Object Distance from the Lens for Object1 is 1.0, it is not the closest to the camera.
3. Since the original sentence only mentions "In front of the building, there's a white car parked, perhaps belonging to one of the hotel guests," and "a parked silver car" is not mentioned, this detail needs to be added. However, how to add it specifically will depend on reviewing all the subsequent objects.
@@@

Object3: a white truck parked on the street
Relative Spatial Positioning: [0.11, 0.55, 0.37, 0.65]
Distance from the Lens: 0.68
Relative Size Proportion in Images (Percentage): 1.05
@@@
1. We can infer that "a white truck parked on the street" better matches the expression "white car" in the original description "In front of the building, there's a white car parked, perhaps belonging to one of the hotel guests." than Object2 "a parked silver car". Therefore, we can modify "white car" to the more detailed "a white truck."
2. Since its Object Relative Spatial Positioning is [0.11, 0.55, 0.37, 0.65], it is also centrally located on the left side of the image, similar to the position of Object2. Additionally, because its Object Distance from the Lens is 0.68, which is smaller than Object2's 0.82, it indicates that Object2 "a parked silver car" is closer to the camera than "a white truck parked on the street." Therefore, the sentence "In front of the building, there's a white car parked, perhaps belonging to one of the hotel guests." can be modified to "In front of the building, there's a white car parked, perhaps belonging to one of the hotel guests. Closer to the camera, there is a parked silver car."
@@@

Object4: back of a stop sign
Relative Spatial Positioning: [0.24, 0.35, 0.32, 0.5]
Distance from the Lens: 0.69
Relative Size Proportion in Images (Percentage): 0.58
@@@
1. Since the Object Relative Spatial Positioning is [0.24, 0.35, 0.32, 0.5], it is centrally located on the left side of the image, similar to Object2 and Object3. 
2. Moreover, its Object Distance from the Lens is 0.79, which is close to Object2's 0.82, indicating that it and Object2 are very close in the image. 
3. Additionally, since "a stop sign" and "a parked silver car" logically coincide as cars generally park near stop signs, it can be inferred that "a silver car parked near a stop sign." Therefore, the previously revised sentence "In front of the building, there's a white car parked, perhaps belonging to one of the hotel guests. Closer to the camera, there is a parked silver car." can be further refined to "In front of the building, there's a white car parked, perhaps belonging to one of the hotel guests. Closer to the camera, there is a parked silver car near a stop sign."
@@@

Object5: a person walking on the sidewalk
Relative Spatial Positioning: [0.98, 0.57, 1.0, 0.6]
Distance from the Lens: 0.0
Relative Size Proportion in Images (Percentage): 0.05
@@@
1. Based on the Object Relative Spatial Positioning [0.98, 0.57, 1.0, 0.6], it can be inferred that "a person walking on the sidewalk" is positioned in the middle of the right side of the image.
2. Because the original description includes the line "Beyond this immediate scene, there's a street with a white crosswalk, suggesting that this location is pedestrian-friendly," we can logically associate "a person walking on the sidewalk" with this statement as a supplemental detail.
3. Furthermore, because the Object Distance from the Lens is 0.0, and the Relative Size Proportion of Objects in Images is 0.05, it indicates that "a person walking on the sidewalk" is positioned very far from the camera and is very small. Therefore, we can modify the original sentence "Beyond this immediate scene, there's a street with a white crosswalk, suggesting that this location is pedestrian-friendly." to "Beyond this immediate scene, there's a street with a white crosswalk, suggesting that this location is pedestrian-friendly. At a distance from the camera, you can see a person walking on the sidewalk, occupying a very small part of the entire image."
@@@

Object6: a white car
Relative Spatial Positioning: [0.11, 0.55, 0.26, 0.62]
Distance from the Lens: 0.67
Relative Size Proportion in Images (Percentage): 0.74
@@@
1. Since its Object Relative Spatial Positioning is [0.11, 0.55, 0.26, 0.62] and the Object Distance from the Lens is 0.67, both are very close to those of Object3 "a white truck parked on the street," which has an Object Relative Spatial Positioning of [0.11, 0.55, 0.37, 0.65] and an Object Distance from the Lens of 0.68, this detail duplicates Object3 and does not need to be added.
@@@


%%%Your Modified Description:%%% At the center of the frame is a **black Rolex clock** mounted on a **black pole**. The clock has a **white face** with **black hands**, indicating the time.\n\nBehind the clock, there's a **brown tree trunk** with a rough texture, adding a touch of nature to the scene. Further back, there's a **white building** with a **red tile roof**, possibly a hotel as indicated by the sign that reads "**HOTEL**".\n\nIn front of the building, there's a white car parked, perhaps belonging to one of the hotel guests. Closer to the camera, there is a parked silver car near a stop sign. Beyond this immediate scene, there's a street with a white crosswalk, suggesting that this location is pedestrian-friendly. At a distance from the camera, you can see a person walking on the sidewalk, occupying a very small part of the entire image.

-------
# Example 3:
%%%The Original Description:%%% In the heart of a winter wonderland, a lone skier, clad in a vibrant orange jacket, carves their way down a pristine, snow-covered slope.  

Object1: a man in a red jacket
Relative Spatial Positioning: [0.8, 0.76, 0.87, 0.93]
Distance from the Lens: 0.96
Relative Size Proportion in Images (Percentage): 0.5
@@@
1. In the original description, it mentions "a lone skier, clad in a vibrant orange jacket." Due to the similarity between an orange jacket and a red jacket, and based on "a lone skier," indicating only one person, the detail "a man in a red jacket" can be disregarded.
@@@

Object2: a backpack on the skiers back
Relative Spatial Positioning: [0.81, 0.76, 0.83, 0.81]
Distance from the Lens: 0.92
Relative Size Proportion in Images (Percentage): 0.04
@@@
1. Because the Object Relative Spatial Positioning and Object Distance from the Lens of this object are very similar to Object1's, "a backpack on the skier's back" most likely refers to "a lone skier, clad in a vibrant orange jacket." Therefore, the detail of "a backpack" should be added, resulting in "a lone skier, clad in a vibrant orange jacket, with a backpack on his back."
@@@

%%%Your Modified Description:%%% In the heart of a winter wonderland, a lone skier with a backpack on his back, clad in a vibrant orange jacket, carves their way down a pristine, snow-covered slope. 

------
# Example4:
%%%The Original Description:%%% The main focus is a bustling street, lined with a variety of buildings and shops. A yellow taxi, the only vehicle visible in the frame, is driving down the road.

Object1: a yellow car is driving down the street
Relative Spatial Positioning: [0.56, 0.65, 0.62, 0.73]
Distance from the Lens: 0.0
Relative Size Proportion in Images (Percentage): 0.4
@@@
1.  Since the original description mentions "A yellow taxi, the only vehicle visible in the frame, is driving down the road," and "a yellow car is driving down the street" are very similar in meaning, it can be assumed that Object1 refers to this sentence. Given that the Object Distance from the Lens is 0.0 and the Relative Size Proportion of Objects in Images (Percentage) is 0.4, it is evident that this is positioned far from the camera and occupies a small portion of the frame. Therefore, the original sentence can be revised to "A yellow taxi, far away from the camera, the only vehicle visible in the frame, is driving down the road."
@@@

Object2: a white truck on the street
Relative Spatial Positioning: [0.84, 0.62, 0.98, 0.82]
Distance from the Lens: 1.0
Relative Size Proportion in Images (Percentage): 1.84
@@@
1. Since the original sentence does not mention Object1, "a white truck on the street," this detail needs to be added. Based on its Object Relative Spatial Positioning of [0.84, 0.62, 0.98, 0.82] and its Object Distance from the Lens of 1.0, we know it is positioned in the bottom right corner of the image. The original description about the street states "A yellow taxi, the only vehicle visible in the frame, is driving down the road, adding a pop of color to the scene.". Since the phrase "the only vehicle visible in the frame" contradicts the information about Object2, we can revise this sentence to: "The main focus is a bustling street, lined with a variety of buildings and shops. A yellow taxi far away from the camera, is driving down the road, and there is a white truck on the street in the bottom right corner of the image close to the camera."
@@@

%%%Your Modified Description:%%% The main focus is a bustling street, lined with a variety of buildings and shops. A yellow taxi far away from the camera, is driving down the road, and there is a white truck on the street in the bottom right corner of the image close to the camera.


###Task###
[**You only need to provide the modified description directly after "%%%Your Modified Description:%%%" that I provided**.]


"""


def main(args):
    total_lines = args.end_line - args.start_line
    stop_tokens = args.stop_tokens.split(",")

    with open(args.input_file, 'r') as input_file, open(args.output_file, 'a') as output_file:
        # prompts = []
        # flags = []
        # imgs = []
        # ori_descs = []
        llm = LLM(model="./llama3/Meta-Llama-3-70B-Instruct", tensor_parallel_size=4)
        for i, line in enumerate(tqdm(input_file, total=args.end_line, desc="Adding objects to the description")):
            if i < args.start_line:
                continue
            if i >= args.end_line:
                break
            output_str = ""  
            json_obj = json.loads(line)
            image = json_obj.get("image")
            objects = json_obj.get("objects", [])
            boxes = json_obj.get("bounding_boxes")
            obj_depths = json_obj.get("object_depth")
            sizes = json_obj.get("size")
            width = json_obj.get("width")
            height = json_obj.get("height")
            description = json_obj.get("description")

            # add_obj_num = len(objects)
            add_obj_num = min(len(objects), 10)

            if add_obj_num < 2:
                modified_desc = description
                
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
                if max_depth == min_depth:
                    modified_desc = description
                   
                else:
                    output_str += "%%%The Original Description:%%% " + description + "\n\n"
                    for i in range(add_obj_num):
                        depth_norm = (obj_depths[i] - min_depth) / (max_depth - min_depth)
                        relative_size = float(sizes[i]) / float(height * width) *100
                        output_str += f"Object{i+1}: {objects[i]}\n"
                        output_str += f"Relative Spatial Positioning: {[norm_boxes[i][0], norm_boxes[i][1], norm_boxes[i][2], norm_boxes[i][3]]}\n"
                        output_str += f"Distance from the Lens: {round(depth_norm,2)}\n"
                        output_str += f"Relative Size Proportion in Images (Percentage): {round(relative_size,2)}\n\n"

                    output_str += "%%%Your Modified Description:%%% "

                    prompt = additional_info + "" + output_str
                    
                    prompt = args.prompt_structure.format(input=prompt)
                    # print(prompt)
                    sampling_params = SamplingParams(temperature=0.75, top_p=0.95, max_tokens=4096, stop=stop_tokens)
                    # Create an LLM.
                    response = llm.generate(prompt, sampling_params)
                    # print(response[0].outputs[0].text)
                    modified_desc= response[0].outputs[0].text                        

            data = {
                "image": image,
                "original_description": description,
                "modified_description": modified_desc
            }
            json.dump(data, output_file)
            output_file.write('\n')
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process image descriptions.")
    parser.add_argument("--input_file", type=str, help="Path to the input JSON file.")
    parser.add_argument("--output_file", type=str, help="Path to the output JSONL file.")
    parser.add_argument("--start_line", type=int, default=0, help="Start line in the input file.")
    parser.add_argument("--end_line", type=int, default=999999999, help="Start line in the input file.")
    parser.add_argument("--stop_tokens", type=str, default="</s>", help="Stop tokens for generation")
    parser.add_argument("--prompt_structure", type=str,
                        default="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {input} ASSISTANT:",
                        help="Prefix for generation")
    args = parser.parse_args()
    main(args)




