import math
import copy
from PIL import ImageDraw, Image
import numpy as np
from typing import Any, Dict, List, Tuple, Optional
import cv2
import os
import io
import base64

color_list_1 = ['red', 'blue', 'green', 'magenta', "purple"]

def img2base64(image:Image, size=384):
    """_summary_

    Args:
        image (Image): _description_
        size (int, optional): the size of Image 384. Defaults to 384.

    Returns:
        base64_data (str): the string of svg
    """
    
    if size != -1:
        image_width, image_height = image.size
        max_ratio = size / max(image_width, image_height)
        new_width, new_height = int(image_width * max_ratio), int(image_height * max_ratio)
        image = image.resize((new_width, new_height))
        
    # 将 PIL 图像转换为字节流
    byte_stream = io.BytesIO()
    image.save(byte_stream, format='PNG')
    byte_stream.seek(0)

    # 将字节流转换为 Base64 编码字符串
    base64_data = base64.b64encode(byte_stream.read()).decode('utf-8')
    
    return base64_data


def vis_mol(image:Image, 
            predictions:List[Dict],
            vis_label = True,
            vis_text = True,
            save_path=None) -> Image:
    """_summary_

    Args:
        image (Image): _description_
        predictions (List[Dict]): _description_
        save_path (str, optional): _description_. Defaults to "temp.png".

    Returns:
        temp_image (Image): 修饰后的图片
    """
    temp_image = copy.deepcopy(image)
    draw = ImageDraw.Draw(temp_image)
    for _, box_dict in enumerate(predictions):
        mol_box = box_dict.get("mol_box") if "mol_box" in box_dict else None
        
        confidence = 1
        if "molecule_prediction" in box_dict:
            if "confidence" in box_dict:
                confidence = box_dict["molecule_prediction"]["confidence"]
        if confidence < 0.3:
            continue
        
        if mol_box is not None:
            x1, y1, x2, y2  = (math.floor(mol_box[0]), math.floor(mol_box[1]), math.ceil(mol_box[2]), math.ceil(mol_box[3]))
            draw.rectangle(
                            [(x1, y1), (x2, y2)], 
                            outline=color_list_1[_%len(color_list_1)], width=3
                            )
        
        if vis_label:
            label_boxes = box_dict.get("label_box") if "label_box" in box_dict else None
            if label_boxes is not None:
                for label_box in label_boxes:
                    x1, y1, x2, y2  = (math.floor(label_box[0]), math.floor(label_box[1]), math.ceil(label_box[2]), math.ceil(label_box[3]))
                
                draw.rectangle(
                                [(x1, y1), (x2, y2)], 
                                outline=color_list_1[_%len(color_list_1)], width=1
                                )
        
        if vis_text:
            text_box = box_dict.get("text_box") if "text_box" in box_dict else None
            if text_box is not None:
                x1, y1, x2, y2  = (math.floor(text_box[0]), math.floor(text_box[1]), math.ceil(text_box[2]), math.ceil(text_box[3]))
                draw.rectangle(
                            [(x1, y1), (x2, y2)], 
                            outline=color_list_1[_%len(color_list_1)], width=3
                            )

    ## 确保能保存
    if (save_path is not None) and os.path.exists(os.path.dirname(save_path)):
        temp_image.save(save_path)
    return temp_image

def vis_table(image:Image, extracted_tables, save_path="temp.png"):
    image_array = np.array(image)
    for table in extracted_tables:
        for row in table.content.values():
            for cell in row:
                cv2.rectangle(image_array, (cell.bbox.x1, cell.bbox.y1), (cell.bbox.x2, cell.bbox.y2), (255, 0, 0), 2)
    Image.fromarray(image_array).save(save_path)

def vis_table_v2(image:Image, table, save_path="temp.png"):
    image_array = np.array(image)
    for row in table.content.values():
        for cell in row:
            cv2.rectangle(image_array, (cell.bbox.x1, cell.bbox.y1), (cell.bbox.x2, cell.bbox.y2), (255, 0, 0), 2)
    Image.fromarray(image_array).save(save_path)
