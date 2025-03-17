## 添加主目录
import os
import sys
main_dir = os.path.abspath(".")
gen_data_dir = os.path.join(main_dir,"chemistry_data")
print("main_dir",main_dir)
sys.path.append(main_dir)

from PIL import Image, ImageDraw, ImageFont
import os
import pickle
import os
import random
import ipdb
import cv2
import numpy as np
import pandas as pd
from rdkit import Chem
import cv2
from rdkit.Chem import PandasTools
from get_indigo_mol import generate_indigo_image
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from textwrap import wrap
import json
import copy
import string
from tqdm import tqdm 

import albumentations as A
import imgkit
import argparse

class CropWhite(A.DualTransform):
    
    def __init__(self, value=(255, 255, 255), pad=0, p=1.0):
        super(CropWhite, self).__init__(p=p)
        self.value = value
        self.pad = pad
        assert pad >= 0

    def update_params(self, params, **kwargs):
        super().update_params(params, **kwargs)
        assert "image" in kwargs
        img = kwargs["image"]
        height, width, _ = img.shape
        try:
            x = (img != self.value).sum(axis=2)
        except Exception as e:
            import ipdb
            ipdb.set_trace()
        if x.sum() == 0:
            return params
        row_sum = x.sum(axis=1)
        top = 0
        while row_sum[top] == 0 and top+1 < height:
            top += 1
        bottom = height
        while row_sum[bottom-1] == 0 and bottom-1 > top:
            bottom -= 1
        col_sum = x.sum(axis=0)
        left = 0
        while col_sum[left] == 0 and left+1 < width:
            left += 1
        right = width
        while col_sum[right-1] == 0 and right-1 > left:
            right -= 1
        # crop_top = max(0, top - self.pad)
        # crop_bottom = max(0, height - bottom - self.pad)
        # crop_left = max(0, left - self.pad)
        # crop_right = max(0, width - right - self.pad)
        # params.update({"crop_top": crop_top, "crop_bottom": crop_bottom,
        #                "crop_left": crop_left, "crop_right": crop_right})
        params.update({"crop_top": top, "crop_bottom": height - bottom,
                       "crop_left": left, "crop_right": width - right})
        return params

    def apply(self, img, crop_top=0, crop_bottom=0, crop_left=0, crop_right=0, **params):
        height, width, _ = img.shape
        img = img[crop_top:height - crop_bottom, crop_left:width - crop_right]
        img = A.augmentations.pad_with_params(
            img, self.pad, self.pad, self.pad, self.pad, border_mode=cv2.BORDER_CONSTANT, value=self.value)
        return img

    def apply_to_keypoint(self, keypoint, crop_top=0, crop_bottom=0, crop_left=0, crop_right=0, **params):
        x, y, angle, scale = keypoint[:4]
        return x - crop_left + self.pad, y - crop_top + self.pad, angle, scale

    def get_transform_init_args_names(self):
        return ('value', 'pad')



def get_transforms_fn(pad=10):
    trans_list = []
    trans_list.append(CropWhite(pad=pad))
    return A.Compose(trans_list, keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
transform_fn = get_transforms_fn(pad=10)
transform_fn_2 = get_transforms_fn(pad=6)


def get_graph_data(original_smiles:str):
    """Generate graph from smiles

    Args:
        original_smiles (str): smiles of original molecule

    Returns:
        img (numpy.ndarray): ndarray of image, can be converted to 'PIL.Image.Image' with `mage.fromarray(img)`
        smiles (str): smiles of transformed molecule
        graph (dict): `coordinates` and `symbol` of transformed molecule
        success (bool): The Flag of success
    """
    img, smiles, graph, success = generate_indigo_image(original_smiles, mol_augment=False, default_option=True)
    return img, smiles, graph, success


def get_result(batch):
    
    idx, original_smiles, save_dir = batch
    
    ## 先检查分子的合法性
    mol = None
    try:
        mol = Chem.MolFromSmiles(original_smiles)
    except Exception as e:
        print(e)
    if mol is None:
        return {}
    
    count = -1
    flag = False
    while True:
        count = count + 1
        if count == 100:
            break
        img, smiles, graph, success =  get_graph_data(original_smiles)
        
        
        if success and graph!={}:
            image = Image.fromarray(img)
            image.save(os.path.join(save_dir, "img", f"{idx}.png"))
            if len(cv2.imread(os.path.join(save_dir, "img", f"{idx}.png")).shape)==3:
                flag = True
                graph["smiles"] = smiles
            
        if flag is True:
            break
    
    if flag:
        return {
            "%d"%(idx):{
            "original_smiles": original_smiles,
            "smiles":smiles, #已经被覆盖了
            "graph":graph,
            }
        }
    else:
        return {}

letters = string.ascii_lowercase
idx_list = [u"\u2160",u"\u2161",u"\u2162",u"\u2163", u"\u2164", u"\u2165", u"\u2166", u"\u2167", u"\u2168", u"\u2169"]
def get_index():
    """获取索引

    Returns:
        result1 (str): _description_
    """
    bracket_flag = False
    result1 = ""
    if random.random()<0.5:
        result1 = random.choice(idx_list)
    else:
        if random.random()<0.5:
            result1 = str(random.randint(1,500))
            if random.random() > 0.5:
                alpha = random.sample(letters, 1)[0]
                ## 有0.3的概率变大写
                if random.random()<0.25:
                    alpha = alpha.upper()
                
                ## 添加括号
                if random.random()<0.25:
                    alpha =  "(" + alpha + ")"
                    bracket_flag = True
            
                ## 添加空格
                if random.random()<0.5:
                    result1 += " "+ alpha
                else:
                    result1 += alpha
        else:
            result1 = str(random.randint(1,1000))
            if random.random() > 0.5:
                roman_num = random.choice(idx_list)
                if random.random()<0.25:
                    roman_num =  "(" + roman_num + ")"
                    bracket_flag = True
                ## 添加空格
                if random.random()<0.5:
                    result1 += " "+ roman_num
                else:
                    result1 += roman_num
    
    if random.random()<0.1 and (bracket_flag is False):
        result1 = "(" + result1 + ")"
    
    return result1

def get_inchi_key(mol):
    import ipdb
    ipdb.set_trace()
    # 获取InChI
    inchi = Chem.MolToInchi(mol)
    return inchi

drug_name_df = pd.read_csv(os.path.join(main_dir, "chemistry_data", "strata-drug-list.csv"))
def get_drug_name():
    count = 0
    while count<10:
        count = count + 1
        random_idx = random.randint(0, len(drug_name_df)-1)
        drug_name = drug_name_df.loc[random_idx, "drug"]
        if len(drug_name)<=12:
            break
        if count == 10:
            drug_name = str(random.randint(0, 100))
            break
    
    if " " not in drug_name:
        if random.random()<0.5:
            return " ".join([drug_name.capitalize()])
        else:
            return " ".join([drug_name])
    else:
        if random.random()<0.5:
            return drug_name.capitalize()
        else:
            return drug_name

df_drug_index = pd.read_csv(os.path.join(main_dir, "chemistry_data", "drug.names.csv"),header=None)
df_drug_index.columns = ["index", "unknown"]
def get_drug_idx():
    drug_index = df_drug_index.sample(1).iloc[0, 0] * random.randint(1, 3)
    return drug_index

def get_logP():
    return -round(random.randint(-1000, 1000)*0.01, 2)

def get_logS():
    return round(random.randint(-1000, 1000)*0.01, 2)

def get_MS():
    return round(random.randint(10, 1000)*0.1, 1)

def get_Vina_Score():
    return -round(random.randint(50, 200)*0.01, 2)

def get_SA_Score():
    return round(random.randint(0, 100)*0.01, 2)

## 缩写表
abbre_dict = {}
with open(os.path.join(main_dir, "chemistry_data", "abbretion.txt"), "r") as f:
    for line in f.readlines():
        line = line.rstrip("\n")
        line_list = line.split()
        if len(line_list) == 2:
            key, value = line_list
            if "*" not in value:
                abbre_dict[key] = value
abbre_dict_key_list = list(abbre_dict.keys())
alphabet = 'abcdefghijklmnopqrstuvwxyz'
def get_RGroup():
    R_Group_dict = {}
    ## 标签
    random_int = random.randint(1, 99)
    with_alpha = random.random()>0.5
    start_alpha_index = random.randint(0, len(alphabet[:-6]))
    cap_token = random.random()>0.8
    
    ## 额外内容
    with_r_group = random.random()>0.2
    if with_r_group:
        temp_proba = random.random()
        if temp_proba < 0.8:
            r_group = "R" + str(random.choice([random.randint(1, 9), ""]))
        else:
            r_group = random.choice(["X", "Y"])
    else:
        r_group = ""
    
    random_token = random.choice([":", "=",])
    
    while len(R_Group_dict) < 10:
        if with_alpha:
            if cap_token:
                alpha = alphabet[(start_alpha_index + len(R_Group_dict))%(len(alphabet))].capitalize()
            else:
                alpha = alphabet[(start_alpha_index + len(R_Group_dict))%(len(alphabet))]
            key = f"{random_int}{alpha}"
        else:
            key = f"{random_int + len(R_Group_dict)}"
        
        group = random.choice(abbre_dict_key_list)
        proba = random.random()
        if proba<0.4:
            group = "-" + group
        elif proba<0.6:
            group = "-" + group + "-"
        
        if r_group != "":
            total_r_group = r_group + random_token + group
        else:
            total_r_group = group
        
        
        R_Group_dict[key] = total_r_group
        if len(R_Group_dict) > 2 and random.random()>0.7:
            break
    
    return R_Group_dict
    

## 获取分子式
def get_formular(mol):
    # 计算分子式
    formula = CalcMolFormula(mol) * random.randint(1, 2)
    return formula

def get_smiles(mol):
    return Chem.MolToSmiles(mol)

def get_activity():
    proba_index = random.random()
    if proba_index <= 0.33:
        result1 = random.choice(["Kd","IC50","Kd"]) + " : " + str(round(random.randint(1,10000)*0.1, 3)) + random.choice(["nm","um"])
    elif proba_index <= 0.66:
        result1 = u"\u25B3"+"T"+":"+ str(random.randint(-10,100)) + u"\u2103"
    else:
        result1 = random.choice(["-","+"]) + u"\u25B3"+"G"+" : "+ str(round(random.randint(-100,100)*0.1, 2)) + " Kcal/mol"
    return result1

def get_yield():
    """Random generate

    Returns:
        result1 (str): yield result
    """
    result1 = random.choice(["yield","ee","E/Z","S/R"]) +" : "+ str(round(random.randint(0, 1000)*0.1, 2)) + "%"
    return result1

## 获取对分子的评论
def get_comment(mol:Chem.rdchem.Mol):
    """获取分子的一些属性星系

    Args:
        mol (Chem.rdchem.Mol): _description_

    Returns:
        result_dict (): _description_
    """
    result_dict = {}
    result_dict["index"] = get_index()
    # result_dict["inchi"] = get_inchi_key(mol)
    # result_dict["smiles"] = get_smiles(mol)
    # result_dict["drug_index"] = get_drug_idx()

    result_dict["Formula"] = get_formular(mol)
    # result_dict["activity"] = get_activity()
    # result_dict["yield"] = get_yield()
    result_dict["drug_name"] = get_drug_name()
    # result_dict["LogP"] = get_logP()
    # result_dict["LogS"] = get_logS()
    result_dict["m/z"] = get_MS()
    result_dict["Vina Score"] = get_Vina_Score()
    result_dict["SA Score"] = get_SA_Score()
    return result_dict

import string
import random
def generate_random_text(length):
    characters = string.ascii_letters + string.digits  # 包含所有字母和数字的字符串
    random_text = ''.join(random.choice(characters) for _ in range(length))  # 从字符集中随机选择字符，重复 length 次
    return random_text

def get_img(env, key):
    with env.begin() as f:
        pickled_data = f.get(key)
        data = pickle.loads(pickled_data)
    
    image = Image.open(data["image_path"])
    return image, data["image_path"]

def get_img_v2(env, key):
    with env.begin() as f:
        pickled_data = f.get(key)
        data = pickle.loads(pickled_data)
        
    return data

def get_blank_area(image):
    # 读取图像
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)#转为灰度图
    ret, binary = cv2.threshold(gray,127,255,0)#转二值图
    contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    
    min_height_list = find_min_heights(contours, image)
    
    area, ans = find_max_box(min_height_list)
    
    box = (ans[0], image.shape[0]-ans[2],  ans[1] - ans[0], ans[2]) # w, y, h, w
    
    return area, box


def find_min_heights(contours, image):
    min_height_list = [np.inf] * image.shape[1]
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        if w == image.shape[1] or h == image.shape[0]:
            continue
        
        for i in range(x, x+w):
            min_height_list[i] = min(min_height_list[i], image.shape[0] - y - h)
    
    for i in range(len(min_height_list)):
        if min_height_list[i] == np.inf:
            j = i + 1
            while j < len(min_height_list):
                if min_height_list[j] != np.inf:
                    min_height_list[i] = min_height_list[j]
                    break
                j = j + 1
            
            if min_height_list[i] == np.inf:
                j = i - 1
                while j >=0:
                    if min_height_list[j] != np.inf:
                        min_height_list[i] = min_height_list[j]
                        break
                    j = j - 1
            
            if min_height_list[i] == np.inf:
                min_height_list[i] = 0
    
    return min_height_list

def find_max_box(heights):
    stack = []
    heights = [0] + heights + [0]
    area = 0
    ans = 0, 0, 0
    for i in range(len(heights)):
        #print(stack)
        while stack and heights[stack[-1]] > heights[i]:
            tmp = stack.pop()
            tmp_res =  (i - stack[-1] - 1) * heights[tmp]
            if tmp_res >= area:
                area = tmp_res
                ans = stack[-1], i, heights[tmp]
        stack.append(i)
    return area, ans

def get_text_dimensions(text_string, font):
    # https://stackoverflow.com/a/46220683/9263761
    ascent, descent = font.getmetrics()

    text_width = font.getmask(text_string).getbbox()[2]
    text_height = font.getmask(text_string).getbbox()[3] + descent
        
    return (text_width, text_height)


def pad_image(image, padding):
    # 获取图像的宽度和高度
    width, height = image.size

    # 计算填充后的高度
    padded_height = height + padding

    # 创建一个新的图像对象，宽度与原图像相同，高度为填充后的高度
    padded_image = Image.new(image.mode, (width, padded_height), (255, 255, 255))

    # 将原图像粘贴到新图像中，保持原有位置不变
    padded_image.paste(image, (0, 0))

    return padded_image


def get_image_mol_box(original_smiles, scale=1.0):
    
    img, smiles, graph, success =  get_graph_data(original_smiles)
    
    image = Image.fromarray(img)
    target_width = int(image.size[0] * scale)
    target_height = int(image.size[1] * scale)
    # 缩放图像
    image = image.resize((target_width, target_height))
    box = (0, 0, image.size[0], image.size[1])
    
    mol_box = {
            "bbox":box,
            "category_id":1
        }
    
    return image, mol_box

def get_image_idx_box(text_string, bold_PIL_font):

    text_width, text_height = get_text_dimensions(text_string, bold_PIL_font)
    text_height = text_height + 5
    # 创建一个新的图像对象，宽和长分别为(text_width, text_height)
    idx_image = Image.new('RGB', (text_width, text_height), (255, 255, 255))
    
    draw = ImageDraw.Draw(idx_image)
    if random.random() < 0.5:
        draw.text((0, 0), text_string, font=bold_PIL_font, fill=(0, 0, 0), spacing=0)
    else:
        draw.text((0, 0), text_string, font=bold_PIL_font, fill=(0, 0, 0), fontweight="bold", spacing=0)
    
    box = (0, 0, text_width, text_height)
    
    idx_box = {
                "bbox":box,
                "category_id":3
            }
    
    return idx_image, idx_box


bold_font_list = [ImageFont.truetype(os.path.join(main_dir,"font","FreeSerifBold.ttf")), ImageFont.truetype(os.path.join(main_dir,"font","Arial-Unicode-Bold.ttf"))]
font_list = [ImageFont.truetype(os.path.join(main_dir,"font","FreeSerifBold.ttf")), ImageFont.truetype(os.path.join(main_dir,"font","Arial Unicode MS.TTF"))]


def get_drug_idx_by_comment(comment, light_PIL_font, bold_PIL_font):
    drug_index = ""
    drug_name = ""
    sep = ""
    prefix = ""
    
    drug_index = comment["index"]
    if random.random()<0.5:
        if comment["index"][0] != "(" and comment["index"][-1] != ")":
            comment["index"] = "(" + comment["index"] + ")"
        drug_index = comment["index"]
        # drug_name = comment["drug_name"]
    # sep = random.choice([":", " ", ","])
    
    ## 添加手性的符号
    proba = random.random()
    if proba<0.75:
        if random.random()<0.6:
            prefix = "(" + random.choice([u'\u00B1', "+", "-"]) + ") "
        elif random.random()<0.8:
            if "FreeSerif" not in bold_PIL_font.getname():
                prefix = prefix + random.choice(['化合物 ', "实施例 ", "Compound ", "Example "])
            else:
                prefix = prefix + random.choice(["Compound ", "Example "])

        else:
            if "FreeSerif" not in bold_PIL_font.getname():
                prefix = random.choice(['化合物 ', "实施例 ", "Compound ", "Example "]) + prefix
            else:
                prefix = random.choice(["Compound ", "Example "]) + prefix

    else:
        prefix = random.choice(['(R)-', "(S)-"])
        if "FreeSerif" not in bold_PIL_font.getname():
            prefix = prefix + random.choice(['化合物 ', "实施例 ", "Compound ", "Example "])
        else:
            prefix = prefix + random.choice(["Compound ", "Example "])
    text_string = prefix + drug_name + sep + drug_index

    return text_string

def get_new_idx_image_box(comment:dict, 
                    bold_PIL_font:ImageFont.FreeTypeFont, 
                    light_PIL_font:ImageFont.FreeTypeFont):
    
    idx_text = get_drug_idx_by_comment(comment, light_PIL_font, bold_PIL_font)
    text_width, text_height = get_text_dimensions(idx_text, bold_PIL_font)
    text_height = text_height + 5
    idx_image = Image.new('RGB', (text_width, text_height), (255, 255, 255))
    draw = ImageDraw.Draw(idx_image)
    if random.random()<0.5:
        draw.text((0, 0), idx_text, font=light_PIL_font, fill=(0, 0, 0))
    else:
        draw.text((0, 0), idx_text, font=bold_PIL_font, fill=(0, 0, 0))
    
    # 计算新的尺寸
    scale = random.randint(100, 150)*0.01 
    new_width = int(idx_image.width * scale)
    new_height = int(idx_image.height * scale)

    # 扩大图像
    resized_image = idx_image.resize((new_width, new_height), Image.LANCZOS)

    augmented = transform_fn_2(image=np.array(resized_image), keypoints=[[0, 0],
                                                                     [resized_image.width, resized_image.height]])
    new_image = Image.fromarray(augmented["image"])

    bbox = {
        "bbox":[0,0, new_image.width, new_image.height],
        "category_id":3,
    }
    return new_image, bbox
    


def get_image_idx_box_V2(drug_idx_proba:float, 
                         comment:dict, 
                         bold_PIL_font:ImageFont.FreeTypeFont, 
                         light_PIL_font:ImageFont.FreeTypeFont):

    ## change size first
    bold_PIL_font = copy.deepcopy(bold_PIL_font)
    light_PIL_font = copy.deepcopy(light_PIL_font)
    light_PIL_font.size = bold_PIL_font.size
    
    drug_index = ""
    drug_name = ""
    sep = ""
    prefix = ""
    
    if drug_idx_proba < 0.4:
        drug_index = comment["index"]
    elif drug_idx_proba < 0.6:
        drug_name = comment["drug_name"]
    elif drug_idx_proba < 0.8:
        if comment["index"][0] != "(" and comment["index"][-1] != ")":
            comment["index"] = "(" + comment["index"] + ")"
        drug_index = comment["index"]
        drug_name = comment["drug_name"]
        sep = random.choice([":", " ", ","])
    
    
    if drug_idx_proba <0.5:
        ## 添加手性的符号
        proba = random.random()
        if proba<0.4:
            prefix = "(" + random.choice([u'\u00B1', "+", "-"]) + ") "
            if random.random()<0.5:
                if "FreeSerif" not in bold_PIL_font.getname():
                    prefix = prefix + random.choice(['化合物 ', "实施例 ", "Compound ", "Example "])
                else:
                    prefix = prefix + random.choice(["Compound ", "Example "])

            else:
                if "FreeSerif" not in bold_PIL_font.getname():
                    prefix = random.choice(['化合物 ', "实施例 ", "Compound ", "Example "]) + prefix
                else:
                    prefix = random.choice(["Compound ", "Example "]) + prefix

        elif proba <0.8:
            prefix = random.choice(['(R)-', "(S)-"])
            if "FreeSerif" not in bold_PIL_font.getname():
                prefix = prefix + random.choice(['化合物 ', "实施例 ", "Compound ", "Example "])
            else:
                prefix = prefix + random.choice(["Compound ", "Example "])
        text_string = prefix + drug_name + sep + drug_index

        text_string = text_string

        text_width, text_height = get_text_dimensions(text_string, bold_PIL_font)
        text_height = text_height + 5
        # 创建一个新的图像对象，宽和长分别为(text_width, text_height)
        idx_image = Image.new('RGB', (text_width, text_height), (255, 255, 255))
        
        draw = ImageDraw.Draw(idx_image)
        
        offset_x = 0
        if prefix != "":
            temp_font =  random.choices([bold_PIL_font, light_PIL_font], weights=[0.66, 0.33])[0]
            temp_text_width, temp_text_height = get_text_dimensions(prefix, temp_font)
            draw.text((offset_x, 0), prefix, font=temp_font, fill=(0, 0, 0))
            offset_x = offset_x + temp_text_width
            
        if drug_idx_proba < 0.4:
            temp_font = random.choices([bold_PIL_font, light_PIL_font], weights=[0.9, 0.1])[0]
            temp_text_width, temp_text_height = get_text_dimensions(drug_index, temp_font)
            draw.text((offset_x, 0), drug_index, font=temp_font, fill=(0, 0, 0))
            offset_x = offset_x + temp_text_width
            
        elif drug_idx_proba < 0.6:
            temp_font =  random.choices([bold_PIL_font, light_PIL_font], weights=[0.33, 0.66])[0]
            temp_text_width, temp_text_height = get_text_dimensions(drug_name, temp_font)
            draw.text((offset_x, 0), drug_name, font=temp_font, fill=(0, 0, 0))
            offset_x = offset_x + temp_text_width
        
        elif drug_idx_proba < 0.7:
            temp_font =  random.choices([bold_PIL_font, light_PIL_font], weights=[0.33, 0.66])[0]
            temp_text_width, temp_text_height = get_text_dimensions(drug_name + sep, temp_font)
            draw.text((offset_x, 0), drug_name + sep, font=temp_font, fill=(0, 0, 0))
            offset_x = offset_x + temp_text_width
            
            temp_font =  random.choices([bold_PIL_font, light_PIL_font], weights=[0.9, 0.1])[0]
            temp_text_width, temp_text_height = get_text_dimensions(drug_index, temp_font)
            draw.text((offset_x, 0), drug_index, font=temp_font, fill=(0, 0, 0))
            offset_x = offset_x + temp_text_width
            
        elif drug_idx_proba < 0.8:
            temp_font =  random.choices([bold_PIL_font, light_PIL_font], weights=[0.9, 0.1])[0]
            temp_text_width, temp_text_height = get_text_dimensions(drug_index, temp_font)
            draw.text((offset_x, 0), drug_index, font=temp_font, fill=(0, 0, 0))
            offset_x = offset_x + temp_text_width
            
            temp_font =  random.choices([bold_PIL_font, light_PIL_font], weights=[0.33, 0.66])[0]
            temp_text_width, temp_text_height = get_text_dimensions(sep + drug_name, temp_font)
            draw.text((offset_x, 0), sep + drug_name, font=temp_font, fill=(0, 0, 0))
            offset_x = offset_x + temp_text_width
        
        
        box = (0, 0, text_width, text_height)
        
        idx_box = {
                    "bbox":box,
                    "category_id":3
                }
        
        return text_string, idx_image, idx_box
    else:
        r_group = get_RGroup()
        total_text_width = 0
        total_text_height = 0
        text_string_list = []
        token = random.choice(["=", ":"])
        offset = 0 #-int(bold_PIL_font.size * random.randint(-5,5)*0.1)
        for key, value in r_group.items():
            text_string = f"{key} {value}"
            text_width, text_height = get_text_dimensions(text_string,bold_PIL_font)
            
            text_string_list.append(text_string)
            
            total_text_width = max(total_text_width, text_width)
            total_text_height = total_text_height + text_height + offset
        
        total_text_height = total_text_height + 5
        idx_image = Image.new('RGB', (total_text_width, total_text_height), (255, 255, 255))
        draw = ImageDraw.Draw(idx_image)
        
        offset_y = offset
        
        for key, value in r_group.items():
            ## 先总体
            text_string = f"{key} {value}"
            text_width, text_height = get_text_dimensions(text_string, bold_PIL_font)
            start_x = total_text_width//2 - text_width // 2
            
            ## 再个体
            start_y = offset_y
            key_text_width, key_text_height = get_text_dimensions(key, bold_PIL_font)
            draw.text((start_x, start_y), key, font=bold_PIL_font, fill=(0, 0, 0), spacing=0)
            
            valud_text_width, value_text_height = get_text_dimensions(" " + value, bold_PIL_font)
            draw.text((start_x + key_text_width, start_y), " " + value, font=light_PIL_font, fill=(0, 0, 0), spacing=0)

            offset_y = offset_y + offset + text_height
        
        box = (0, 0, total_text_width, total_text_height)
        idx_box = {
                    "bbox":box,
                    "category_id":3
                }
        
        return "\n".join(text_string_list), idx_image, idx_box


def get_image_txt_box(comment, light_PIL_font, CHAR_LIMIT):
    
    offset = 0#-int(light_PIL_font.size * random.randint(-3,15)*0.1)
    
    ## 确保一定有内容
    sample_list = []
    keys_list = [_ for _ in comment.keys() if (_ not in ["index", "drug_name"])]
    sample_list = random.sample(keys_list, random.randint(1, len(keys_list)))
    random.shuffle(sample_list)
    with_colon = random.random()>0
    text_string_list = []
    total_text_width = 0
    total_text_height = 0
    
    for sample in sample_list:
        if sample == "activity" and sample == "yield":
            text = str(comment[sample])
            
        elif sample == "LogP" and sample == "LogS" :
            text = str(sample) + " : " + str(comment[sample])
        else:
            if with_colon:
                text = str(sample) + " : " + str(comment[sample])
            else:
                text = str(comment[sample])
            
        ## 检测是否跨行
        text_lines = wrap(text, CHAR_LIMIT)
        text_string_list.extend(text_lines)
        for text_string in text_lines:
            text_width, text_height = get_text_dimensions(text_string, light_PIL_font)
            
            total_text_width = max(total_text_width, text_width)
            total_text_height = total_text_height + offset + text_height
            
    
    total_text_height = total_text_height + offset + 5
    txt_image = Image.new('RGB', (total_text_width, total_text_height), (255, 255, 255))
    
    draw = ImageDraw.Draw(txt_image)
    
    offset_y = offset
    bold_flag = random.random() > 0.5
    
    for text_string in text_string_list:
        text_width, text_height = get_text_dimensions(text_string, light_PIL_font)
        start_x = total_text_width//2 - text_width // 2
        start_y = offset_y
        if bold_flag is False:
            draw.text((start_x, start_y), text_string, font=light_PIL_font, fill=(0, 0, 0), spacing=0)
        else:
            draw.text((start_x, start_y), text_string, font=light_PIL_font, fill=(0, 0, 0), fontweight="bold", spacing=0)

        offset_y = offset_y + offset + text_height
    
    box = (0, 0, total_text_width, total_text_height)
    txt_box = {
                "bbox":box,
                "category_id":2
            }
    
    return txt_image, txt_box

def get_image_random_txt_box(random_string, light_PIL_font, CHAR_LIMIT):
    
    offset = -int(light_PIL_font.size * 0.15)
    ## 检测是否跨行
    text_string_list = []
    text_lines = wrap(random_string, CHAR_LIMIT)
    text_string_list.extend(text_lines)
    total_text_width = 0
    total_text_height = 0
    for text_string in text_lines:
        text_width, text_height = get_text_dimensions(text_string, light_PIL_font)
        
        total_text_width = max(total_text_width, text_width)
        total_text_height = total_text_height + offset + text_height
            
    
    total_text_height = total_text_height + offset + 5
    txt_image = Image.new('RGB', (total_text_width, total_text_height), (255, 255, 255))
    
    draw = ImageDraw.Draw(txt_image)
    
    
    offset_y = 0
    for text_string in text_string_list:
        text_width, text_height = get_text_dimensions(text_string, light_PIL_font)
        start_x = total_text_width//2 - text_width // 2
        start_y = offset_y
        draw.text((start_x, start_y), text_string, font=light_PIL_font, fill=(0, 0, 0))
        offset_y = offset_y + offset + text_height
    
    box = (0, 0, total_text_width, total_text_height)
    txt_box = {
                "bbox":box,
                "category_id":2
            }
    
    return txt_image, txt_box

def paste_top_mid(ori_image, idx_image, idx_box, bboxes, proba=None):
    if proba is None:
        proba = random.random()
    
    ## offset
    
    offset_2 = random.randint(0, int(idx_image.size[1]*1))

    width = max(ori_image.size[0], idx_image.size[0])
    height = ori_image.size[1] + idx_image.size[1] + offset_2
    
    new_image = Image.new('RGB', (width, height), (255, 255, 255))
    ## 先贴索引
    new_image.paste(idx_image, (width//2 - idx_image.size[0]//2, 0))
    ## 再贴分子
    new_image.paste(ori_image, (width//2 - ori_image.size[0]//2, idx_image.size[1] + offset_2))

     ## 先添加坐标的offset
    for bbox in bboxes:
        box = bbox["bbox"]
        new_box = (width//2 - ori_image.size[0]//2 + box[0], box[1] + idx_image.size[1] + offset_2, box[2], box[3])
        bbox["bbox"] = new_box

    idx_box["bbox"] = (
        width//2 - idx_image.size[0]//2,
        idx_box["bbox"][1],
        idx_box["bbox"][2],
        idx_box["bbox"][3]
    )
    ## 修改mol的坐标
    bboxes.append(idx_box)

    return new_image, bboxes

def paste_top_left(ori_image, idx_image, idx_box, bboxes, proba=None):
    if proba is None:
        proba = random.random()
    
    ## offset
    offset_1 = random.randint(0, int(idx_image.size[0]*0.5))
    offset_2 = random.randint(0, int(idx_image.size[1]*0.5))
    width = ori_image.size[0] + idx_image.size[0] + offset_1
    height = ori_image.size[1] + idx_image.size[1] + offset_2
    
    new_image = Image.new('RGB', (width, height), (255, 255, 255))
    ## 先贴索引
    new_image.paste(idx_image, (0, 0))
    ## 再贴分子
    new_image.paste(ori_image, (idx_image.size[0] + offset_1, idx_image.size[1] + offset_2))

     ## 先添加坐标的offset
    for bbox in bboxes:
        box = bbox["bbox"]
        new_box = (box[0] + idx_image.size[0] + offset_1, box[1] + idx_image.size[1] + offset_2, box[2], box[3])
        bbox["bbox"] = new_box

    ## 修改mol的坐标
    bboxes.append(idx_box)

    return new_image, bboxes


def paste_right_mid(ori_image, idx_image, idx_box, bboxes):
    
    ## 右中
    offset = random.randint(0, int(idx_image.size[0]))
    offset_y = random.randint(-10, 10)
    width = ori_image.size[0] + idx_image.size[0] + offset
    height = max(ori_image.size[1], idx_image.size[1])
    
    ## 画布
    new_image = Image.new('RGB', (width, height), (255, 255, 255))
    
    # 将原图像粘贴到新图像中，保持原有位置不变
    ## 先贴分子(分子需要水平居中)
    new_image.paste(ori_image, (0, 0))
    ## 再贴索引
    new_image.paste(idx_image, (ori_image.size[0] + offset, height//2 - idx_image.size[1]//2 + offset_y))
    
    # temp_mol_box = mol_box.pop("bbox")
    # temp_mol_box = (0+temp_mol_box[1], 0+temp_mol_box[1], 
    #                 temp_mol_box[2], temp_mol_box[3])
    # mol_box["bbox"] = temp_mol_box

    ## 先添加坐标的offset
    for bbox in bboxes:
        box = bbox["bbox"]
        new_box = (box[0] + 0, box[1] + 0, box[2], box[3])
        bbox["bbox"] = new_box
    
    ## 添加额外的box
    if isinstance(idx_box, dict):
        if "bbox" in idx_box:
            idx_box_ = copy.deepcopy(idx_box)
            temp_idx_box = idx_box_.pop("bbox")
            temp_idx_box = (ori_image.size[0] + offset + temp_idx_box[0], 
                            height//2 - idx_image.size[1]//2 + offset_y + temp_idx_box[1], 
                            temp_idx_box[2], 
                            temp_idx_box[3])
            idx_box_["bbox"] = temp_idx_box
        
            bboxes.append(idx_box_)
    if isinstance(idx, list):
        for idx_box_ in idx_box:
            if "bbox" in idx_box_:
                idx_box_ = copy.deepcopy(idx_box)
                temp_idx_box = idx_box_.pop("bbox")
                temp_idx_box = (ori_image.size[0] + offset + temp_idx_box[0], 
                                height//2 - idx_image.size[1]//2 + offset_y + temp_idx_box[1], 
                                temp_idx_box[2], 
                                temp_idx_box[3])
                idx_box_["bbox"] = temp_idx_box
            
                bboxes.append(idx_box_)

    return new_image, bboxes

def paste_down_mid(ori_image, txt_image, txt_box, bboxes, proba=None):
    if proba is None:
        proba = random.random()
    
    offset = random.randint(0, 20)

    width = max(ori_image.size[0], txt_image.size[0])
    height = ori_image.size[1] + txt_image.size[1] + offset
    
    new_image = Image.new('RGB', (width, height), (255, 255, 255))
    
    ## 先贴分子
    new_image.paste(ori_image, (width//2-ori_image.size[0]//2, 0))
    
    ## 再贴文本
    new_image.paste(txt_image, (width//2-txt_image.size[0]//2, ori_image.size[1] + offset))

    ## 先添加offset
    for bbox in bboxes:
        box = bbox["bbox"]
        new_box = (box[0] + width//2-ori_image.size[0]//2, box[1], box[2], box[3])
        bbox["bbox"] = new_box
    
    temp_txt_box = txt_box.pop("bbox")

    temp_txt_box = (width//2-txt_image.size[0]//2+temp_txt_box[0], ori_image.size[1] + offset + temp_txt_box[1], 
                    temp_txt_box[2], temp_txt_box[3])
    txt_box["bbox"] = temp_txt_box
    
    bboxes.append(txt_box)

    return new_image, bboxes

def add_text_to_region(ori_image, text, text_width, text_height, PIL_font, offset_x, offset_y):
    temp_new_image = Image.new('RGB', (text_width+10, text_height+10), (255, 255, 255))
    temp_draw = ImageDraw.Draw(temp_new_image)
    temp_draw.text((0, 0), text, font=PIL_font, fill=(0, 0, 0))
    new_image = Image.fromarray(transform_fn_2(image=np.array(temp_new_image),keypoints=[])["image"])
    new_image = new_image.resize((int(text_width*0.85), int(text_height*0.85)))
    ori_image.paste(new_image, (offset_x + int(text_width*0.075), offset_y + int(text_height*0.075)))
    



def get_total_result(original_smiles:str,
                    proba_index:float, 
                    proba_text:float, 
                    scale:float,
                    bold_PIL_font:ImageFont.FreeTypeFont,
                    light_PIL_font:ImageFont.FreeTypeFont,
                    CHAR_LIMIT:int):
    """_summary_

    Args:
        original_smiles (str): smiles of molecule
        proba_index (float): 索引位置的概率，如果概率<0.5，按照常规出现在下方，如果概率<0.85出现在左上，否则出现在正上方
        proba_text (float): 文本位置的概率，如果概率<0.25,不出现，如果概率<0.75，出现在正下方，否则出现在右边
        scale (float): 分子缩放尺度
        bold_PIL_font (ImageFont.FreeTypeFont): 字体
        light_PIL_font (ImageFont.FreeTypeFont): 加粗字体
        CHAR_LIMIT (int): 每行字数的限制

    Returns:
        new_image (PIL.Image.Image) : 拼接后的分子图像
        bboxes (list(dict)): 比如
            ```
            [
            {'bbox': (0, 0, 974, 315), 'category_id': 1}, 
            {'category_id': 3, 'bbox': (359, 330, 256, 62)}, 
            {'category_id': 2, 'bbox': (247, 402, 480, 201)}
            ]
            ```
        其中1表示分子, 2表示文本，3表示索引
        
    """
    
    ## check mol is valid
    mol = None
    try:
        mol = Chem.MolFromSmiles(original_smiles)
    except:
        pass
    if mol is None:
        return None, None
    
    bboxes = []
    
    try:
        mol_image, mol_box = get_image_mol_box(original_smiles, scale)
    except:
        return None, None
    
    new_image = mol_image
    bboxes.append(mol_box)

    ## 获取分子的评述
    comment = get_comment(mol)

    idx_image, idx_box = get_new_idx_image_box(comment, bold_PIL_font, light_PIL_font)

    txt_image, txt_box = get_image_txt_box(comment, light_PIL_font, CHAR_LIMIT)
    
    ## 把索引粘贴在下方
    if proba_index<0.5:
        new_image, bboxes = paste_down_mid(new_image, idx_image, idx_box, bboxes)
    ## 把索引粘贴在左上
    elif proba_index<0.85:
        new_image, bboxes = paste_top_left(new_image, idx_image, idx_box, bboxes)
    ## 把索引粘贴在上中
    else:
        new_image, bboxes = paste_top_mid(new_image, idx_image, idx_box, bboxes)
    
    ## 不粘贴
    if proba_text<0.25:
        pass
    ## 中下
    elif proba_text<0.75:
        new_image, bboxes = paste_down_mid(new_image, txt_image, txt_box, bboxes)
    ## 右方
    else:
        new_image, bboxes = paste_right_mid(new_image, txt_image, txt_box, bboxes)

    return new_image, bboxes

import io
import base64

def img2base64(image:Image):
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='PNG')
    image_bytes.seek(0)

    # # 将图像转换为字符流
    # image_stream = image_bytes.read()
    # 将图像转换为base64编码的字符串
    image_base64 = base64.b64encode(image_bytes.getvalue()).decode()
    return image_base64


df_solvent = pd.read_csv(os.path.join(main_dir, "chemistry_data/organic_solvent_2.csv"), index_col=0)
del df_solvent["Unnamed: 0"]
color_list_1 = ['red', 'blue', 'green', 'magenta', "purple"]
df_solvent = df_solvent[~df_solvent["canonical_smiles"].isna()]
df_compound = df_solvent[~df_solvent['canonical_smiles'].str.contains('\.')]
df_solvent = df_solvent.reset_index(drop=True)
PandasTools.AddMoleculeColumnToFrame(df_solvent,'canonical_smiles','Molecule')
df_solvent = df_solvent[~df_solvent["Molecule"].isna()]
df_solvent['MW'] = df_solvent['Molecule'].apply(lambda x:Chem.rdMolDescriptors.CalcExactMolWt(x))
df_solvent = df_solvent[df_solvent['MW']>=100]

font_size = 12
if __name__ == "__main__" :
    # 创建解析器
    parser = argparse.ArgumentParser(description="Specify the directory to save results.")
    
    # 添加参数
    parser.add_argument('--save_dir', type=str, default=None, help='save_dir')
    
    # 解析参数
    args = parser.parse_args()

    file_path = os.path.join(main_dir, "chemistry_data/train_200k.csv")
    df0 = pd.read_csv(file_path, index_col=0)
    # 过滤包含句点的数据
    df0 = df0[~df0['SMILES'].str.contains('\.')]
    df1 = df0.sample(500)
    df1 = df1.reset_index(drop=True)
    df1 = df1[df1["num_atoms"]<=35]
    df1 = df1[~df1['SMILES'].str.contains('\.')]
    df_temp = pd.DataFrame(df_solvent["canonical_smiles"])
    df_temp.columns = ["SMILES"]
    df_temp["pubchem_cid"] = np.arange(len(df_temp))
    df1 = pd.concat([df1, df_temp])
    
    df_long_chain = pd.read_csv(os.path.join(main_dir, "chemistry_data/long_chain.csv"))
    for smiles in df_long_chain.smiles:
        df1.loc[len(df1), "SMILES"] = smiles
    df1 = df1[~df1['SMILES'].str.contains('\.')]
    df1 = df1.reset_index(drop=True)
    
    print("len(df):",len(df1))
        
    i = 0
    mode_1_flag = True
    total_result = []
    
    idx = 0
    debug = True
    
    if args.save_dir is None or os.path.exists(args.save_dir) is False:
        save_dir = os.path.join(main_dir, "result", "mol_text_1116")
    else:
        save_dir = args.save_dir

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "img"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "label"), exist_ok=True)
    
    
    for i in tqdm(range(0, len(df1), 4)):
        bbox = []
        corefs = []
        source_smiles = []
        
        proba_index = random.random()
        proba_text = random.random()
        proba_concat_direction = random.random()
        
        CHAR_LIMIT = random.randint(20, 35)
        if random.random()<0.5:
            bold_PIL_font = ImageFont.truetype(os.path.join(main_dir,"font","FreeSerifBold.ttf"), #"FreeSerif-4aeK.ttf"
                                        size=int(font_size* 4 * random.choice([0.85, 1.0, ])))
            light_PIL_font = ImageFont.truetype(os.path.join(main_dir,"font","FreeSerif-4aeK.ttf"), 
                                            size=int(bold_PIL_font.size*random.choice([0.8, 0.85, 0.9, 0.95, 1.0])))
        else:
            bold_PIL_font = ImageFont.truetype(os.path.join(main_dir,"font","Arial-Unicode-Bold.ttf"), #"FreeSerif-4aeK.ttf"
                                        size=int(font_size* 4 * random.choice([0.85, 1.0, ])))
            light_PIL_font = ImageFont.truetype(os.path.join(main_dir,"font","Arial Unicode MS.TTF"), 
                                            size=int(bold_PIL_font.size*random.choice([0.8, 0.85, 0.9, 0.95, 1.0])))
        
        scale = random.choice([0.8, 0.9, 1.0])
        
        new_image_list = []
        result_list = []
        if (i+4) >= len(df1):
            continue
        
        for _ in range(i, i+4):
            original_smiles = df1.loc[_, "SMILES"]
            source_smiles.append(original_smiles)
            new_image, result = get_total_result(original_smiles,
                                                proba_index, 
                                                proba_text, 
                                                scale,
                                                bold_PIL_font,
                                                light_PIL_font,
                                                CHAR_LIMIT)
            
            if new_image is not None:
                new_image_list.append(new_image)
                result_list.append(result)
        
        offset = random.randint(50, 150)
        
        if proba_concat_direction<0.5:
            width = 0
            height = 0
            for _ in range(len(new_image_list)):
                width = width + offset + new_image_list[_].size[0]
                height = max(new_image_list[_].size[1], height)
            
            new_new_image = Image.new('RGB', (width, height), (255, 255, 255))
            x_offset = 0
            y_offset = 0
            count = 0
            bboxes = []
            corefs = []
            for _ in range(len(new_image_list)):
                new_image = new_image_list[_]
                temp_bboxes = result_list[_]
                
                ## 先贴分子
                new_new_image.paste(new_image, (x_offset, height//2 - new_image.size[1]//2))
                
                coref = []
                for temp_box_dict in temp_bboxes:
                    temp_box = temp_box_dict["bbox"]
                    temp_box = (
                        temp_box[0] + x_offset, temp_box[1] + y_offset + height//2 - new_image.size[1]//2,
                        temp_box[2], temp_box[3]
                    )
                    temp_box_dict["bbox"] = temp_box
                    
                    temp_box_dict["idx"] = count
                    coref.append(count)
                    count  = count + 1
                    
                bboxes.extend(copy.deepcopy(temp_bboxes))
                corefs.append(copy.deepcopy(coref))
                
                x_offset = x_offset + offset + new_image.size[0]
        
        else:
            width = 0
            height = 0
            for _ in range(len(new_image_list)):
                width = max(new_image_list[_].size[0], width)
                height = height + offset + new_image_list[_].size[1]
            
            new_new_image = Image.new('RGB', (width, height), (255, 255, 255))
            x_offset = 0
            y_offset = 0
            count = 0
            bboxes = []
            corefs = []
            for _ in range(len(new_image_list)):
                new_image = new_image_list[_]
                temp_bboxes = result_list[_]
                
                ## 先贴分子
                new_new_image.paste(new_image, (width//2 - new_image.size[0]//2, y_offset))
                
                coref = []
                for temp_box_dict in temp_bboxes:
                    temp_box = temp_box_dict["bbox"]
                    temp_box = (
                        temp_box[0] + width//2 - new_image.size[0]//2, temp_box[1] + y_offset,
                        temp_box[2], temp_box[3]
                    )
                    temp_box_dict["bbox"] = temp_box
                    
                    temp_box_dict["idx"] = count
                    coref.append(count)
                    count  = count + 1
                
                bboxes.extend(copy.deepcopy(temp_bboxes))
                corefs.append(copy.deepcopy(coref))
                
                y_offset = y_offset + offset + new_image.size[1]
        
        result = {
            "bboxes": bboxes,
            "corefs": corefs,
        }

        debug = True
        if debug:
            import copy
            temp_image = copy.deepcopy(new_new_image)
            draw = ImageDraw.Draw(temp_image)
            color_list_1 = ['red', 'blue', 'green']
            color_list_2 = ['magenta', "purple"]
            for k, coref in enumerate(result["corefs"]):
                for _ in coref:
                    x1, y1, w, h = result["bboxes"][_]["bbox"]
                    x2, y2 = x1 + w, y1 + h
                    rectangle_coords = [(x1, y1), (x2, y2)]
                    draw.rectangle(rectangle_coords, outline=color_list_1[k%(len(color_list_1))], width=3)
            temp_image.save("temp_2.png")
        
        
        save_path = os.path.join(save_dir, "img", f"{idx}.png")
        new_new_image.save(save_path)

        labelme_data = {
            'version': "5.2.1",
            'flags': {},
            'shapes': [],
            'imagePath': os.path.basename(save_path),  # 图像路径
            'imageData': img2base64(new_new_image),  # 图像数据，如果有可以填入Base64编码
            'imageHeight': new_new_image.size[1],  # 图像的高度
            'imageWidth': new_new_image.size[0]  # 图像的宽度
        }


        labelme_data["shapes"] = []
        shapes = []
        for temp_result in result['bboxes']:
            bbox = temp_result["bbox"]
            shapes.append(
                            {"label": f'{temp_result["category_id"]}', 
                            "points": [[bbox[0], bbox[1]], [bbox[0]+bbox[2], bbox[1]+bbox[3]]], 
                            "group_id": None, 
                            "shape_type": "rectangle", 
                            "flags": {}}
                        )
        labelme_data["shapes"] = shapes
        
        labelme_save_path = save_path.split(".")
        labelme_save_path[-1] = "json"
        labelme_save_path = ".".join(labelme_save_path)
        with open(labelme_save_path, "w") as f:
            f.write(json.dumps(labelme_data))
        
        idx = idx + 1