import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys

main_dir = os.path.dirname(os.path.abspath(__file__))
print("main_dir", main_dir)
sys.path.append(main_dir)

import rdkit
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
import random
import cairosvg
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import io
import base64
import pandas as pd
from rdkit.Chem import PandasTools
import copy
import cv2
from img2table import document
import numpy as np
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
import math
from textwrap import wrap
import copy
from vis_utils import vis_table
from tqdm import tqdm
import json
import albumentations as A
import imgkit
import argparse

## 找到最大非空白部分的函数
#ref:https://github.com/thomas0809/MolScribe/blob/7296a30413eb55436702011efdff78131f66d162/molscribe/augment.py#L97
## to align boderless table and bodered table
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
        x = (img != self.value).sum(axis=2)
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

## 找到最大非空白部分的函数
def get_transforms_fn(pad=10):
    trans_list = []
    trans_list.append(CropWhite(pad=pad))
    return A.Compose(trans_list, keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
transform_fn = get_transforms_fn(pad=10)
transform_fn_2 = get_transforms_fn(pad=2)

## 生成标签
import string
letters = string.ascii_lowercase ## 26个字符
roman_num_list = [u"\u2163", u"\u2164", u"\u2165", u"\u2166", u"\u2167", u"\u2168", u"\u2169"] ## 罗马数字
def get_index():
    ## bracket_flag：是否已经添加括号
    bracket_flag = False
    result1 = ""
    if random.random()<0.5:
        result1 = random.choice(roman_num_list)
    else:
        if random.random()<0.5:
            ## 随机数字
            result1 = str(random.randint(1,1000))
            if random.random() > 0.5:

                ## 首字母
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
            ## 随机数字
            result1 = str(random.randint(1,1000))
            if random.random() > 0.5:
                ## 随机罗马数字
                roman_num = random.choice(roman_num_list)
                if random.random() < 0.25:
                    roman_num =  "(" + roman_num + ")"
                    bracket_flag = True

                ## 添加空格
                if random.random()<0.5:
                    result1 += " "+ roman_num
                else:
                    result1 += roman_num
    
    if random.random()<0.1 and (bracket_flag is False):
        result1 = "(" + result1 + ")"
            
    ## 添加±
    if random.random()<0.1:
        if random.random() < 0.5:
            result1 = u'\u00B1' + " "+ result1
        else:
            result1 = "(" + u'\u00B1' + ") " + result1
    
    return result1

def get_end_atoms_idx(mol,
                    only_single_bond=True, 
                    zero_charge=False):
    """获取末端的原子

    Args:
        mol (rdkit.Chem.rdchem.Mol): 分子
        only_single_bond (bool, optional): 是否只考虑单键. Defaults to True.

    Returns:
        end_atom_idx_list(list(int)): 末端原子序号的列表
    """
    # 获取末端原子序号
    end_atom_idx_list = []
    ## 遍历
    for atom in mol.GetAtoms():
        if zero_charge:
            atom_charge = atom.GetFormalCharge()
            if atom_charge!=0:
                continue

        neighbors = atom.GetNeighbors()
        ## 末端键的特征，只有一个邻居节点
        if len(neighbors) == 1:
            ## 是否只考虑单键的情况
            if only_single_bond:
                bond = mol.GetBondBetweenAtoms(neighbors[0].GetIdx(), atom.GetIdx())
                if bond.GetBondType() == rdkit.Chem.rdchem.BondType.SINGLE:
                    end_atom_idx_list.append(atom.GetIdx())
            else:
                end_atom_idx_list.append(atom.GetIdx())
    
    return end_atom_idx_list

def star_replace(mol):
    """对分子中的原子进行替换
        替换的规则为：随机选择末端原子进行替换

    Args:
        mol (rdkit.Chem.rdchem.Mol): 分子

    Returns:
        mol (rdkit.Chem.rdchem.Mol): 替换后的分子
    """

    ## 寻找末端的原子
    end_atoms_idx = get_end_atoms_idx(mol, only_single_bond=True)

    ## 随机选择替换个数
    if len(end_atoms_idx)>=2:
        select_nums = random.randint(1, len(end_atoms_idx))
    else:
        select_nums = 1
    
    ## 只选择符合条件的进行替换
    select_atom_idx = []
    random.shuffle(end_atoms_idx)
    _ = 0
    while (_<len(end_atoms_idx)) and (len(select_atom_idx)<select_nums):

        atom_idx = end_atoms_idx[_]
        atom = mol.GetAtomWithIdx(atom_idx)

        ## 确保电荷为0以及不是dummy原子占位
        if atom.GetFormalCharge()!=0:
            pass
        elif atom.GetSymbol()=="*":
            pass
        else:
            select_atom_idx.append(atom_idx)

        _ = _ + 1

    # 替换
    for atom_idx in select_atom_idx:
        if atom.GetFormalCharge()!=0:
            continue
        if atom.GetSymbol()=="*":
            continue
        ## 随机获取替换的symbol
        atom = mol.GetAtomWithIdx(atom_idx)
        ## 将替换的原子变为*
        ## Chem.GetPeriodicTable().GetAtomicNumber(dummy_symbol) = 0
        atom.SetNumExplicitHs(0)
        atom.SetAtomicNum(0)
        atom.SetFormalCharge(0)
    return mol

def random_char_replace(mol):
    """对分子中的原子进行替换
        替换的规则为：随机选择末端原子进行替换

    Args:
        mol (rdkit.Chem.rdchem.Mol): 分子

    Returns:
        mol (rdkit.Chem.rdchem.Mol): 替换后的分子
    """

    ## 寻找末端的原子
    end_atoms_idx = get_end_atoms_idx(mol, only_single_bond=True)

    ## 随机选择替换个数
    if len(end_atoms_idx)>=2:
        select_nums = random.randint(1, len(end_atoms_idx))
    else:
        select_nums = 1
    
    ## 只选择符合条件的进行替换
    select_atom_idx = []
    random.shuffle(end_atoms_idx)
    _ = 0
    while (_<len(end_atoms_idx)) and (len(select_atom_idx)<select_nums):

        atom_idx = end_atoms_idx[_]
        atom = mol.GetAtomWithIdx(atom_idx)

        ## 确保电荷为0以及不是dummy原子占位
        if atom.GetFormalCharge()!=0:
            pass
        elif atom.GetSymbol()=="*":
            pass
        else:
            select_atom_idx.append(atom_idx)

        _ = _ + 1

    # 替换
    for atom_idx in select_atom_idx:
        if atom.GetFormalCharge()!=0:
            continue
        if atom.GetSymbol()=="*":
            continue
        ## 随机获取替换的symbol
        atom = mol.GetAtomWithIdx(atom_idx)
        ## 将替换的原子变为*
        ## Chem.GetPeriodicTable().GetAtomicNumber(dummy_symbol) = 0
        atom.SetNumExplicitHs(0)
        atom.SetAtomicNum(random.choices([1,6,7,8,15,16,35,53], weights=[0.15,0.15,0.15,0.15,0.1,0.1,0.1,0.05])[0])
        atom.SetFormalCharge(0)
    return mol


def svg2img(svgstring: str):
    """将SVG的字符串转化为图像

    Args:
        svgstring (str): SVG的字符串

    Returns:
        PIL.Image.image: 图像
    """
    png_data = cairosvg.svg2png(bytestring=svgstring)

    # 将PNG数据加载到Pillow的Image对象中
    image = Image.open(BytesIO(png_data))
    return image

def get_logP():
    """随机生成logP

    Returns:
        float: 随机生成logP数值
    """
    return -round(random.randint(-1000, 1000)*0.01, 2)

def get_logS():
    """随机生成logS

    Returns:
        float: 随机生成logS数值
    """
    return round(random.randint(-1000, 1000)*0.01, 2)

## 获取分子式
def get_formular(mol):
    """生成分子的分子式

    Args:
        mol (Chem.rdchem.Mol): 分子

    Returns:
        formula (str): 分子式
    """
    # 计算分子式
    formula = CalcMolFormula(mol) * random.randint(1, 2)
    return formula

def get_smiles(mol):
    """获取分子的smiles

    Args:
        mol (Chem.rdchem.Mol): 分子

    Returns:
        smiles (str): 分子的smiles式
    """
    return Chem.MolToSmiles(mol)

def get_activity():
    """随机生成分子的活性数据

    Returns:
        result1 (str): 分子的活性数据
    """
    proba_1 = random.random()
    ## sitation 1
    if proba_1 <= 0.33:
        result1 = random.choice(["Kd","IC50","Kd"]) + " : " + str(round(random.randint(1,10000)*0.1, 3)) + random.choice(["nm","um"])
    ## situation 2
    elif proba_1 <= 0.66:
        # u"\u25B3" = △; u"\u2103"=℃
        result1 = u"\u25B3"+"T"+":"+ str(random.randint(-10,100)) + u"\u2103"
    ## situation 3
    else:
        result1 = random.choice(["-","+"]) + u"\u25B3"+"G"+" : "+ str(round(random.randint(-100,100)*0.1, 2)) + " Kcal/mol"
    return result1

def get_yield():
    result1 = random.choice(["yield","ee","E/Z","S/R"]) +" : "+ str(round(random.randint(0, 1000)*0.1, 2)) + "%"
    return result1

def get_quantity(): 
    result1 = f"{round(random.choice([0.5, 1.0, 1.5, 2.0, 10.0, 5.0]), 1)}" +" equiv"
    return result1


def draw_molecule(mol, size=384, 
                is_black=True, 
                is_wave_line=False, 
                with_coord=False, 
                with_wedge=False,
                highlight=False,
                angle=None,
                debug=False):
    """生成分子的图片

    Args:
        mol (rdkit.Chem.rdchem.Mol): 分子
        size (int, optional): 图片的尺寸. Defaults to 384.
        is_black (bool, optional): 是否使用rdkit的黑白风格绘图，默认为True. Defaults to True.
        is_wave_line (bool, optional): 是否对R_group使用波浪线的绘图风格. Defaults to False.
        with_coord (bool, optional): _description_. Defaults to True.
        with_wedge (bool, optional): 是否提前计算好了手性的键. Defaults to True.
        random_font_type (bool, optional): 是够使用随机的字体类型. Defaults to True.
        add_note (bool, optional): 是否使用随机的字体类型. Defaults to True.
        refine_charge (bool, optional): 是否使用unicode编码的charge. Defaults to False.

    Returns:
        image (PIL.Image.image) : 图像
    """

    if (with_coord is False):
        ## 先计算好手性键
        if with_wedge is False:
            rdDepictor.Compute2DCoords(mol, clearConfs=True) #clearConfs=True会改变构像
            try:
                ps = Chem.BondWedgingParameters()
                ps.wedgeTwoBondsIfPossible = True
                Chem.WedgeMolBonds(mol, mol.GetConformer(), ps)
            except:
                print("手信键不可应用")
        
        ## 绘图
        # (width, height)
        d = rdMolDraw2D.MolDraw2DSVG(-1,size)
        dopts = rdMolDraw2D.MolDrawOptions()
        if is_black:
            dopts.useBWAtomPalette()

        dopts.scaleBondWidth = True
        dopts.bondLineWidth = 3.5#random.randint(4, 7)*0.5
        dopts.rotate = random.choice([random.randint(-45, 45), random.randint(-135, 215)]) if angle is None else angle
        
        ## 字体大小
        dopts.minFontSize = -1
        dopts.maxFontSize = -1

        # 绘制波浪线的选项
        if is_wave_line:
            dopts.dummiesAreAttachments = True
        
        if random.random()>0.9:
            dopts.addAtomIndices = True

        d.SetDrawOptions(dopts)
        rdMolDraw2D.PrepareAndDrawMolecule(d, mol)

        d.FinishDrawing()

        drawing = d.GetDrawingText()

    else:
        d = rdMolDraw2D.MolDraw2DSVG(-1, size)
        dopts = d.drawOptions()
        ## 高亮模块的设置
        if highlight is False:
            ## 关闭高亮模块
            dopts.variableAtomRadius = 0
            dopts.variableBondWidthMultiplier = 0
        else:
            dopts.variableAtomRadius = random.choices([0.2, 0.4, 0.6, 0.8])
            dopts.variableBondWidthMultiplier = random.choices([20, 40, 60, 80])
            dopts.setVariableAttachmentColour = random.choices([(1.0, 0.5, 0.5), (0.75, 0.75, 0.75)])
            
        dopts.annotationFontScale = 0.8
        dopts.useMolBlockWedging = True
        # dopts.bondLineWidth = random.randint(2, 5)*0.5
        dopts.bondLineWidth = 3.5
        dopts.multipleBondOffset = random.randint(1, 6)*0.05
        dopts.rotate = random.choice([random.randint(-45, 45), random.randint(-135, 215)])
        dopts.scaleBondWidth = True
        ## https://github.com/rdkit/rdkit/issues/2496
        ## 原子序号
        if random.random()>0.5:
            dopts.includeAtomNumbers = True
        ## 显式甲基
        if random.random()>0.5:
            dopts.explicitMethyl = True
        ## 手信符号
        if random.random()>0.5:
            dopts.includeChiralFlagLabel = True
        
        ## 
        if random.random()>0.75:
            dopts.addStereoAnnotation = True
        if random.random()>0.75:
            dopts.simplifiedStereoGroupLabel = True
        if random.random()>0.9:
            dopts.unspecifiedStereoIsUnknown = True
        
        # 原子序号
        if random.random()>0.9:
            dopts.addAtomIndices = True
        
        ## 字体大小
        dopts.minFontSize = -1
        dopts.maxFontSize = -1
        
        ## 手性键
        if with_wedge is False:
            try:
                ps = Chem.BondWedgingParameters()
                ps.wedgeTwoBondsIfPossible = True
                Chem.WedgeMolBonds(mol, mol.GetConformer(), ps)
            except:
                pass
        if is_black:
            dopts.useBWAtomPalette()
        # 绘制波浪线的选项
        if is_wave_line:
            dopts.dummiesAreAttachments = True
        
        if with_coord is False:
            rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
        else:
            d.DrawMolecule(mol)
        d.FinishDrawing()
        drawing = d.GetDrawingText()
    
    # drawing = drawing.replace("fill:#FFFFFF;", "fill:transparent;")
    image = svg2img(drawing)

    return image


def get_borderless_html(html_content:str="", mode:int=0, pad:int=0, horizen_pad:int=0):
    """"""
    if mode == 0:
        temp_string = "td {vertical-align: top;}"
    else:
        temp_string = "td {vertical-align: middle;}"
        
    if random.random()<0.75:
        temp_string += """
        tr:last-child td {
            border-bottom: 1px solid black; /* 设置最后一行的下边框 */
        }
        """
    
    vertical_flag = False
    if random.random()<0.5:
        if random.random()>-1:
            temp_string += """
            td {
                border-left: 1px solid white; /* 右边框 */
                border-right: 1px solid white; /* 右边框 */
                border-bottom: 1px solid black; /* 下边框 */
                border-top: 1px solid black; /* 上边框 */
                padding: %dpx
            }
            """%(pad)
        else:
            vertical_flag = True
            temp_string += """
            td {
                border-bottom: 1px solid white; /* 下边框 */
                border-top: 1px solid white; /* 下边框 */
                border-left: 1px solid black; /* 下边框 */
                border-right: 1px solid black; /* 下边框 */
                padding: %dpx
            }
            """%(pad)
    else:
        ## 真* 三线表
        temp_string += """
            td {
                border-left: 1px solid white; /* 右边框 */
                border-right: 1px solid white; /* 右边框 */
                border-bottom: 1px solid white; /* 下边框 */
                border-top: 1px solid white; /* 上边框 */
                padding: %dpx
            }
            """%(pad)


    
    if random.random()<0.8:
        temp_string += """
        th {
            border-top: 2px solid black; /* 设置表头的上边框 */
            border-bottom: 2px solid black; /* 设置表头的下边框 */
        }
        """
        if vertical_flag:
            temp_string += """
        th {
            border-left: 2px solid black; /* 设置表头的上边框 */
            border-right: 2px solid black; /* 设置表头的下边框 */
        }
        """
            

    else:
        color = "gray" #random.choice(["gray","lightgray"])
        temp_string += """
        th {
            border-top: 2px solid %s; /* 设置表头的上边框 */
            border-bottom: 2px solid %s; /* 设置表头的下边框 */
            color:white; /* 设置字体为白色 */
        }
        """%(color, color)
        
        temp_string += """
        th {
            background-color: %s; /* 设置成灰色表头的上边框 */
            border-left: 2px solid %s; /* 左边框 */
            border-right: 2px solid %s; /* 右边框 */
        }
        """%(color, color, color)
    
    temp_string += """
        th, td {
            padding-left: %dpx;
            padding-right: %dpx;
        }
        """%(max(pad, horizen_pad), max(pad, horizen_pad))
    
    
    html_content_with_css = """
    <html>
    <head>
        <style>
        table {
            max-width:1200px;
            border: 0px solid black; /* 设置表格外边框 */
            border-collapse: collapse; /* 合并表格边框 */
            height: auto;
            white-space:normal;
        }
        
        th, td {
            border: 1px solid white;
            text-align: center; /* 将内容居中对齐 */
            text-overflow: ellipsis;
            }
        
        img {
        transform: scale(1); /* 将图像放大1.5倍 */
        z-index: -99; /* 设置 z-index 为 -99 */
        }
        """ + temp_string + \
    """
        </style>
    </head>
    <body>
    """+ html_content +\
    """
    </body>
    </html>
    """
    return html_content_with_css



def get_border_html(html_content, mode=0, pad=0, horizen_pad=0):
    if mode == 0:
        temp_string = "td {vertical-align: top;}"
    else:
        temp_string = "td {vertical-align: middle;}"
    temp_string += """
        td {
            border-bottom: 1px solid black; /* 下边框 */
            border-top: 1px solid black; /* 上边框 */
            border-left: 1px solid black; /* 左边框 */
            border-right: 1px solid black; /* 右边框 */
            padding: %dpx
        }
        """%(pad)
    
    temp_string += """
        th, td {
            padding-left: %dpx;
            padding-right: %dpx;
        }
        """%(max(pad, horizen_pad), max(pad, horizen_pad))
    
    html_content_with_css = """
    <html>
    <head>
        <style>
        table {
            max-width:1200px;
            border: 0px solid black; /* 设置表格外边框 */
            border-collapse: collapse; /* 合并表格边框 */
            height: auto;
            white-space:normal;
        }
        
        th, td {
            border: 1px solid black;
            text-align: center; /* 将内容居中对齐 */
            text-overflow: ellipsis;
            }
            
        th {
            border-top: 2px solid black; /* 设置表头的上边框 */
            border-bottom: 2px solid black; /* 设置表头的下边框 */
        }
        img {
        transform: scale(1); /* 将图像放大1.5倍 */
        z-index: -99; /* 设置 z-index 为 -99 */
        }
        """ + temp_string + \
    """
        </style>
    </head>
    <body>
    """+ html_content +\
    """
    </body>
    </html>
    """
    return html_content_with_css


def img2base64(image:Image):
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='PNG')
    image_bytes.seek(0)

    # # 将图像转换为字符流
    # image_stream = image_bytes.read()
    # 将图像转换为base64编码的字符串
    image_base64 = base64.b64encode(image_bytes.getvalue()).decode()
    return image_base64

def get_mol_image(mol, size=-1, angle=None):
    temp_mol = copy.deepcopy(mol)
    end_atoms = get_end_atoms_idx(temp_mol)
    if random.random()<0.75 and len(end_atoms)>0:
        temp_mol = star_replace(temp_mol)
        image = draw_molecule(temp_mol, size=size, is_wave_line=True, angle=angle)
    else:
        image = draw_molecule(mol, size=size, angle=angle)
        
    # 将 PIL 图像转换为字节流
    byte_stream = io.BytesIO()
    image.save(byte_stream, format='PNG')
    byte_stream.seek(0)

    # 将字节流转换为 Base64 编码字符串
    base64_data = base64.b64encode(byte_stream.read()).decode('utf-8')

    # 创建包含数据 URI 的 HTML 内容
    html_content = f'<img src="data:image/png;base64,{base64_data}" alt="Molecule Image">'
    
    return html_content


def df_2_html(df, borderless=False, mode=0):
    html = df.to_html(index=False).replace("&lt;","<").replace("&gt;",">")
    if borderless:
        html = get_borderless_html(html, mode)
    else:
        html = get_border_html(html, mode)
    return html

def html_to_image_V2(html_content, output_path):
    imgkit.from_string(html_content, output_path)


def get_table_from_image(image:Image):
    ## 全局的表格
    # 将图像保存在io.BytesIO中
    image_io = io.BytesIO()
    image.save(image_io, format='PNG')
    image_io.seek(0)  # 这将让你可以从io.BytesIO对象的开始处读取数据
    
    ## 抽取表格
    img = document.Image(src=image_io)
    extracted_tables = img.extract_tables()
    return extracted_tables

def get_scale_image(image:Image, scale=1.0):
    target_width = int(image.size[0] * scale)
    target_height = int(image.size[1] * scale)
    # 缩放图像
    image = image.resize((target_width, target_height))
    return image

import string
import random
def generate_random_text(length):
    characters = string.ascii_letters + string.digits  # 包含所有字母和数字的字符串
    random_text = ''.join(random.choice(characters) for _ in range(length))  # 从字符集中随机选择字符，重复 length 次
    return random_text

## 缩写表
abbre_dict = {}
with open(os.path.join(main_dir, "chemistry_data", "abbretion.txt"), "r") as f:
    for line in f.readlines():
        line = line.rstrip("\n")
        line_list = line.split()
        if len(line_list) == 2:
            key,value = line_list
            if "*" not in value:
                abbre_dict[key] = value

abbre_dict["H"] = "H"
                
with open(os.path.join(main_dir, "chemistry_data", "FunctionalGroups.txt"), "r") as f:
    for line in f.readlines():
        line = line.rstrip("\n")
        if len(line)>0:
            if line[0] == "-":
                line_list = line.split()
                ## 只要某种固定模式的
                if len(line_list)==3:
                    abbre_dict[line_list[-1]] = line_list[0]
                    

def generate_random_Chemistry_text(with_prefix=False):
    chem_string = random.choice(list(abbre_dict.keys()))
    if with_prefix:
        chem_string = "-" + chem_string
    i = 0
    chem_string_result = []
    while i < len(chem_string):
        if chem_string[i].isdigit():
            j = i + 1
            while j < len(chem_string):
                if chem_string[j].isdigit():
                    j = j + 1
                else:
                    break
            chem_string_result.append("<sub>"+chem_string[i:j]+"</sub>")
            i = j
        else:
            chem_string_result.append(chem_string[i])
            i = i + 1
    return "".join(chem_string_result)
    
            

def get_text_dimensions(text_string, font):
    # https://stackoverflow.com/a/46220683/9263761
    ascent, descent = font.getmetrics()
    text_width = font.getmask(text_string).getbbox()[2]
    text_height = font.getmask(text_string).getbbox()[3] + descent
        
    return (text_width, text_height)



font_size = 12

drug_name_df = pd.read_csv(os.path.join(main_dir,"chemistry_data","strata-drug-list.csv"))
def get_random_drug_name(capitalize=False):
    count = 0
    while count<10:
        count = count + 1
        random_idx = random.randint(0, len(drug_name_df)-1)
        drug_name = drug_name_df.loc[random_idx, "drug"]
        if len(drug_name)<15:
            break
        if count == 10:
            drug_name = str(random.randint(0, 100))
            break
        
    if capitalize:
        drug_name = drug_name.capitalize()
    return drug_name



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default="table_11", help="保存文件的目录")
    parser.add_argument('--with_multi', type=int, default=1, help="保存文件的目录")
    args = parser.parse_args()

    args.save_dir = os.path.join("result", args.save_dir)

    if args.with_multi == 1:
        args.with_multi = True
    else:
        args.with_multi = False

    df_solvent = pd.read_csv(os.path.join(main_dir, "chemistry_data/organic_solvent_2.csv"), index_col=0)
    del df_solvent["Unnamed: 0"]
    df_solvent = df_solvent[~df_solvent["canonical_smiles"].isna()]
    df_solvent = df_solvent[~df_solvent['canonical_smiles'].str.contains('\.')]
    df_solvent = df_solvent.reset_index(drop=True)
    PandasTools.AddMoleculeColumnToFrame(df_solvent,'canonical_smiles','Molecule')
    df_solvent = df_solvent[~df_solvent["Molecule"].isna()]
    idx_list = []
    for idx in df_solvent.index:
        mol = df_solvent.loc[idx, "Molecule"]
        if len(mol.GetAtoms()) >=4:
            idx_list.append(idx)
    df_solvent = df_solvent.loc[idx_list]
    df_solvent = df_solvent.reset_index(drop=True)
    
    df_compound = None
    df_compound = pd.read_csv(os.path.join(main_dir, "chemistry_data/train_200k.csv"),  index_col=0)
    df_compound = df_compound[~df_compound['SMILES'].str.contains('\.')]
    condition1 = df_compound["num_atoms"]>=0
    condition2 = df_compound["num_atoms"]<=30
    df_compound = df_compound[(condition1 & condition2)]
    # df_compound = df_compound.sample(1000)
    df_compound = df_compound.reset_index(drop=True)
    df_compound = df_compound.reset_index(drop=False)
    
    PandasTools.AddMoleculeColumnToFrame(df_compound,'SMILES','Molecule')
    df_compound = df_compound[~df_compound["Molecule"].isna()]
    if "pubchem_cid" in df_compound.columns:
        del df_compound["pubchem_cid"]
    if "InChI" in df_compound.columns:
        del df_compound["InChI"]
    
    save_dir = args.save_dir #"/mnt/c/Users/wulin/Desktop/table_5"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "vis"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "temp_dir"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "img"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "label"), exist_ok=True)
    
    total_result = []
    df_solvent["Permeablity"] = ["%.2f/%.2f"%(random.random()*random.choice([0.1, 1, 10, 100, 100]), random.random()*random.choice([0.1, 1, 10, 100, 100])) for _ in range(len(df_solvent))]
    df_solvent["Activity"] = ["%s%.2f%s"%(random.choice(["","≈",">","≥",">>","<<","≤","<"]),random.random()*random.choice([0.1, 1, 10, 100, 100]), random.choice(["nM","uM"])) for _ in range(len(df_solvent))]
    columns_list = [temp for temp in df_solvent.columns.tolist() if temp != "Molecule"]
    unit_list = ["h","°C","10<sup>6<sup>cm/s","s", "%", "L/Kg","mg/Kg","min","ng/g", ""]
    for img_idx in tqdm(range(30000)):
        sample_columns_number = random.randint(2, 8)
        if sample_columns_number>0:
            new_columns_list = random.sample(columns_list, sample_columns_number)
        else:
            new_columns_list = []
        
        is_with_drug_name = random.random()<0.5
        if is_with_drug_name:
            capitalize = random.random()<0.5
        
        is_block = random.random()<0.5
        
        plot_mode = random.choices([1, 2, 3], weights=[2,4,0])[0]#去掉mode3，交替方式
        if plot_mode == 1:
            sample_number = random.randint(5, 75)
        elif plot_mode == 2:
            sample_number = random.randint(5, 25)
        else:
            sample_number = random.randint(5, 25)
        
        if plot_mode == 1:
            is_pure_num = random.random()<0.33
        if plot_mode == 2:
            txt_length = sample_number // random.randint(2, 4)
        if plot_mode == 3:
            mol_offset = random.randint(0, 1)
        
        if  plot_mode in [2, 3]:
            with_prefix = random.random()>0.5
            with_drug_name = random.random()>0.5
            
        temp_df = df_solvent.sample(sample_number)
        temp_df = temp_df.reset_index(drop=True)
        new_df = pd.DataFrame()

        mol_0 = df_solvent.loc[random.randint(0, len(df_solvent)-1), "Molecule"]
        use_mol_0 = False
        # mol_size = random.choice([150, 100, 200])
        if len(mol.GetAtoms())<=12:
            use_mol_0= random.random()<0.4
        if use_mol_0:
            mol_size = random.randint(30, 60)
        else:
            mol_size = random.randint(40, 150)
        
        for _, idx in enumerate(temp_df.index):
            if is_with_drug_name is False:
                new_idx: str = get_index()
                if is_block:
                    new_df.loc[_, "idx"] = "<b>"+new_idx+"</b>"
                else:
                    new_df.loc[_, "idx"] = new_idx
                    
            else:
                new_idx: str =  get_random_drug_name(capitalize=False)
                if random.random()<0.5:
                    if is_block:
                        new_df.loc[_, "mol"] = "<b>"+new_idx+"</b>"
                    else:
                        new_df.loc[_, "mol"] = new_idx
                else:
                    if is_block:
                        new_df.loc[_, "R"] = "<b>"+new_idx+"</b>"
                    else:
                        new_df.loc[_, "R"] = new_idx
            
            if plot_mode == 1:
                ## 纯数字模式
                if is_pure_num:
                    proba = random.random()
                    for column in new_columns_list:
                        num = round(random.randint(-10000, 10000) * 0.1, random.randint(0,2))
                        if proba<0.6:
                            temp_string = str(num)
                        elif proba < 0.7:
                            temp_string = random.choice(["++", "+"])
                        elif proba < 0.8:
                            temp_string = random.choice(["-", "--","None","/"])
                        else:
                            temp_string = "/"
                        new_df.loc[_, column] = temp_string
                else:
                    for column in new_columns_list:
                        if column in ['Functional Group', 'Molecular Mass (g/mol)']:
                            if random.random()<0.25:
                                new_df.loc[_, column] = random.choice(["-", "--","None", "/"])
                            else:
                                new_df.loc[_, column] = temp_df.loc[idx, column]
                        else:
                            new_df.loc[_, column] = temp_df.loc[idx, column]
                        
            elif plot_mode == 2:
                if _ < txt_length:
                    temp_tex = generate_random_Chemistry_text(with_prefix=True)
                    new_df.loc[_, "molecule"] = temp_tex
                else:
                    if use_mol_0:
                        temp_mol = copy.deepcopy(mol_0)
                        end_atoms = get_end_atoms_idx(mol)
                        if random.random()<0.75 and len(end_atoms)>0:
                            temp_mol = random_char_replace(temp_mol)
                            mol = copy.deepcopy(temp_mol)
                    else:
                        if random.random()<0.85:
                            mol = temp_df.loc[idx, "Molecule"]
                        else:
                            _idx = random.randint(0, len(df_compound)-1)
                            mol = df_compound.loc[_idx, "Molecule"]
                    try:
                        if use_mol_0:
                            new_df.loc[_, "molecule"] = get_mol_image(mol, mol_size, angle=0)
                        else:
                            ## 随机角度
                            new_df.loc[_, "molecule"] = get_mol_image(mol, mol_size)
                        
                    except:
                        new_df.loc[_, "molecule"] = random.choice(["-","None","/"])
                
                for column in new_columns_list:
                    if column in ['Functional Group', 'Molecular Mass (g/mol)']:
                        if random.random()<0.25:
                            new_df.loc[_, column] = new_df.loc[_, column] = random.choice(["-", "--","None","/"])
                        else:
                            new_df.loc[_, column] = temp_df.loc[idx, column]
                    else:
                        new_df.loc[_, column] = temp_df.loc[idx, column]
            
            elif plot_mode == 3:
                if (mol_offset + _)%2==0:
                    
                    if use_mol_0:
                        temp_mol = copy.deepcopy(mol_0)
                        end_atoms = get_end_atoms_idx(mol)
                        if random.random()<0.75 and len(end_atoms)>0:
                            temp_mol = random_char_replace(temp_mol)
                            mol = copy.deepcopy(temp_mol)
                    else:
                        if random.random()<0.85:
                            mol = temp_df.loc[idx, "Molecule"]
                        else:
                            _idx = random.randint(0, len(df_compound)-1)
                            mol = df_compound.loc[_idx, "Molecule"]
                    try:
                        if use_mol_0:
                            new_df.loc[_, "molecule"] = get_mol_image(mol, mol_size, angle=0)
                        else:
                            ## 随机角度
                            new_df.loc[_, "molecule"] = get_mol_image(mol, mol_size)
                    except:
                        new_df.loc[_, "molecule"] = random.choice(["-","None","/"])
                else:
                    temp_tex = generate_random_Chemistry_text(with_prefix)
                    new_df.loc[_, "molecule"] = temp_tex
        
                for column in new_columns_list:
                    if column in ['Functional Group', 'Molecular Mass (g/mol)']:
                        if random.random()<0.25:
                            new_df.loc[_, column] = random.choice(["-", "--","/","None",""])
                        else:
                            new_df.loc[_, column] = temp_df.loc[idx, column]
                    else:
                        new_df.loc[_, column] = temp_df.loc[idx, column]
        
        
        start_column_idx_list = []
        if random.random()<-1 and plot_mode==1:
            token = random.choice([""])
            if random.random()<0.5 and (len(new_df.columns)-2)>0:
                start_row_idx = random.randint(random.choice([len(new_df)//2, len(new_df)//3, len(new_df)//4]), len(new_df)-1)
                if len(new_df.columns)-2 == 1 or len(new_df.columns)-2 == 2:
                    start_column_idx_list = [len(new_df.columns)-1]
                else:
                    start_column_idx_list = random.sample([len(new_df.columns)-1, len(new_df.columns)-2, len(new_df.columns)-3], 2)
                
                for _ in new_df.index:
                    if _ >= start_row_idx:
                        for __ in start_column_idx_list:
                            if random.random()<0.6:
                                new_df.iloc[_, __] = token
        
        if random.random()>0.75:
            mode = 0
        else:
            mode = 1

        mol_idx = 1
        columns = new_df.columns.to_list()

        ##随机镂空
        temp_proba = random.random()
        ## 保持原来的就行了
        if temp_proba < 0.3:
            pass
        elif temp_proba < 0.6:
            ## 交换第一列（索引）和第二列（分子）
            columns[0], columns[1] = columns[1], columns[0]
            new_df = new_df[columns]
            mol_idx = 0
            
        else:
            count = 0
            while count<50:
                count = count + 1
                random_indx = random.randint(0, len(columns)-1)
                if random_indx in start_column_idx_list:
                    continue
                columns[random_indx], columns[mol_idx] = columns[mol_idx], columns[random_indx]
                new_df = new_df[columns]
                mol_idx = random_indx
        
        
        # len_header = 1#new_df.columns[0]
        ## 表头的长度
        len_header = random.choices([1, 2])[0]
        temp_columns = new_df.columns.to_list()
        if len_header == 2:
            random_columns = [(((temp_columns[i], 
                                 random.choices(["result<br>(T)", "activity(kd)", "inhibition(%)", "IC<sub>50<sub>", "\u25B3T(°C)", "entry", r"%inhibition", "yeild", "Kd", "Ki",
                                                "Result", "Actvity<br>(%s)"%(random.choice(["Ki","Kd","IC<sub>50<sub>"])), "Inhibition<br>(%)", "Entry", r"%Inhibition", "Yeild<br>(%)","ee(%)"],)[0])) ) for i in range(len(temp_columns))]
            # new_columns = pd.MultiIndex.from_tuples(random_columns)
            # new_df.columns = new_columns
            
        elif len_header == 1:
            random_columns = temp_columns

        new_temp_df = pd.DataFrame(random_columns).T
        new_temp_df.columns = new_df.columns.to_list()

        new_df = pd.concat([new_temp_df, new_df])
        new_df = new_df.reset_index(drop=True)

        merge_dict = {}
        ignore_list = []

        ## 合并索引
        proba = random.random()
        ## 合并表头的索引
        if proba<0.15 and len_header==2:
            merge_dict[(0,0)] = {"rowspan":f"{len_header}"}
            ignore_list.append((1,0))
        
        elif len_header>=3 and proba<0.15:
            start = random.randint(0,1)
            merge_dict[(start,0)] = {"rowspan":f"{2}"}
            ignore_list.append((start+1,0))
        
        
        nums_row, nums_col = new_df.shape
        # print("合并表头")
        # ## 合并表头
        j = 1
        if args.with_multi:
            factor = max(random.choice([1, 1.5, 2, 2.5, 3]), 1)
            while j < nums_col:
                i = 0
                merge_proba = random.random()
                merge_cols = 0
                if merge_proba>0: #需要修改的概率
                    merge_cols = random.choices([1,2,3,4,5],weights=[8,8,8,8,2])[0]
                    merge_cols = max(min(nums_col-1-j-1, merge_cols), 0)
                    if merge_cols == 0:
                        pass
                    else:
                        if (i,j) in merge_dict:
                            merge_dict[(i,j)]["colspan"] = f"{merge_cols+1}"
                        else:
                            merge_dict[(i,j)] = {"colspan":f"{merge_cols+1}"}
                        for temp_j in range(j+1, j+merge_cols+1):
                            ignore_list.append((i, temp_j))
                        
                        ## 随机替换分子
                        if random.random()<0.3:
                            temp_mol = df_compound.loc[random.randint(0, len(df_compound)-1), "Molecule"]
                            new_df.iloc[i, j] = get_mol_image(temp_mol, int(mol_size*factor))

                
                if merge_cols > 0:
                    i = i + 1
                    max_merge_cols = merge_cols
                    ## 遍历每一行
                    while i < len_header:
                        if max_merge_cols%6==0:
                            temp_merge_cols = max_merge_cols//random.randint(1,2,3)
                        elif max_merge_cols%2==0:
                            temp_merge_cols = max_merge_cols//random.randint(1,2)
                        elif max_merge_cols%3==0:
                            temp_merge_cols = max_merge_cols//random.randint(1,3)
                        elif max_merge_cols%5==0:
                            temp_merge_cols = max_merge_cols//random.randint(1,5)
                        else:
                            temp_merge_cols = max_merge_cols
                        
                        if random.random()<0.5:
                            if temp_merge_cols<max_merge_cols:
                                max_merge_cols = temp_merge_cols

                        if temp_merge_cols>0:
                            for temp_j in range(j, j+merge_cols+1, temp_merge_cols+1):
                                if temp_j+temp_merge_cols+1>j+merge_cols+1:
                                    continue
                                merge_proba_2 = random.random()
                                if merge_proba_2>0:#需要修改的概率
                                    if (i, temp_j) in merge_dict:
                                        merge_dict[(i, temp_j)]["colspan"] = f"{temp_merge_cols+1}"
                                    else:
                                        merge_dict[(i, temp_j)] = {"colspan":f"{temp_merge_cols+1}"}
                                    for temp_j_2 in range(temp_j+1, temp_j+temp_merge_cols+1):
                                        ignore_list.append((i, temp_j_2))
                        
                        i = i + 1
                
                j = j + merge_cols + 1
        
            
            ## 合并分子
            i = len_header + 1 if random.random()>0.5 else len_header
            # print("合并分子")
            start_j = random.choices([0, 1, 2], weights=[4,4,2])[0]
            while i < nums_row:
                j = start_j ## 从start_j开始
                merge_proba = random.random()
                merge_rows = 0
                if merge_proba>0:#需要修改的概率
                    merge_rows = random.choices([1,2,3,4,5],weights=[8,8,8,8,3])[0]
                    merge_rows = max(min(nums_row-1-i-1, merge_rows), 0)
                    if merge_rows==0:
                        pass
                    else:
                        if (i,j) in merge_dict:
                            merge_dict[(i,j)]["rowspan"] = f"{merge_rows+1}"
                        else:
                            merge_dict[(i,j)] = {"rowspan":f"{merge_rows+1}"}
                        for temp_i in range(i+1, i+merge_rows+1):
                            ignore_list.append((temp_i,j))
                
                if merge_rows > 0:
                    max_merge_rows = merge_rows
                    ## 遍历每一列
                    j = j + 1
                    while j < nums_col:
                        if max_merge_rows%6==0:
                            temp_merge_rows = max_merge_rows//random.randint(1,2,3)
                        elif max_merge_rows%2==0:
                            temp_merge_rows = max_merge_rows//random.randint(1,2)
                        elif max_merge_rows%3==0:
                            temp_merge_rows = max_merge_rows//random.randint(1,3)
                        elif max_merge_rows%5==0:
                            temp_merge_rows = max_merge_rows//random.randint(1,5)
                        else:
                            temp_merge_rows = max_merge_rows
                        
                        if random.random()<0.15:
                            if temp_merge_rows<max_merge_rows:
                                max_merge_rows = temp_merge_rows
                            

                        if temp_merge_rows>0:
                            for temp_i in range(i, i+merge_rows+1, temp_merge_rows+1):
                                merge_proba_2 = random.random()
                                if temp_i + temp_merge_rows+1 > i+merge_rows+1:
                                    continue
                                if merge_proba_2>0:#需要修改的概率
                                    if (temp_i, j) in merge_dict:
                                        merge_dict[(temp_i, j)]["rowspan"] = f"{temp_merge_rows+1}"
                                    else:
                                        merge_dict[(temp_i, j)] = {"rowspan":f"{temp_merge_rows+1}"}
                                    for temp_i_2 in range(temp_i+1, temp_i+temp_merge_rows+1):
                                        ignore_list.append((temp_i_2, j))
                                
                        
                        j = j + 1

                i = i + merge_rows + 1
                # print(i, merge_rows, nums_row)


        # print("table2html")
        nums_row, nums_col = new_df.shape
        table_html = "<table>"
        flag_tbody = False
        bold_header = random.random()>0.5
        max_count = random.randint(5,20)
        with_width = random.random()>0.75
        if with_width:
            random_mol_part = random.randint(0,2)
            if plot_mode == 2:
                width_percent = 100//(new_df.shape[1]+random_mol_part)+1
            else:
                mol_idx = None
                width_percent = 100//(new_df.shape[1])+1

        for row_idx in range(nums_row):
            if len_header>0:
                if row_idx==0:
                    table_html +="<thead>"
            if row_idx==len_header:
                table_html +="<tbody>"
                flag_tbody = True
            
            table_html += "<tr>"
            for col_idx in range(nums_col):
                if (row_idx, col_idx) in ignore_list:
                    # print(f"ignore ({row_idx, col_idx})")
                    continue

                cell_value = new_df.iloc[row_idx, col_idx]
                if cell_value is None or cell_value == "":
                    cell_value = "None"
                rowspan = "1"
                colspan = "1"

                if (row_idx, col_idx) in merge_dict.keys():
                    rowspan = merge_dict[(row_idx, col_idx)].get("rowspan", "1")
                    colspan = merge_dict[(row_idx, col_idx)].get("colspan", "1")

                css_class = f"cell-{row_idx}-{col_idx}"
                cell_value = str(cell_value)
                
                if with_width is False:
                    if row_idx<len_header:
                        if bold_header is False:
                            table_html += f'<th class="{css_class}" rowspan="{rowspan}" colspan="{colspan}">{cell_value}</th>'
                        else:
                            table_html += f'<th class="{css_class}" rowspan="{rowspan}" colspan="{colspan}"><b>{cell_value}</b></th>'
                    else:
                        table_html += f'<td class="{css_class}" rowspan="{rowspan}" colspan="{colspan}">{cell_value}</td>'
                else:
                    if (col_idx != mol_idx): 
                        if row_idx<len_header:
                            if bold_header is False:
                                table_html += f'<th class="{css_class}" rowspan="{rowspan}" colspan="{colspan}" style="width:{width_percent}%">{cell_value}</th>'
                            else:
                                table_html += f'<th class="{css_class}" rowspan="{rowspan}" colspan="{colspan}" style="width:{width_percent}%"><b>{cell_value}</b></th>'
                        else:
                            table_html += f'<td class="{css_class}" rowspan="{rowspan}" colspan="{colspan}" style="width:{width_percent}%">{cell_value}</td>'
                    else:
                        if row_idx<len_header:
                            if bold_header is False:
                                table_html += f'<th class="{css_class}" rowspan="{rowspan}" colspan="{colspan}" style="width:{width_percent*(random_mol_part+1)}%">{cell_value}</th>'
                            else:
                                table_html += f'<th class="{css_class}" rowspan="{rowspan}" colspan="{colspan}" style="width:{width_percent*(random_mol_part+1)}%"><b>{cell_value}</b></th>'
                        else:
                            table_html += f'<td class="{css_class}" rowspan="{rowspan}" colspan="{colspan}" style="width:{width_percent*(random_mol_part+1)}%">{cell_value}</td>'

            table_html += "</tr>"

            if row_idx==len_header-1:
                table_html += "</thead>"
        
        if flag_tbody:
            table_html += "</tbody>"
        
        table_html += "</table>"

        pad = random.choices([0, random.randint(1, int(font_size*2))], weights=[10,2])[0]

        if plot_mode == 2:
            horizen_pad = random.randint(5, 50)
        else:
            horizen_pad = random.randint(2, 10)
        
        border_table_html = get_border_html(table_html, mode, pad, horizen_pad)
        borderless_table_html = get_borderless_html(table_html, mode, pad, horizen_pad)

        # original_border_path = os.path.join(save_dir, "temp_dir", f"{img_idx}_border_ori.png")
        border_path = os.path.join(save_dir, "temp_dir", f"{img_idx}_border.png")
        borderless_path = os.path.join(save_dir, "temp_dir", f"{img_idx}_borderless.png")


        # print("table2html 2")
        # html_to_image_V2(original_border_table_html, original_border_path)
        html_to_image_V2(border_table_html, border_path)
        html_to_image_V2(borderless_table_html, borderless_path)

        border_image = cv2.imread(border_path)
        border_image = cv2.cvtColor(border_image, cv2.COLOR_BGR2RGB)
        h, w = border_image.shape[:2] 
        border_image_result = transform_fn(image=border_image, keypoints=[[0,0], [w, h]])
        new_height, new_width = border_image_result["image"].shape[:2]
        border_image = Image.fromarray(border_image_result["image"])

        borderless_image = Image.open(borderless_path)
        x_0 = -border_image_result["keypoints"][0][0]
        y_0 = -border_image_result["keypoints"][0][1]
        borderless_image = borderless_image.crop(( x_0,  y_0, x_0 + new_width, y_0 + new_height))
        # border_image = Image.open(border_path)
        # borderless_image = Image.open(borderless_path)
        
        extracted_tables = get_table_from_image(border_image)

        if random.random()<0.85:
            new_image = borderless_image
        else:
            new_image = border_image
        
        if len(extracted_tables) == 1:
            rights = []
            downs = []
            existed_box_dict = {}
            for j, row in enumerate(extracted_tables[0].content.values()): ## j：行号
                for k, cell in enumerate(row): # k: 列号
                    box = (cell.bbox.x1, cell.bbox.y1, cell.bbox.x2, cell.bbox.y2)
                    cell.value = box
                    downs.append(cell.bbox.y1)
                    downs.append(cell.bbox.y2)
                    rights.append(cell.bbox.x1)
                    rights.append(cell.bbox.x2)
                    existed_box_dict[box] = existed_box_dict.get(box, 0) + 1
            
            rights = list(sorted(list(set(rights))))
            downs = list(sorted(list(set(downs))))
            last_existed_box_dict = {k:v for k,v in existed_box_dict.items() if v>=2}

            bboxes = []

            x_min = rights[0]
            x_max = rights[-1]
            local_id = 0
            ## add table
            bboxes.append(
                    {
                        "id": local_id,
                        "bbox": [0, 0, new_image.width, new_image.height],
                        "category_id": 0
                    }
                )
            local_id = local_id + 1
            for i in range(0, len(downs)-1):
                local_id = local_id + 1
                y_min = downs[i]
                y_max = downs[i+1]

                
                if i < len_header:
                    ## row header
                    bboxes.append(
                        {
                            "id": local_id,
                            "bbox": [x_min, y_min, x_max-x_min, y_max-y_min],
                            "category_id": 4
                        }
                    )
                else:
                    ## row
                    bboxes.append(
                        {
                            "id": local_id,
                            "bbox": [x_min, y_min, x_max-x_min, y_max-y_min],
                            "category_id": 2
                        }
                    )
            
            y_min = downs[0]
            y_max = downs[-1]
            for j in range(0, len(rights)-1):
                local_id = local_id + 1
                x_min = rights[j]
                x_max = rights[j+1]
                
                ## columns
                bboxes.append(
                    {
                        "id": local_id,
                        "bbox": [x_min, y_min, x_max-x_min, y_max-y_min],
                        "category_id": 1
                    }
                )

            for box,v in existed_box_dict.items():
                if v>1:
                    local_id = local_id + 1
                    x_min, y_min, x_max, y_max = box
                    ## span
                    bboxes.append(
                        {
                            "id": local_id,
                            "bbox": [x_min, y_min, x_max-x_min, y_max-y_min],
                            "category_id": 5
                        }
                    )
            
            debug = True
            if debug:
                temp_new_image = copy.deepcopy(new_image)
                draw: ImageDraw = ImageDraw.Draw(temp_new_image)
                for bbox in bboxes:
                    x1, y1, w, h = bbox["bbox"]
                    x2, y2 =  x1 + w, y1 + h
                    if bbox["category_id"]==0:
                        draw.rectangle(
                                [(x1, y1), (x2, y2)], outline="black", width=1
                            )
                    elif bbox["category_id"]==1:
                        draw.rectangle(
                                [(x1, y1), (x2, y2)], outline="red", width=1
                            )
                    elif bbox["category_id"]==2:
                        draw.rectangle(
                                [(x1, y1), (x2, y2)], outline="blue", width=1
                            )
                    elif bbox["category_id"]==3:
                        draw.rectangle(
                                [(x1, y1), (x2, y2)], outline="gold", width=3
                            )
                    if bbox["category_id"]==4:
                        draw.rectangle(
                                [(x1, y1), (x2, y2)], outline="gold", width=3
                            )
                        
                temp_new_image.save(os.path.join(save_dir,"vis",f"{img_idx}_0.png"))
                temp_new_image = copy.deepcopy(new_image)
                draw: ImageDraw = ImageDraw.Draw(temp_new_image)
                for bbox in bboxes:
                    x1, y1, w, h = bbox["bbox"]
                    x2, y2 =  x1 + w, y1 + h
                    if bbox["category_id"]==5:
                        draw.rectangle(
                                [(x1, y1), (x2, y2)], outline="cyan", width=1
                            )
                temp_new_image.save(os.path.join(save_dir,"vis",f"{img_idx}_1.png"))
                    
                # temp_new_image.save("temp_teble.png")
                # vis_table(border_image, extracted_tables, "temp_table2.png")
                # border_image.save("temp_table3.png")
                # temp_new_image.save(os.path.join(save_dir,"vis",f"{img_idx}.png"))
            
            save_path = os.path.join(save_dir,"img",f"{img_idx}.png")
            new_image.save(save_path)
            # vis_table(border_image, extracted_tables, os.path.join(save_dir,"vis",f"{img_idx}.png"))
            temp_dict = {
                "id": img_idx,
                "file_name": save_path,
                "bboxes": bboxes,
                "corefs": [],
                "width": new_image.size[0],
                "height": new_image.size[1],
            }
            
            total_result.append(copy.deepcopy(temp_dict))

            os.remove(border_path)
            os.remove(borderless_path)

            import ipdb
            ipdb.set_trace()

            
        if (img_idx+1)%2000 == 0:
            last_result =  {
                "categories": [
                                {"id": 0, "name": "table"}, 
                                {"id": 1, "name": "table column"}, 
                                {"id": 2, "name": "table row"}, 
                                {"id": 3, "name": "table column header"},
                                {"id": 4, "name": "table projected row header"},
                                {"id": 5, "name": "table spanning cell"},
                                {"id": 6, "name": "no object"},
                              ], 
                "licenses": [{"name": "", "id": 0, "url": ""}],
                "images":total_result
            }

            with open(os.path.join(save_dir,"label","total.json"), "w") as f:
                f.write(json.dumps(last_result))
            
            
    
    last_result =  {
        "categories": [{"id": 1, "name": "row"}, {"id": 2, "name": "columns"}, {"id": 3, "name": "span"}], 
        "licenses": [{"name": "", "id": 0, "url": ""}],
        "images":total_result
    }
    
    ## this label https://huggingface.co/microsoft/table-transformer-structure-recognition/blob/main/config.json
    last_result =  {
        "categories": [
                        {"id": 0, "name": "table"}, 
                        {"id": 1, "name": "table column"}, 
                        {"id": 2, "name": "table row"}, 
                        {"id": 3, "name": "table column header"},
                        {"id": 4, "name": "table projected row header"},
                        {"id": 5, "name": "table spanning cell"},
                        {"id": 6, "name": "no object"},
                        ],
        "licenses": [{"name": "", "id": 0, "url": ""}],
        "images":total_result
    }


    with open(os.path.join(save_dir,"label","total.json"), "w") as f:
        f.write(json.dumps(last_result))