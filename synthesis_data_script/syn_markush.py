import os
import sys
main_dir = os.path.dirname(os.path.abspath(__file__))
chemistry_data_dir = os.path.join(main_dir,"chemistry_data")
print("main_dir",main_dir)
sys.path.append(main_dir)

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDepictor
from rdkit.Chem import rdchem
from rdkit.Chem import Descriptors, rdMolDescriptors, QED
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.Draw import rdMolDraw2D

import re
from io import BytesIO, StringIO
import cairosvg
import cv2
from cv2 import imread
from PIL import ImageDraw, Image, ImageFont
from textwrap import wrap

import ipdb
import math
import numpy as np
import pandas as pd
import copy
from tqdm import tqdm
import random

import func_timeout
from func_timeout import func_set_timeout
import multiprocessing as mp
from multiprocessing import Pool
from multiprocessing import get_context

import albumentations as A
from augment import SafeRotate, CropWhite, PadWhite, SaltAndPepperNoise
from indigo import Indigo
from indigo.renderer import IndigoRenderer

cv2.setNumThreads(1)

INDIGO_HYGROGEN_PROB = 0.2
INDIGO_FUNCTIONAL_GROUP_PROB = 0.8
INDIGO_CONDENSED_PROB = 0.5
INDIGO_RGROUP_PROB = 0.5
INDIGO_COMMENT_PROB = 0.3
INDIGO_DEARMOTIZE_PROB = 0.8
INDIGO_COLOR_PROB = 0.2

def normalize_nodes(nodes, flip_y=True):
    x, y = nodes[:, 0], nodes[:, 1]
    minx, maxx = min(x), max(x)
    miny, maxy = min(y), max(y)
    x = (x - minx) / max(maxx - minx, 1e-6)
    if flip_y:
        y = (maxy - y) / max(maxy - miny, 1e-6)
    else:
        y = (y - miny) / max(maxy - miny, 1e-6)
    return np.stack([x, y], axis=1)

def add_comment(indigo):
    if random.random() < INDIGO_COMMENT_PROB:
        indigo.setOption('render-comment', str(random.randint(1, 20)) + random.choice(string.ascii_letters))
        indigo.setOption('render-comment-font-size', random.randint(40, 60))
        indigo.setOption('render-comment-alignment', random.choice([0, 0.5, 1]))
        indigo.setOption('render-comment-position', random.choice(['top', 'bottom']))
        indigo.setOption('render-comment-offset', random.randint(2, 30))

COLORS = {
    u'c': '0.0,0.75,0.75', u'b': '0.0,0.0,1.0', u'g': '0.0,0.5,0.0', u'y': '0.75,0.75,0',
    u'k': '0.0,0.0,0.0', u'r': '1.0,0.0,0.0', u'm': '0.75,0,0.75'
}


def add_color(indigo, mol):
    if random.random() < INDIGO_COLOR_PROB:
        indigo.setOption('render-coloring', True)
    if random.random() < INDIGO_COLOR_PROB:
        indigo.setOption('render-base-color', random.choice(list(COLORS.values())))
    if random.random() < INDIGO_COLOR_PROB:
        if random.random() < 0.5:
            indigo.setOption('render-highlight-color-enabled', True)
            indigo.setOption('render-highlight-color', random.choice(list(COLORS.values())))
        if random.random() < 0.5:
            indigo.setOption('render-highlight-thickness-enabled', True)
        for atom in mol.iterateAtoms():
            if random.random() < 0.1:
                atom.highlight()
    return mol


expected_atomic_nums_and_symbols = [
    (1, 'H'),
    # (2, 'He'),
    (3, 'Li'),
    (4, 'Be'),
    (5, 'B'),
    (6, 'C'),
    (7, 'N'),
    (8, 'O'),
    (9, 'F'),
    # (10, 'Ne'),
    (11, 'Na'),
    (12, 'Mg'),
    (13, 'Al'),
    (14, 'Si'),
    (15, 'P'),
    (16, 'S'),
    (17, 'Cl'),
    # (18, 'Ar'),
    (19, 'K'),
    (20, 'Ca'),
    # (21, 'Sc'),
    # (22, 'Ti'),
    (23, 'V'),
    (24, 'Cr'),
    (25, 'Mn'),
    (26, 'Fe'),
    (27, 'Co'),
    (28, 'Ni'),
    (29, 'Cu'),
    (30, 'Zn'),
    # (31, 'Ga'),
    # (32, 'Ge'),
    (33, 'As'),
    (34, 'Se'),
    (35, 'Br'),
    # (36, 'Kr'),
    # (37, 'Rb'),
    # (38, 'Sr'),
    # (39, 'Y'),
    # (40, 'Zr'),
    # (41, 'Nb'),
    # (42, 'Mo'),
    # (43, 'Tc'),
    # (44, 'Ru'),
    # (45, 'Rh'),
    (46, 'Pd'),
    (47, 'Ag'),
    # (48, 'Cd'),
    # (49, 'In'),
    # (50, 'Sn'),
    # (51, 'Sb'),
    # (52, 'Te'),
    (53, 'I'),
    # (54, 'Xe'),
    # (55, 'Cs'),
    (56, 'Ba'),
    # (57, 'La'),
    # (58, 'Ce'),
    # (59, 'Pr'),
    # (60, 'Nd'),
    # (61, 'Pm'),
    # (62, 'Sm'),
    # (63, 'Eu'),
    # (64, 'Gd'),
    # (65, 'Tb'),
    # (66, 'Dy'),
    # (67, 'Ho'),
    # (68, 'Er'),
    # (69, 'Tm'),
    # (70, 'Yb'),
    # (71, 'Lu'),
    # (72, 'Hf'),
    # (73, 'Ta'),
    # (74, 'W'),
    # (75, 'Re'),
    # (76, 'Os'),
    # (77, 'Ir'),
    (78, 'Pt'),
    (79, 'Au'),
    (80, 'Hg'),
    # (81, 'Tl'),
    (82, 'Pb'),
    # (83, 'Bi'),
    # (84, 'Po'),
    # (85, 'At'),
    # (86, 'Rn'),
    # (87, 'Fr'),
    # (88, 'Ra'),
    # (89, 'Ac'),
    # (90, 'Th'),
    # (91, 'Pa'),
    # (92, 'U'),
    # (93, 'Np'),
    # (94, 'Pu'),
    # (95, 'Am'),
    # (96, 'Cm'),
    # (97, 'Bk'),
    # (98, 'Cf'),
    # (99, 'Es'),
    # (100, 'Fm'),
    # (101, 'Md'),
    # (102, 'No'),
    # (103, 'Lr'),
    # (104, 'Rf'),
    # (105, 'Db'),
    # (106, 'Sg'),
    # (107, 'Bh'),
    # (108, 'Hs'),
    # (109, 'Mt'),
    # (110, 'Ds'),
    # (111, 'Rg'),
    # (112, 'Cn'),
    # (113, 'Nh'),
    # (114, 'Fl'),
    # (115, 'Mc'),
    # (116, 'Lv'),
    # (117, 'Ts'),
    # (118, 'Og')
]

num2symbol = {key:value for (key, value) in expected_atomic_nums_and_symbols}
symbol2num = {value:key for (key, value) in expected_atomic_nums_and_symbols}


total_expected_atomic_nums_and_symbols = [
    (1, 'H'),
    (2, 'He'),
    (3, 'Li'),
    (4, 'Be'),
    (5, 'B'),
    (6, 'C'),
    (7, 'N'),
    (8, 'O'),
    (9, 'F'),
    (10, 'Ne'),
    (11, 'Na'),
    (12, 'Mg'),
    (13, 'Al'),
    (14, 'Si'),
    (15, 'P'),
    (16, 'S'),
    (17, 'Cl'),
    (18, 'Ar'),
    (19, 'K'),
    (20, 'Ca'),
    (21, 'Sc'),
    (22, 'Ti'),
    (23, 'V'),
    (24, 'Cr'),
    (25, 'Mn'),
    (26, 'Fe'),
    (27, 'Co'),
    (28, 'Ni'),
    (29, 'Cu'),
    (30, 'Zn'),
    (31, 'Ga'),
    (32, 'Ge'),
    (33, 'As'),
    (34, 'Se'),
    (35, 'Br'),
    (36, 'Kr'),
    (37, 'Rb'),
    (38, 'Sr'),
    (39, 'Y'),
    (40, 'Zr'),
    (41, 'Nb'),
    (42, 'Mo'),
    (43, 'Tc'),
    (44, 'Ru'),
    (45, 'Rh'),
    (46, 'Pd'),
    (47, 'Ag'),
    (48, 'Cd'),
    (49, 'In'),
    (50, 'Sn'),
    (51, 'Sb'),
    (52, 'Te'),
    (53, 'I'),
    (54, 'Xe'),
    (55, 'Cs'),
    (56, 'Ba'),
    (57, 'La'),
    (58, 'Ce'),
    (59, 'Pr'),
    (60, 'Nd'),
    (61, 'Pm'),
    (62, 'Sm'),
    (63, 'Eu'),
    (64, 'Gd'),
    (65, 'Tb'),
    (66, 'Dy'),
    (67, 'Ho'),
    (68, 'Er'),
    (69, 'Tm'),
    (70, 'Yb'),
    (71, 'Lu'),
    (72, 'Hf'),
    (73, 'Ta'),
    (74, 'W'),
    (75, 'Re'),
    (76, 'Os'),
    (77, 'Ir'),
    (78, 'Pt'),
    (79, 'Au'),
    (80, 'Hg'),
    (81, 'Tl'),
    (82, 'Pb'),
    (83, 'Bi'),
    (84, 'Po'),
    (85, 'At'),
    (86, 'Rn'),
    (87, 'Fr'),
    (88, 'Ra'),
    (89, 'Ac'),
    (90, 'Th'),
    (91, 'Pa'),
    (92, 'U'),
    (93, 'Np'),
    (94, 'Pu'),
    (95, 'Am'),
    (96, 'Cm'),
    (97, 'Bk'),
    (98, 'Cf'),
    (99, 'Es'),
    (100, 'Fm'),
    (101, 'Md'),
    (102, 'No'),
    (103, 'Lr'),
    (104, 'Rf'),
    (105, 'Db'),
    (106, 'Sg'),
    (107, 'Bh'),
    (108, 'Hs'),
    (109, 'Mt'),
    (110, 'Ds'),
    (111, 'Rg'),
    (112, 'Cn'),
    (113, 'Nh'),
    (114, 'Fl'),
    (115, 'Mc'),
    (116, 'Lv'),
    (117, 'Ts'),
    (118, 'Og')
]

total_num2symbol = {key:value for (key, value) in total_expected_atomic_nums_and_symbols}
total_symbol2num = {value:key for (key, value) in total_expected_atomic_nums_and_symbols}


##————————————————官能团—————————————————————
from rdkit.Chem import FragmentCatalog
fName = "FunctionalGroups.txt"
fName = os.path.join(chemistry_data_dir, fName)
# 根据官能团库实例化一个参数器
fparams = FragmentCatalog.FragCatParams(1, 6, fName)
# 查看官能团库中包含的官能团数量
fparams_num = fparams.GetNumFuncGroups()
smiles_list = []
for i in range(fparams_num):
    smiles_list.append(Chem.MolToSmiles(fparams.GetFuncGroup(i)))
small_fragment = [
    "*C",
    "*Cl",
    "*Br",
    "*I",
    "*F",
    "*CC",
    "*CO",
    "*N",
    "*S",
    "*CCBr",
    "*C(C)Br",
    "*C(C)C",
    "*N",
    "*C=C",
    "*C#C",
    "*C(=O)O",
    "*C(=O)N",
    "*C#N",
    "*[N+]#[C-]",
    # "*CC(=O)O",
    "*SC",
    # "*OC(=O)CC",
    "*OC(=O)",
    # "*NC(=O)C",
    "*NC(=O)C",
    # "*[N+][O-](=O)",
    # "*NS(=O)(=O)C",
    # "*NS(=O)(=O)",
    # "*S(=O)(=O)N",
]
small_fragment = list(set(small_fragment+smiles_list))
del small_fragment[small_fragment.index("*#N")]
del small_fragment[small_fragment.index("*N=NC")]
del small_fragment[small_fragment.index("*N(O)=O")]
del small_fragment[small_fragment.index("*N#N")]
del small_fragment[small_fragment.index("*=S")]
del small_fragment[small_fragment.index("*=NC")]
del small_fragment[small_fragment.index("*=NO")]
del small_fragment[small_fragment.index("*C(=O)OC")]
del small_fragment[small_fragment.index("*S(=O)(=O)OC")]
del small_fragment[small_fragment.index("*=O")]
del small_fragment[small_fragment.index("*NS(C)(=O)=O")]
del small_fragment[small_fragment.index("*S(C)(=O)=O")]
# del small_fragment[small_fragment.index("*C=C*C#C")]
del small_fragment[small_fragment.index("*=N")]
new_smiles_list = []
for smiles in small_fragment:
    try:
        if len(Chem.MolFromSmiles(smiles).GetAtoms())<=2 and smiles[1].isalpha():
            new_smiles_list.append(smiles)
    except Exception as e:
        print(e)
small_fragment = new_smiles_list


#----------------------cropwhite操作的函数-----------------------
def get_transforms_2():
    trans_list = []
    trans_list.append(CropWhite(pad=0))
    return A.Compose(trans_list, keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
fn2 = get_transforms_2()


#-----------------------get config of cdk----------------------
def get_cdk_plugin():
    from scyjava import to_python as j2p
    from scyjava import config, jimport
    config.add_endpoints('org.openscience.cdk:cdk-bundle:2.8')
    from scyjava import jimport
    IChemObjectBuilder = jimport('org.openscience.cdk.interfaces.IChemObjectBuilder')
    SilentChemObjectBuilder = jimport('org.openscience.cdk.silent.SilentChemObjectBuilder')
    SmilesParser = jimport('org.openscience.cdk.smiles.SmilesParser')
    Color = jimport('java.awt.Color')
    DepictionGenerator = jimport('org.openscience.cdk.depict.DepictionGenerator')
    StringWriter = jimport('java.io.StringWriter')
    MDLV3000Writer = jimport('org.openscience.cdk.io.MDLV3000Writer')
    StructureDiagramGenerator = jimport('org.openscience.cdk.layout.StructureDiagramGenerator')

    bldr = SilentChemObjectBuilder.getInstance()
    smipar = SmilesParser(bldr)
    sdg = StructureDiagramGenerator()

    cdk_plugin = {}
    cdk_plugin["smipar"] = smipar
    cdk_plugin["sdg"] = sdg
    # cdk_plugin["writer"] = writer
    cdk_plugin["StringWriter"] = StringWriter ## 每次写mol文件的时候重新写
    cdk_plugin["j2p"] = j2p
    cdk_plugin["MDLV3000Writer"] = MDLV3000Writer
    return cdk_plugin

#---------交叉点计算----------
def find_intersection(points):
    """计算交点

    Args:
        points (_type_): _description_

    Returns:
        _type_: _description_
    """

    x1, y1 = points[0]
    x2, y2 = points[1]
    x3, y3 = points[2]
    x4, y4 = points[3]

    # 计算直线1的参数
    a1 = y2 - y1
    b1 = x1 - x2
    c1 = x2 * y1 - x1 * y2

    # 计算直线2的参数
    a2 = y4 - y3
    b2 = x3 - x4
    c2 = x4 * y3 - x3 * y4

    # 计算交点的x坐标
    x = (b1 * c2 - b2 * c1) / (a1 * b2 - a2 * b1)

    # 计算交点的y坐标
    y = (c1 * a2 - c2 * a1) / (a1 * b2 - a2 * b1)

    if ((x-x1)**2+(y-y1)**2 + (x-x2)**2+(y-y2)**2) > ((x2-x1)**2+(y2-y1)**2):
        return [None, None]
    elif ((x-x3)**2+(y-y3)**2 + (x-x4)**2+(y-y4)**2) > ((x4-x3)**2+(y4-y3)**2):
        return [None, None]
    else:
        return [x, y]

#------------保持最大骨架-------------
def keep_scaffold_fn(mol):
    smiles = Chem.MolToSmiles(mol)
    # 提取化合物的骨架
    scaffold = MurckoScaffold.MurckoScaffoldSmilesFromSmiles(smiles)

    # 创建骨架的 RDKit 分子对象
    scaffold_mol = Chem.MolFromSmiles(scaffold)

    mol = Chem.MolFromSmiles(Chem.MolToSmiles(scaffold_mol))
    return mol

def get_mol_block_v3000(smiles:str, cdk_plugin:dict):
    """从cdk中获取molblock(V3000)

    Args:
        smiles (str): 
        cdk_plugin (dict): cdk的配置

    Returns:
        molfile_v3000 (str): molblock(V3000)
    """
    ## 初始化
    smipar = cdk_plugin["smipar"]
    sdg = cdk_plugin["sdg"]
    StringWriter = cdk_plugin["StringWriter"]
    writer = StringWriter() ## 每次写的时候需要初始化，不能一直用一个, 类似一个sdf
    j2p = cdk_plugin["j2p"]
    MDLV3000Writer = cdk_plugin["MDLV3000Writer"]

    mol = smipar.parseSmiles(smiles)
    
    # sdg.setMolecule(mol)
    sdg.generateCoordinates(mol)

    # 创建 MDLV3000Writer 对象
    mdl_writer = MDLV3000Writer(writer)

    # 将分子对象存储为 Molfile V3000 格式
    mdl_writer.write(mol)

    # 关闭 MDLV3000Writer 对象
    mdl_writer.close()

    # 获取 Molfile V3000 格式的字符串表示
    molfile_v3000 = j2p(writer.toString())

    return molfile_v3000

#---------------------keep largest smiles--------------
def keep_largest_component_fn(smiles:str):
    """保持最长的分子的smiles, 按照'.'进行split

    Args:
        smiles (str): smiles

    Returns:
        new_smiles (str): 新的smiles
    """
    smiles_list = smiles.split(".")
    if len(smiles_list) == 1:
        return smiles
    else:
        max_len_subsmiles = 0
        ## 用字典记录结果
        result = {}

        ## 获取最长的smiles片段，并记录在在result中result[len_subsmiles]=[subsmiles]
        for subsmiles in smiles_list:
            ## 这里简单地对sub_smiles的长度进行判读即可
            length = len(subsmiles)
            if length not in result:
                result[length] = [subsmiles]
            else:
                result[length].append(subsmiles)
            
            if length > max_len_subsmiles:
                max_len_subsmiles = length

        ## 先简单的保留最长的smiles吧
        new_smiles = (".").join(result[max_len_subsmiles][:1])#只保留第一个

        return new_smiles

## 缩写表
import string
abbre_dict = {}
with open(os.path.join(chemistry_data_dir, "abbretion.txt"), "r") as f:
    for line in f.readlines():
        line = line.rstrip("\n")
        line_list = line.split()
        if len(line_list) == 2:
            key,value = line_list
            if len(key)<=5:
                abbre_dict[key] = value

def abbretion_replace(mol):
    """对分子中的原子进行替换
        替换的规则为：随机选择末端原子进行替换

    Args:
        mol (rdkit.Chem.rdchem.Mol): 分子

    Returns:
        mol (rdkit.Chem.rdchem.Mol): 替换后的分子
    """

    ## 寻找末端的原子
    ## keep validality of molecule
    end_atoms_idx = get_end_atoms_idx(mol, only_single_bond=True, zero_charge=True)

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
        if random.random()<0.3:
            symbol = random.sample(abbre_dict.keys(), 1)[0]
        else:
            random_symbol = "".join([random.choice(string.ascii_letters) for _ in range(random.randint(1, 3))])
            
            if random.random()<0.3:
                symbol = random_symbol + "".join(["\'" for _ in range(random.randint(1, 2))]) + str(int(random.randint(1,1000)))
            # elif random.random()<1:
            #     symbol = random_symbol + str(int(random.randint(1,1000))) + "".join(["\'" for _ in range(random.randint(1, 3))])
            else:
                symbol = random_symbol + str(int(random.randint(1,1000)))
            

        atom = mol.GetAtomWithIdx(atom_idx)
        Chem.SetAtomAlias(atom, symbol)
        ## 将替换的原子变为*
        ## Chem.GetPeriodicTable().GetAtomicNumber(dummy_symbol) = 0
        atom.SetNumExplicitHs(0)
        atom.SetAtomicNum(0)
        atom.SetFormalCharge(0)
    return mol

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

def extract_element_symbols(text):
    try:
        pattern = r'\[([A-Za-z]+)\d*[\+\-]?\]'
        matches = re.findall(pattern, text)
        return matches[0]
    except:
        return text


def get_result_smiles(symbols, smiles):
    result = ""
    first_idx = 0
    second_idx = 0

    symbols = [extract_element_symbols(_) for _ in symbols]

    while first_idx<len(symbols) and second_idx<len(smiles):

        if len(symbols[first_idx]) == 1:
            Flag = False
            while second_idx<len(smiles) and (Flag is False):
                if smiles[second_idx]=="*" and symbols[first_idx]=="*":
                    Flag = True
                elif symbols[first_idx] == smiles[second_idx].upper():
                    Flag = True
                result += smiles[second_idx]
                second_idx = second_idx + 1
                if (Flag is True):
                    break

        else: #len(symbols[first_idx]) > 1
            Flag = False
            while second_idx<len(smiles) and Flag==False:
                if smiles[second_idx]=="*":
                    if (symbols[first_idx] not in total_symbol2num.keys()):
                        result += "[" + symbols[first_idx] + "]"
                    else:
                        result += symbols[first_idx]
                    Flag = True
                    second_idx = second_idx + 1
                elif smiles[second_idx: second_idx+len(symbols[first_idx])] == symbols[first_idx]:
                    result += symbols[first_idx]
                    second_idx = second_idx + len(symbols[first_idx])
                    Flag = True
                else:
                    result += smiles[second_idx]
                    second_idx = second_idx + 1
                
                if (Flag is True):
                    break
        
        first_idx = first_idx + 1

    while second_idx<len(smiles):
        result += smiles[second_idx]
        second_idx = second_idx + 1
    
    success = False
    if (len(smiles)==second_idx) and (len(symbols)==first_idx):
        success = True

    return result, success


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


def draw_molecule(mol, 
                size=768, 
                is_black=True, 
                is_wave_line=False, 
                zero_padding=False, 
                with_coord=True,
                highlight=False,):
    """_summary_

    Args:
        mol (rdkit.Chem.rdchem.Mol): 分子
        size (int, optional): 图片的尺寸. Defaults to 384.
        is_black (bool, optional): 是否使用rdkit的黑白风格绘图，默认为True. Defaults to True.
        is_wave_line (bool, optional): 是否对R_group使用波浪线的绘图风格. Defaults to False.
        zero_padding (bool, optional): _description_. Defaults to False.
        with_coord (bool, optional): _description_. Defaults to True.
        with_wedge (bool, optional): 是否提前计算好了手性的键. Defaults to True.

    Returns:
        image (PIL.Image.image) : 图像
        coords_list (List): 坐标 
    """

    ## 从smiles而来的分子
    if (with_coord is False):
        rdDepictor.Compute2DCoords(mol, clearConfs=True) #clearConfs=True会改变构像
        ps = Chem.BondWedgingParameters()
        ps.wedgeTwoBondsIfPossible = True
        Chem.WedgeMolBonds(mol, mol.GetConformer(), ps)
        d = rdMolDraw2D.MolDraw2DSVG(size,size)
        dopts = rdMolDraw2D.MolDrawOptions()
        if zero_padding:
            dopts.padding = 0
        if is_black:
            dopts.useBWAtomPalette()

        # 绘制波浪线的选项
        if is_wave_line:
            dopts.dummiesAreAttachments = True

        # d = Draw.MolDraw2DCairo(-1,-1)
        for atom in mol.GetAtoms():
            atom_idx = atom.GetIdx()
            atom_alias = Chem.GetAtomAlias(atom)
            if atom_alias!="":
                ## 会报错
                ## 修正了上下标
                ## 第0个元素不是数字
                if atom_alias[0].isdigit() is False:
                    convert_atom_alias = ""
                    i = 0
                    while i < len(atom_alias):
                        ## 新增跳过0619
                        if atom_alias[i].isdigit():
                            j = i 
                            while j < len(atom_alias):
                                if atom_alias[j].isdigit():
                                    j = j + 1
                                    continue
                                else:
                                    break
                            convert_atom_alias += "<sub>"+atom_alias[i:j]+"</sub>"
                            i = j
                        else:
                            convert_atom_alias += atom_alias[i]
                            i = i + 1
                    dopts.atomLabels[atom_idx] = convert_atom_alias
                else:
                    dopts.atomLabels[atom_idx] = atom_alias

        d.SetDrawOptions(dopts)
        rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
        d.FinishDrawing()
        drawing = d.GetDrawingText()

    # 从molblock而来的分子
    else:
        d = rdMolDraw2D.MolDraw2DSVG(size, size)
        ## 关闭高亮模块
        dopts = d.drawOptions()
        ## 高亮模块的设置
        if highlight is False:
            ## 关闭高亮模块
            dopts.variableAtomRadius = 0
            dopts.variableBondWidthMultiplier = 0
        else:
            dopts.variableAtomRadius = random.choice([0.2, 0.4, 0.6, 0.8])
            dopts.variableBondWidthMultiplier = random.choice([20, 40, 60, 80])//4
            dopts.setVariableAttachmentColour = random.choice([(1.0, 0.5, 0.5), (0.75, 0.75, 0.75)])

        dopts.annotationFontScale = 0.8
        dopts.useMolBlockWedging = True
        dopts.bondLineWidth = random.randint(2, 5)*0.5
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
        
        ## 原子序号
        # if random.random()>0.5:
        #     dopts.addAtomIndices = True
            
        if zero_padding:
            dopts.padding = 0
        ps = Chem.BondWedgingParameters()
        ps.wedgeTwoBondsIfPossible = True
        Chem.WedgeMolBonds(mol, mol.GetConformer(), ps)
        if is_black:
            dopts.useBWAtomPalette()
        # 绘制波浪线的选项
        if is_wave_line:
            dopts.dummiesAreAttachments = True
        # d = Draw.MolDraw2DCairo(-1,-1)
        for atom in mol.GetAtoms():
            atom_idx = atom.GetIdx()
            atom_alias = Chem.GetAtomAlias(atom)
            if atom_alias!="":
                if atom_alias[0].isdigit() is False:
                    convert_atom_alias = ""
                    i = 0
                    while i < len(atom_alias):
                        ## 新增跳过0619
                        if atom_alias[i].isdigit():
                            j = i 
                            while j < len(atom_alias):
                                if atom_alias[j].isdigit():
                                    j = j + 1
                                    continue
                                else:
                                    break
                            convert_atom_alias += "<sub>"+atom_alias[i:j]+"</sub>"
                            i = j
                        else:
                            convert_atom_alias += atom_alias[i]
                            i = i + 1
                    dopts.atomLabels[atom_idx] = convert_atom_alias
                else:
                    dopts.atomLabels[atom_idx] = atom_alias

        if with_coord is False:
            rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
        else:
            d.DrawMolecule(mol)
        d.FinishDrawing()
        drawing = d.GetDrawingText()
    
    image = svg2img(drawing)

    # atoms = mol.GetAtoms()
    coords_list = []
    for iatom in range(mol.GetNumAtoms()):
        p = d.GetDrawCoords(iatom)
        coords_list.append((p.x, p.y))
    
    return image, coords_list

def convert_bond_type_2_int(bond):
    if bond.GetBondType() == rdkit.Chem.rdchem.BondType.SINGLE:
        return 1
    elif bond.GetBondType() == rdkit.Chem.rdchem.BondType.DOUBLE:
        return 2
    elif bond.GetBondType() == rdkit.Chem.rdchem.BondType.TRIPLE:
        return 3
    elif bond.GetBondType() == rdkit.Chem.rdchem.BondType.AROMATIC:
        return 4

def get_graph(mol, coords_list, shuffle_nodes=False):
    """生成数据

    Args:
        mol (rdkit.Chem.rdchem.Mol): 分子
        coords_list (List): 坐标
        shuffle_nodes (bool, optional): 使用smiles作为标签时，推荐关闭. Defaults to False.

    Returns:
        graph (dict): 包含 
            graph['coords'] (List): 坐标
            graph['symbols'] (List) : 原子符号
            graph['symbols_rdkit'] (List): rdkit的原子符号), 
            graph['edges'] (List): 边,
            graph['molblock'] (List): molblock,
            graph['rdkit_smiles'] (str): molblock,
    """
    chiral_atoms = []
    if "@" in Chem.MolToSmiles(mol,canonical=False):
        mol.UpdatePropertyCache(strict=False)
        chiral_centers = Chem.FindMolChiralCenters(mol)
        if len(chiral_centers)>0:
            chiral_atoms = [atom_idx for (atom_idx, _) in chiral_centers]

    coords, symbols, symbols_rdkit = [], [], []
    index_map = {}
    atoms = [atom for atom in mol.GetAtoms()]
    if shuffle_nodes:
        random.shuffle(atoms)
    for i, atom in enumerate(atoms):
        atom_symbol = atom.GetSymbol()
        atom_charge = atom.GetFormalCharge()
        if atom_charge == 0:
            pass
        ## MolScribe 有点搞，生成阶段不带电荷，预测阶段却带了电荷
        elif (atom_charge == 1):
            atom_symbol = f"[{atom_symbol}+]"
        elif (atom_charge == -1):
            atom_symbol = f"[{atom_symbol}-]"
        elif atom_charge > 1:
            atom_symbol = f"[{atom_symbol}{abs(atom_charge)}+]"
        elif atom_charge < -1:
            atom_symbol = f"[{atom_symbol}{abs(atom_charge)}-]"
        
        atom_symbol_alais = Chem.GetAtomAlias(atom) if Chem.GetAtomAlias(atom)!="" else atom_symbol
        symbols.append(atom_symbol_alais)
        symbols_rdkit.append(atom_symbol)
        index_map[atom.GetIdx()] = i
        coords.append(list(coords_list[atom.GetIdx()]))
    
    n = len(symbols)
    edges = np.zeros((n, n), dtype=int)
    for bond in mol.GetBonds():
        s = index_map[bond.GetBeginAtomIdx()]
        t = index_map[bond.GetEndAtomIdx()]
        # 1/2/3/4 : single/double/triple/aromatic
        edges[s, t] = convert_bond_type_2_int(bond)
        edges[t, s] = convert_bond_type_2_int(bond)
        if (len(chiral_atoms)>0) and (bond.GetBondDir() in [rdkit.Chem.rdchem.BondDir.BEGINDASH, rdkit.Chem.rdchem.BondDir.BEGINWEDGE]):
            ## 确保从手性中心指向其他原子
            ## 同时确保必有一个原子是手心中心
            if (bond.GetEndAtomIdx() in chiral_atoms) and (bond.GetBeginAtomIdx() not in chiral_atoms):
                s, t = t, s
            elif (bond.GetBeginAtomIdx() in chiral_atoms) and (bond.GetEndAtomIdx() not in chiral_atoms):
                pass
            else:
                print("error in stero in rdkit")
                return {'num_atoms': -1}

            if bond.GetBondDir() == rdkit.Chem.rdchem.BondDir.BEGINDASH:
                edges[s, t] = 6
                edges[t, s] = 5
            else:
                edges[s, t] = 5
                edges[t, s] = 6
    
    ## 新增加了
    rdkit_smiles = ""
    try:
        rdkit_smiles = Chem.MolToSmiles(mol, canonical=False)
    except:
        pass

    new_smiles, success = get_result_smiles(symbols, rdkit_smiles)

    graph = {
        "smiles":new_smiles,
        'coords': coords,
        'symbols': symbols,
        "symbols_rdkit":symbols_rdkit,
        'edges': edges,
        'num_atoms': len(symbols),
        "rdkit_smiles": rdkit_smiles,
    }
    return graph



def indigo_get_graph(rdkit_mol, 
                    shuffle_nodes:bool=False, 
                    pseudo_coords:bool=False, 
                    debug:bool=False,
                    default_option=False):
    """使用indigo处理数据

    Args:
        mol_file (path): mol文件保存路劲
        shuffle_nodes (bool, optional): 是否需要打乱原子.以字符串方式进行生成时，不推荐打乱. Defaults to False.
        pseudo_coords (bool, optional): 暂时没有用到. Defaults to False.
        debug (bool, optional): 是否启用debug. Defaults to False.


    Returns:
        graph (dict): 包含 
            graph['coords'] (List): 坐标
            graph['symbols'] (List) : 原子符号
            graph['symbols_rdkit'] (List): rdkit的原子符号), 
            graph['edges'] (List): 边,
            graph['molblock'] (List): molblock,
            graph['rdkit_smiles'] (str): molblock,
    """
    
    # Create an Indigo object
    indigo_obj = Indigo()
    ## the option of render
    renderer = IndigoRenderer(indigo_obj)
    indigo_obj.setOption('render-output-format', 'png')
    indigo_obj.setOption('render-background-color', '1,1,1')
    indigo_obj.setOption('render-stereo-style', 'none')
    indigo_obj.setOption('render-label-mode', 'hetero')
    indigo_obj.setOption('render-font-family', 'Arial')
    if not default_option:
        thickness = random.uniform(0.5, 2)  # limit the sum of the following two parameters to be smaller than 4
        indigo_obj.setOption('render-relative-thickness', thickness)
        indigo_obj.setOption('render-bond-line-width', random.uniform(1, 2.5 - thickness))
        if random.random() < 0.5:
            indigo_obj.setOption('render-font-family', random.choice(['Arial', 'Times', 'Courier', 'Helvetica']))
        # indigo_obj.setOption('render-label-mode', random.choice(['hetero', 'terminal-hetero']))
        # indigo_obj.setOption('render-implicit-hydrogens-visible', random.choice([True, False]))
        # if random.random() < 0.1:
        #     indigo_obj.setOption('render-stereo-style', 'old')
        ## 序号
        # if random.random() < 0.2:
        #     indigo_obj.setOption('render-atom-ids-visible', True)
    
    if random.random() < 0.5:
        indigo_obj.setOption('render-font-family', random.choice(['Arial', 'Times', 'Courier', 'Helvetica']))
    # indigo_obj.setOption('render-label-mode', random.choice(['hetero', 'terminal-hetero']))
    indigo_obj.setOption('render-implicit-hydrogens-visible', random.choice([True, False]))
    # if random.random() < 0.1:
    #     indigo_obj.setOption('render-stereo-style', 'old')

    mol_block = Chem.MolToMolBlock(rdkit_mol).replace("CONNECT=HT ", "").replace("CONNECT=EU ","")
    # 将Mol文件字符串写入内存中
    mol_file = StringIO(mol_block)
    # 从内存中读取Mol文件
    mol = indigo_obj.loadMolecule(mol_file.getvalue())

    ## 芳香化
    if random.random() < 0.75:
        mol.aromatize()
    add_comment(indigo_obj)
    mol = add_color(indigo_obj, mol)

    # mol, smiles = generate_output_smiles(indigo_obj, mol)

    ## 绘图
    buf = renderer.renderToBuffer(mol) #加上了坐标
    ## 获取图像
    image = cv2.imdecode(np.asarray(bytearray(buf), dtype=np.uint8), 1)  # decode buffer to image
    
    ## 获取手信原子，并记录在列表中
    chiral_atoms = [atom.index() for atom in mol.iterateStereocenters()]

    ## 强行读取分子成为rdkit_mol
    rdkit_mol = Chem.MolFromMolBlock(mol.molfile(), sanitize=False, removeHs=False, strictParsing=False)
    coords, symbols, symbols_rdkit = [], [], []
    index_map = {}
    atoms = [atom for atom in mol.iterateAtoms()]
    if shuffle_nodes:
        random.shuffle(atoms)
    replace_list = []
    for i, atom in enumerate(atoms):
        if pseudo_coords:
            x, y, z = atom.xyz()
        else:
            x, y = atom.coords()
        coords.append([x, y])
        atom_symbol = atom.symbol()
        atom_charge = atom.charge()

        ## 如果不在symbol2num的字典中，则使得rdkit_atom_symbol变为*
        if atom_symbol not in total_symbol2num.keys():
            # print(atom_symbol)
            rdkit_atom = rdkit_mol.GetAtomWithIdx(i)
            Chem.SetAtomAlias(rdkit_atom, atom_symbol)
            rdkit_atom.SetNumExplicitHs(0)
            rdkit_atom.SetAtomicNum(0) #Chem.GetPeriodicTable().GetAtomicNumber(dummy_symbol)
            rdkit_atom.SetFormalCharge(0)
            rdkit_atom_symbol = "*"
            replace_list.append(i)
        else:
            rdkit_atom_symbol = atom_symbol

        ## 记录带电荷的的属性，也可以不记录
        if atom_charge == 0:
            pass
        elif (atom_charge == 1):
            atom_symbol = f"[{atom_symbol}+]"
            rdkit_atom_symbol = f"[{rdkit_atom_symbol}+]"
        elif (atom_charge == -1):
            atom_symbol = f"[{atom_symbol}-]"
            rdkit_atom_symbol = f"[{rdkit_atom_symbol}-]"
        elif atom_charge > 1:
            atom_symbol = f"[{atom_symbol}{abs(atom_charge)}+]"
            rdkit_atom_symbol = f"[{rdkit_atom_symbol}{abs(atom_charge)}+]"
        elif atom_charge < -1:
            atom_symbol = f"[{atom_symbol}{abs(atom_charge)}-]"
            rdkit_atom_symbol = f"[{rdkit_atom_symbol}{abs(atom_charge)}-]"

        symbols.append(atom_symbol)
        symbols_rdkit.append(rdkit_atom_symbol)
        index_map[atom.index()] = i
    
    if pseudo_coords: #default: False
        coords = normalize_nodes(np.array(coords))
        h, w, _ = image.shape
        coords[:, 0] = coords[:, 0] * w
        coords[:, 1] = coords[:, 1] * h
    
    n = len(symbols)
    edges = np.zeros((n, n), dtype=int)
    for bond in mol.iterateBonds():
        s = index_map[bond.source().index()]
        t = index_map[bond.destination().index()]
        # 1/2/3/4 : single/double/triple/aromatic
        edges[s, t] = bond.bondOrder()
        edges[t, s] = bond.bondOrder()
        if bond.bondStereo() in [5, 6]:
            ## 确保是从src→target
            if (bond.destination().index() in chiral_atoms) and (bond.source().index() not in chiral_atoms):
                s, t = t, s
            elif (bond.destination().index() not in chiral_atoms) and (bond.source().index() in chiral_atoms):
                pass
            else:
                print("error in stero in indigo")
                ## 异常
                return image, {'num_atoms': -1}

            edges[s, t] = bond.bondStereo()
            edges[t, s] = 11 - bond.bondStereo()
    
    ## 结果
    rdkit_smiles = ""
    try:
        rdkit_smiles = Chem.MolToSmiles(rdkit_mol, canonical=False)
    except:
        pass

    ## TODO: indigo的smiles没有写进来
    graph = {
        'coords': coords,
        'symbols': symbols,
        "symbols_rdkit":symbols_rdkit,
        'edges': edges,
        'num_atoms': len(symbols),
        "molblock":Chem.MolToMolBlock(rdkit_mol, forceV3000=True),
        "rdkit_smiles":rdkit_smiles,
    }
    image = Image.fromarray(image)
    return image, graph


def get_chirality_h_in_ring(mol):
    """
    Args:
        mol ()
    """
    chiral_centers = Chem.FindMolChiralCenters(mol)

    ring = mol.GetRingInfo() # rdkit.Chem.rdchem.RingInf
    atominfos = ring.AtomRings() # tuple(tuple())
    ring_atom_list = []
    for atom_list in atominfos:
        ring_atom_list.extend(atom_list)

    ring_chiral_centers = [] ## 记录环上手性H的的数量
    for (atom_idx, _) in chiral_centers:
        if atom_idx in ring_atom_list:
            ring_chiral_centers.append(atom_idx)
    
    return ring_chiral_centers

def add_chirality_h_in_ring(mol):
    ring_chiral_centers = get_chirality_h_in_ring(mol)
    
    if len(ring_chiral_centers)>0:
        res = Chem.AddHs(mol, addCoords=False)
        rdDepictor.Compute2DCoords(res, clearConfs=True) #clearConfs=True会改变构像
        ps = Chem.BondWedgingParameters()
        ps.wedgeTwoBondsIfPossible = True
        Chem.WedgeMolBonds(res, res.GetConformer(), ps)

        res = Chem.RWMol(res)
        res.BeginBatchEdit()
        for atom in res.GetAtoms():
            if atom.GetAtomicNum() == 1 and atom.GetDegree() == 1:
                neighbor = atom.GetNeighbors()[0]
                if neighbor.GetChiralTag() in (Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW):
                    # print(f"neighbor {neighbor.GetIdx()} is chiral.")
                    # print(f"Atom {atom.GetIdx()} is chiral.\n")
                    pass
                else:
                    res.RemoveAtom(atom.GetIdx())

        res.CommitBatchEdit()
        res = res.GetMol()
        return res
    
    else:
        return mol


dummy_symbol = "*"
def RGroup_2(mol:Chem.rdchem.Mol,
            cdk_plugin:dict, 
            is_dash:bool=False, 
            is_wave_line:bool=False, 
            dash_line:bool=False):
            
    """
    Args:
        mol (Chem.rdchem.Mol): molecule.
        cdk_plugin (dict): the plugin of cdk in get MolFile with V3000 Format. Don't use in this situation.
        is_dash (bool) : insertaction of dash line for r-group
        is_wave_line (bool): insertaction of wave line for r-group
        dash_line (bool): dansh bond for rgroup

    Return:
        image (PIL.Image.Image): image of molecule
        graph (dict): dict of molecule, containing `coords`, `smiles`, `extention`, `molblock`, `edges` and so on
    """
    ## get index of terminal atom
    ## ensuring the charge is 0, which is convenient for handling
    end_atoms_idx = get_end_atoms_idx(mol, only_single_bond=True, zero_charge=True)

    if len(end_atoms_idx) < 1:
        print("no terminal atom in molecule")
        return None, None

    ## <2 terminal atom
    elif len(end_atoms_idx) <= 2:
        extention = end_atoms_idx
    
    else:
        ## random select one atom
        num_elements = random.randint(1, len(end_atoms_idx)-1)
        extention = random.sample(end_atoms_idx, num_elements)
        extention.sort()
    
    ## replace with C element
    for atom_index in extention:
        atom = mol.GetAtomWithIdx(atom_index)
        atom.SetAtomicNum(6)

    image, coords_list = draw_molecule(copy.deepcopy(mol), with_coord=False, is_wave_line=False)

    # random select mode
    if (is_wave_line is False) and (is_dash is False) and (dash_line is False):
        choice = random.choice([1,2,3])
        if choice == 1:
            is_dash = True
        elif choice == 2:
            is_wave_line = True
        else:
            dash_line = True
    
        print("choice", choice)

    length_ratio = random.choice([3, 4, 5, 6, 7, 8])
    
    ## enumerate index of atom in extention
    for atom_index in extention:
        ## get atom
        atom = mol.GetAtomWithIdx(atom_index)
        ## get neightbor atoms (only one)
        neighbor_atom = atom.GetNeighbors()[0]
        ## get index to neighbor atom
        neighbor_atom_idx = neighbor_atom.GetIdx()

        ## get coordinates of atom
        atom_coord = coords_list[atom_index]
        ## get coordinates of neighbor atom
        neighbor_atom_coord = coords_list[neighbor_atom_idx]

        ## 从1/2处进行shift
        ratio = random.choice([1, 5/8, 2/3, 3/4, 5/6, 1/2]) ## 确保这里的数值大于1/2

        ## 获得相交坐标
        middle_coord = [(atom_coord[0]*ratio + neighbor_atom_coord[0]*(1-ratio)), 
                        (atom_coord[1]*ratio + neighbor_atom_coord[1]*(1-ratio))]
        

        ## 获取键长
        bond_length = math.sqrt((neighbor_atom_coord[0]-atom_coord[0])**2 + (neighbor_atom_coord[1]-atom_coord[1])**2)

        ## 获取角度（弧度）
        angle = math.atan2(neighbor_atom_coord[0] - atom_coord[0], 
                            atom_coord[1] - neighbor_atom_coord[1]) ## 弧度
        
        ## 绘制波浪线
        draw = ImageDraw.Draw(image)
        if is_wave_line:
            ## 控制为5/8的最大键长
            ## 不要x2
            x1 = - bond_length /8 * length_ratio 
            x2 = bond_length /8 * length_ratio 
            
            ## 随机控制振幅
            amp_ratio = random.choice([1, 0.9, 1.1])
            amplitude = bond_length/8 /4 * amp_ratio
            
            for x in range(int(x1)*100, int(x2)*100):
                delta_x = (x/100)
                delta_y = amplitude * math.cos((x/100-x1))
                new_x = delta_x*math.cos(angle) - delta_y*math.sin(angle) + middle_coord[0] #delta_y*math.sin(angle)
                new_y = delta_y*math.cos(angle) + delta_x*math.sin(angle) + middle_coord[1] #delta_y*math.cos(angle)
                
                draw.point((new_x, new_y), fill=(0, 0, 0))
                draw.point((new_x, new_y+0.1), fill=(0, 0, 0))
                draw.point((new_x, new_y-0.1), fill=(0, 0, 0))
        
        elif is_dash:
            x1 = - bond_length /8 * length_ratio
            x2 = bond_length /8 * length_ratio
            plot_bond_length = bond_length / 8 * length_ratio * 2
            dash_length = plot_bond_length / 9 
            segment_length = plot_bond_length / 9
            segments = 5
            # draw = ImageDraw.Draw(image)
            # draw.fontmode = "L"
            cv2_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            for i in range(segments):
                start_x = x1 + i * (dash_length + segment_length)
                end_x = start_x + segment_length
                delta_y = 0
                delta_start_x = start_x
                delta_end_x = end_x

                new_start_x = delta_start_x*math.cos(angle) - delta_y*math.sin(angle) + middle_coord[0] #delta_y*math.sin(angle)
                new_start_y = delta_y*math.cos(angle) + delta_start_x*math.sin(angle) + middle_coord[1] #delta_y*math.cos(angle)

                new_end_x = delta_end_x*math.cos(angle) - delta_y*math.sin(angle) + middle_coord[0] #delta_y*math.sin(angle)
                new_end_y = delta_y*math.cos(angle) + delta_end_x*math.sin(angle) + middle_coord[1] #delta_y*math.cos(angle)

                color = (0, 0, 0)

                # Line thickness of 5 px
                thickness = 2

                if length_ratio>=4:
                    line_type = cv2.LINE_AA  # 使用抗锯齿算法
                    cv2_image = cv2.line(cv2_image, (int(new_start_x), int(new_start_y)), (int(new_end_x), int(new_end_y)), color, thickness, line_type)
                else:
                    cv2_image = cv2.line(cv2_image, (int(new_start_x), int(new_start_y)), (int(new_end_x), int(new_end_y)), color, thickness)

            image = Image.fromarray(cv2_image)

        elif dash_line:
            
            angle = math.atan2(neighbor_atom_coord[0] - atom_coord[0], 
                            atom_coord[1] - neighbor_atom_coord[1]) ## 弧度

            x2 = neighbor_atom_coord[0]
            x1 = atom_coord[0]
            y2 = neighbor_atom_coord[1]
            y1 = atom_coord[1]
            segments = 4
            # draw = ImageDraw.Draw(image)
            cv2_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            for i in range(segments):
                start_x = x1 + ((i * 3)/(3*(segments)+1) ) *(x2-x1)
                end_x = x1 + ((i * 3+1)/(3*(segments)+1) ) *(x2-x1)
                start_y = y1 + ((i * 3)/(3*(segments)+1) ) *(y2-y1)
                end_y = y1 + ((i * 3+1)/(3*(segments)+1) ) *(y2-y1)
                
                # draw.line([(new_start_x, new_start_y), (new_end_x, new_end_y)], fill=(0, 0, 0), width=random.choice([2,3]), joint='curve')

                # Using cv2.line() method
                # Draw a diagonal black line with thickness of 5 px
                color = (255, 255, 255)

                # Line thickness of 5 px
                thickness = 2

                # Use anti-aliasing algorithm
                line_type = cv2.LINE_AA  # 使用抗锯齿算法
                cv2.line(cv2_image, (int(start_x), int(start_y)), (int(end_x), int(end_y)), color, thickness, line_type)

            image = Image.fromarray(cv2_image)

    ## 再用*进行替换
    for atom_index in extention:
        atom = mol.GetAtomWithIdx(atom_index)
        atom.SetAtomicNum(0) # Chem.GetPeriodicTable().GetAtomicNumber(dummy_symbol)
    
    
    graph = get_graph(mol, coords_list)
    graph["smiles"] = Chem.MolToSmiles(mol, canonical=False)
    graph["extention"] = ""
    graph["molblock"] = Chem.MolToMolBlock(mol, forceV3000=True)
    graph["new_smiles"] = graph["rdkit_smiles"]
    
    # print(graph["smiles"])

    return image, graph


def Normal_mol(mol, cdk_plugin):
    """随机生成带上下标的缩写来替换末端原子， 生成rdkit风格的 

    Args:
        mol (Chem.rdchem.Mol):
        cdk_plugin

    Returns:
        image (PIL.Image.image): 图片
        graph (dict): 字典
    """
    try:
        refer_smiles = Chem.MolToSmiles(mol)

        ## 在RDKit中，mol.UpdatePropertyCache()方法用于更新分子对象的属性缓存。
        ## RDKit在处理分子时，会计算并缓存许多分子的属性，例如原子的隐式氢数量、原子的价电子数量、分子的共轭系统等。
        ## 这些属性对于分子的进一步处理和分析非常重要。
        mol.UpdatePropertyCache()
        
        ## 是够添加所有的Hs
        if random.random()>0.9:
            Chem.AddHs(mol)
        else:
            ## 添加手信H
            mol = add_chirality_h_in_ring(mol)
        
        # if random.random()>0.5:
        mol = abbretion_replace(mol)

        if refer_smiles == Chem.MolToSmiles(mol):
            if random.random()>0.5:
                molfile_v3000_block = get_mol_block_v3000(refer_smiles, cdk_plugin)
                mol = Chem.MolFromMolBlock(molfile_v3000_block, removeHs=False)
                # Chem.Kekulize(mol)
                # if mol is None:
                #     return None, None
                image, coords_list = draw_molecule(copy.deepcopy(mol), with_coord=True)
            
            else:
                # Chem.Kekulize(mol)
                # if mol is None:
                #     return None, None
                image, coords_list = draw_molecule(copy.deepcopy(mol), with_coord=False)

        else:
            # Chem.Kekulize(mol)
            # if mol is None:
            #     return None, None
            image, coords_list = draw_molecule(copy.deepcopy(mol), with_coord=False)

        graph = get_graph(mol, coords_list)

        # graph["smiles"] = Chem.MolToSmiles(mol, canonical=False)
        graph["extention"] = ""
        graph["molblock"] = Chem.MolToMolBlock(mol, forceV3000=True)
        graph["molblock_v2000"] = Chem.MolToMolBlock(mol, forceV3000=False)

        debug = True
        if debug:
            image.save("temp_3.png")
            print(graph["smiles"])
        
        graph["new_smiles"] = graph["smiles"]

        return image, graph
    except Exception as e:
        print(e)
        import traceback
        traceback.print_exc()
        return None, None

def uncertainty_position_v2(mol:Chem.rdchem.Mol, 
                        cdk_plugin:dict, 
                        remove_fused_ring:bool = True, 
                        aug_angle:bool = True,
                        is_indigo:bool = False,
                        keep_scaffold:bool=False,
                        highlight:bool=False,
                        debug:bool=False):

    if keep_scaffold:
        mol:Chem.rdchem.Mol = keep_scaffold_fn(mol)
    
    #会画在并环上，有点问题
    if remove_fused_ring:
        # result_ring_list:[[0, 22, 21, 3, 2, 1], [5, 6, 7, 8, 9, 10, 11, 18, 19, 20], [14, 13, 17, 16, 15]]
        # [False, True, False]
        result_ring_list, fused_token_list = get_merged_ring(mol)
        ## 剔除并环,稠环和桥环
        new_result_ring_list = []
        for _ in range(len(result_ring_list)):
            if fused_token_list[_] is False:
                new_result_ring_list.append(result_ring_list[_])
        atominfos = new_result_ring_list
    
    ## 只考虑原子数小于8的环
    new_atominfos = []
    for atom_ring in atominfos:
        if len(atom_ring)<8:
            new_atominfos.append(atom_ring)
    atominfos = new_atominfos

    if len(atominfos)>0:
        ## 只考虑一个
        num_elements = 1
        extention_rings = random.sample(atominfos, num_elements)
        # print("select")
        total_index_list = []
        start_id_list = []
        end_id_list = []
        cross_points_list = []
        for ring in extention_rings:
            ## 只选奇数个
            round_count = 0
            Flag = False
            while True:
                select_num_atom = random.choice([_ for _ in range(2, len(ring)+1) if _%2 == 1 ])
                index_list = []
                count = 0
                while len(index_list)<len(ring):
                    ## 初始的时候随机选择原子
                    if len(index_list) == 0:
                        atom_idx = random.sample(ring, 1)[0]
                        index_list.append(atom_idx)
                    else:
                        start_atom_idx = index_list[-1]
                        for end_atom_idx in ring:
                            if end_atom_idx not in index_list:
                                bond = mol.GetBondBetweenAtoms(end_atom_idx, start_atom_idx)
                                if bond is not None:
                                    index_list.append(end_atom_idx)
                                    break
                    
                    if len(index_list)>=select_num_atom:
                        break
                
                if (len(index_list) % 2 == 1):
                    mid_atom_idx = index_list[len(index_list)//2]
                    atom = mol.GetAtomWithIdx(mid_atom_idx)
                    neighbor = atom.GetNeighbors()
                    if len(neighbor) == 2:
                        Flag = True
                        break

                round_count = round_count + 1
            
            if Flag is True:
                total_index_list.append(index_list)
        
        if len(total_index_list) == 0:
            return None, None
        
        smiles = Chem.MolToSmiles(mol, canonical=False)
        smiles_extention = ""
        extention_string = "m:"
        start_id = mol.GetNumAtoms()
        for _ in range(len(total_index_list)):
            ##这边frag只会是两个
            frag = random.sample(small_fragment, 1)[0]
            smiles_extention += "."+frag
            extention_string+=f"{start_id}:"
            for k, atom_idx in enumerate(total_index_list[_]):
                extention_string += f"{atom_idx}"
                if k == len(total_index_list[_]) - 1:
                    if _ != (len(total_index_list)-1):
                        extention_string += ","
                else:
                    extention_string += "."

            try:
                count = Chem.MolFromSmiles(frag).GetNumAtoms()
            except Exception as e:
                print(e)
                # ipdb.set_trace()
                return None, None
            start_id_list.append(start_id)
            end_id_list.append(start_id + count - 1)
            start_id = start_id + count
        
        last_smiles_string = smiles + smiles_extention + " |" + extention_string + "|"
        molfile_v3000_block = get_mol_block_v3000(last_smiles_string, cdk_plugin)

        mol = Chem.MolFromMolBlock(molfile_v3000_block, removeHs=False) ##仅仅是个画图的工具人
        if random.random()>0.5:
            mol = abbretion_replace(mol)

        mid_atom_degree_dict = {}
        for i, index_list in enumerate(total_index_list):
            mid_atom_idx = index_list[len(index_list)//2]
            Flag = False
            for ring in extention_rings:
                if mid_atom_idx in ring:
                    Flag = True
                    break
            ring_coords = []
            for atom_idx in ring:
                position = mol.GetConformer().GetAtomPosition(atom_idx)
                ring_coords.append((position.x, position.y))
            
            ring_center_coords = np.mean(np.array(ring_coords),axis=0).tolist()

            mid_atom_coords = mol.GetConformer().GetAtomPosition(mid_atom_idx)
            mid_atom_coords = [mid_atom_coords.x, mid_atom_coords.y]
            neight_mid_atom_idx = index_list[len(index_list)//2-1]
            neight_mid_atom_coords = mol.GetConformer().GetAtomPosition(neight_mid_atom_idx)
            neight_mid_atom_coords = [neight_mid_atom_coords.x, neight_mid_atom_coords.y]
            bond_length = math.sqrt((neight_mid_atom_coords[0]-mid_atom_coords[0])**2 + (neight_mid_atom_coords[1]-mid_atom_coords[1])**2)
            length_to_center = math.sqrt((ring_center_coords[0]-mid_atom_coords[0])**2 + (ring_center_coords[1]-mid_atom_coords[1])**2)
            # total_lenth = bond_length + length_to_center

            angle = math.atan2(mid_atom_coords[1] - ring_center_coords[1], 
                                mid_atom_coords[0] - ring_center_coords[0])
            
            shift_degree = 0
            if aug_angle:
                # 加减5度
                shift_degree = random.choice([0, 1, -1, 2.5, -2.5, -3.5, 3.5]) #

            if shift_degree!=0:
                # 将原始角度转换为度数
                degrees = math.degrees(angle)
                # 将新的角度转换为弧度
                angle = math.radians(shift_degree + degrees)
            
            mid_atom_degree_dict[mid_atom_idx] = shift_degree

            ratio = random.choice([2,3,4])
            x1 = ring_center_coords[0] + length_to_center*(ratio-1)/ratio * math.cos(angle) ## 中心点
            y1 = ring_center_coords[1] + length_to_center*(ratio-1)/ratio * math.sin(angle) ## 中心点
            x2 = ring_center_coords[0] + (length_to_center*(ratio-1)/ratio + bond_length) * math.cos(angle)
            y2 = ring_center_coords[1] + (length_to_center*(ratio-1)/ratio + bond_length) * math.sin(angle)

            start_id = start_id_list[i]
            end_id = end_id_list[i]
            conf = mol.GetConformer()
            conf.Set3D(False)
            conf.SetAtomPosition(start_id, (x1, y1, 0))
            conf.SetAtomPosition(end_id, (x2, y2, 0))

        ## 判断分子合法性
        try:
            Chem.Kekulize(mol)
        except Chem.rdchem.KekulizeException:
            return None, None
        if mol is None:
            return None, None
        
        if is_indigo:
            image, graph = indigo_get_graph(mol)
            coords_list = graph["coords"]
        else:
            # if highlight is True, it will highlight the related the prefix atom
            image, coords_list = draw_molecule(copy.deepcopy(mol), highlight=False) 
            graph = get_graph(mol, coords_list)
        
        if graph["num_atoms"] == -1:
            return None, None

        graph["smiles"] = Chem.MolToSmiles(mol, canonical=False)
        graph["extention"] = extention_string
        graph["molblock"] = Chem.MolToMolBlock(mol, forceV3000=True)
        graph["molblock_v2000"] = Chem.MolToMolBlock(mol, forceV3000=False)
        
        result, success = get_result_smiles(graph["symbols"], graph["smiles"])
        graph["new_smiles"] = result
        if success is False:
            return None, None

        cross_points_list = []
        for mid_atom_idx, shift_degree in mid_atom_degree_dict.items():
            cross_points_list.append(coords_list[mid_atom_idx])
        graph["cross_points"] = cross_points_list

        # debug = True
        if debug:
            draw = ImageDraw.Draw(image)
            for i, point in enumerate(cross_points_list):
                x, y = point
                draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="red")
            image.save("temp_2.png")

        return image, graph

    else:
        return None, None

def get_merged_ring(mol, atominfos=None):
    """将稠环，螺环，桥环合并

    Args:
        mol (rdkit.Chem.rdchem.Mol): 分子.

    Return:
        result_ring_list (list): 原子序号的列表, 
        fused_token_list (list): 是否为并环的列表, fused_token_list中对应的位置为True的时候表示该环为稠环，螺环，桥环
    """
    ## 获取环的原子信息
    if atominfos is None:
        ## new version
        atominfos = []
        for ring in Chem.GetSSSR(mol):
            atominfos.append(list(ring))
        
        # old version
        # ring = mol.GetRingInfo() # rdkit.Chem.rdchem.RingInf
        # atominfos = ring.AtomRings() # tuple(tuple())
    
    ## 将环的原子信息转化为列表
    ring_list = [[_ for _ in ring] for ring in atominfos] # list(list())
    ## 维护一个列表来表示该环是否已经被合并
    is_add_list = [False] * len(atominfos) # list(), 控制是否已经将环加入的token的列表

    ring_index_list = []
    result_ring_list = []
    fused_token_list = []
    i = 0
    fused_token = False
    new_list_token = True ## 控制是否生成新的table
    while i <len(ring_list):

        ## 如果全部已经添加则退出
        if sum(is_add_list) == len(is_add_list):
            break
        
        ## 如果已经加入了,就跳过
        if is_add_list[i] is True:
            i = i + 1
            continue

        else:
            if new_list_token:
                ring_index_list.append([])
                new_list_token = False
            
            if i not in ring_index_list[-1]:
                ring_index_list[-1].append(i)
        
        ## 环1
        ring_1 = ring_list[i]
        for j in range(i+1, len(ring_list)):

            ## 如果已经加入了,就跳过
            if is_add_list[j] is True:
                continue
            
            ## 环2
            ring_2 = ring_list[j]
            ## 如果ring_1和ring_2之间有重合，就将ring_2的索引加入到ring_index_list，并设置is_add_list[j] = True
            if len(set(ring_1)&set(ring_2))>0:
                is_add_list[j] = True
                ring_index_list[-1].append(j)

        temp_ring = []
        for idx in ring_index_list[-1]:
            temp_ring.extend(ring_list[idx])
        temp_ring = list(set(temp_ring))
        
        # print(len(temp_ring), len(ring_list[i]))
        ## 比较ring_list[i]和temp_ring的长度，如果长度一致，说明没有新的原子加入，就遍历下一个原子；
        if len(temp_ring) == len(ring_list[i]):
            result_ring_list.append(ring_list[i])
            fused_token_list.append(fused_token)
            is_add_list[i] = True
            i = i + 1
            new_list_token = True
            fused_token = False
        ## 反之，则表明有新的原子加入，则继续在已经加入的基础上进行遍历
        else:
            fused_token = True
            ring_list[i] = temp_ring
    
    return result_ring_list, fused_token_list

def uncertainty_position_v1(mol: Chem.rdchem.Mol, 
                        cdk_plugin : dict, 
                        remove_fused_ring :bool = True, 
                        is_indigo : bool = False,
                        keep_scaffold : bool = False,
                        highlight : bool = False,
                        debug : bool = False):
    
    if keep_scaffold:
        mol = keep_scaffold_fn(mol)

    if remove_fused_ring:
        # 会画在并环上，有点问题
        result_ring_list, fused_token_list = get_merged_ring(mol)
        # result_ring_list = [result_ring_list[_] for _ in range(len(result_ring_list)) if fused_token_list[_] is False]
        new_result_ring_list = []
        for _ in range(len(result_ring_list)):
            if fused_token_list[_] is False:
                new_result_ring_list.append(result_ring_list[_])
        atominfos = new_result_ring_list
    else:
        ## 不会考虑桥环
        ring_info = mol.GetRingInfo()
        # tuple(tuple())
        atominfos = ring_info.AtomRings() 
    
    new_atominfos = []
    for atom_ring in atominfos:
        ## 对小于8个的原子进行处理
        if len(atom_ring)<8:
            new_atominfos.append(atom_ring)
    atominfos = new_atominfos
    # print(result_ring_list, fused_token_list)
    # print(atominfos)
    if len(atominfos)>0:
        # print(min(3, len(atominfos)))
        ## 随机选择进行插入
        if len(atominfos)>=4:
            num_elements = random.randint(1, min(2, len(atominfos))) #total_index_list
        else:
            num_elements = 1
        
        extention_rings = random.sample(atominfos, num_elements)
        
        total_index_list = []
        for ring in extention_rings:
            ## 只选偶数个
            select_num_atom = random.choice([_ for _ in range(1, len(ring)+1) if _%2 == 0 ])
            index_list = []
            count = 0
            while len(index_list)<len(ring):
                ## 初始的时候随机选择原子
                if len(index_list) == 0:
                    atom_idx = random.sample(ring, 1)[0]
                    index_list.append(atom_idx)
                else:
                    start_atom_idx = index_list[-1]
                    for end_atom_idx in ring:
                        if end_atom_idx not in index_list:
                            bond = mol.GetBondBetweenAtoms(end_atom_idx, start_atom_idx)
                            if bond is not None:
                                index_list.append(end_atom_idx)
                                break
                
                if len(index_list)>=select_num_atom:
                    break
        
            total_index_list.append(index_list)
        
        # print("finish select")
        if len(total_index_list) == 0:
            return None, None
        
        smiles = Chem.MolToSmiles(mol, canonical=False)
        smiles_extention = ""
        extention_string = "m:"
        start_id = mol.GetNumAtoms()
        atom_pair_dict = {}
        for _ in range(len(total_index_list)):
            frag = random.sample(small_fragment, 1)[0]
            smiles_extention += "."+frag
            extention_string+=f"{start_id}:"
            for k, atom_idx in enumerate(total_index_list[_]):
                extention_string += f"{atom_idx}"
                if k == len(total_index_list[_]) - 1:
                    if _ != (len(total_index_list)-1):
                        extention_string += ","
                else:
                    extention_string += "."
                    
            try:
                count = Chem.MolFromSmiles(frag).GetNumAtoms()
                atom_pair_dict[(start_id, start_id + count -1)] = total_index_list[_]
            except Exception as e:
                print(e)
                # ipdb.set_trace()
                return None, None
            start_id = start_id + count
        
        last_smiles_string = smiles + smiles_extention + " |" + extention_string + "|"
        molfile_v3000_block = get_mol_block_v3000(last_smiles_string, cdk_plugin)

        mol = Chem.MolFromMolBlock(molfile_v3000_block, removeHs=False) ##仅仅是个画图的工具人
        if random.random()>0.5:
            mol = abbretion_replace(mol)
        
        try:
            Chem.Kekulize(mol)
        except Chem.rdchem.KekulizeException:
            return None, None
        if mol is None:
            return None, None
            
        if is_indigo:
            image, graph = indigo_get_graph(mol)
            coords_list = graph["coords"]
        else:
            image, coords_list = draw_molecule(copy.deepcopy(mol), highlight=False) #highlight=True
            graph = get_graph(mol, coords_list)
        
        if graph["num_atoms"] == -1:
            return None, None

        graph["smiles"] = Chem.MolToSmiles(mol, canonical=False)
        graph["extention"] = extention_string
        graph["molblock"] = Chem.MolToMolBlock(mol, forceV3000=True)
        graph["molblock_v2000"] = Chem.MolToMolBlock(mol, forceV3000=False)
        
        result, success = get_result_smiles(graph["symbols"], graph["smiles"])
        graph["new_smiles"] = result
        if success is False:
            return None, None

        cross_points_list = []
        for atom_pair, index_list in atom_pair_dict.items():
            start_atom_idx = atom_pair[0]
            # start_atom = mol.GetAtomWithIdx(start_atom_idx)
            start_atom_coords = coords_list[start_atom_idx]
            end_atom_idx = atom_pair[1]
            # end_atom = mol.GetAtomWithIdx(end_atom_idx)
            end_atom_coords = coords_list[end_atom_idx]

            distance = np.inf
            result = None

            for _ in range(len(index_list)):
                mid_1 = index_list[_%len(index_list)]
                mid_1_coords = coords_list[mid_1]
                mid_2 = index_list[(_+1)%len(index_list)]
                mid_2_coords = coords_list[mid_2]

                ## 判断两个原子有没有键
                bond = mol.GetBondBetweenAtoms(mid_1, mid_2)
                if bond is None:
                    continue

                points = (start_atom_coords, end_atom_coords, mid_1_coords, mid_2_coords)
                intersection = find_intersection(points)
                if intersection[0] is None:
                    continue
                temp_distance = math.sqrt((intersection[0]-end_atom_coords[0])**2 + (intersection[1]-end_atom_coords[1])**2)
                
                if temp_distance < distance:
                    distance = temp_distance
                    result = intersection

            if result is not None:
                cross_points_list.append(result)

        # graph["cross_points"] = cross_points_list

        # debug = True
        # if debug:
        #     draw = ImageDraw.Draw(image)
        #     for i, point in enumerate(cross_points_list):
        #         x, y = point
        #         draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="red")
        #     for atom_pair, index_list in atom_pair_dict.items():
        #         start_atom_idx = atom_pair[0]
        #         # start_atom = mol.GetAtomWithIdx(start_atom_idx)
        #         start_atom_coords = coords_list[start_atom_idx]
        #         end_atom_idx = atom_pair[1]
        #         # end_atom = mol.GetAtomWithIdx(end_atom_idx)
        #         end_atom_coords = coords_list[end_atom_idx]
        #         draw.ellipse((start_atom_coords[0] - 5, start_atom_coords[1] - 5, start_atom_coords[0] + 5, start_atom_coords[1] + 5), fill="blue")
        #         draw.ellipse((end_atom_coords[0] - 5, end_atom_coords[1] - 5, end_atom_coords[0] + 5, end_atom_coords[1] + 5), fill="green")
        #     image.save("temp_2.png")
        #     print(graph["new_smiles"])
            
        #     import ipdb
        #     ipdb.set_trace()
        

        return image, graph

    else:
        return None, None

def get_polymer_result(mol, 
                       end_atoms_idx_list, 
                       cdk_plugin, 
                       is_wave_line=False):

    ## 对常见的两个下标随机采样
    sub = random.sample(["m", "n", ""], 1)[0]
    if sub != "":
        ## 有0.5的概率添加=号
        if random.random()>0.5:
            ## 如果添加了=号，则进一步有0.5的概率添加一个数字或者两个数字
            randint = random.randint(1,2)
            if randint == 1:
                random_num = random.randint(1, 10)
                sub += f"={random_num}"
            else:
                random_num_list = []
                while len(random_num_list)<2:
                    random_num = random.randint(1, 10)
                    if random_num not in random_num_list:
                        random_num_list.append(random_num)
                random_num_list.sort()
                sub += f"={random_num_list[0]}-{random_num_list[1]}"

    ## 获取smiles
    smiles = Chem.MolToSmiles(mol, canonical=False)
    extention_string = "Sg:n:"
    extention_list = []
    for atom in mol.GetAtoms():
        if atom.GetIdx() not in end_atoms_idx_list:
            extention_list.append(f"{atom.GetIdx()}")
    
    if len(extention_list)>15:
        return None, None
    
    extention_string += (",".join(extention_list)) + ":"+sub+":ht"
    last_smiles_string = smiles + " |" + extention_string + "|"
    molfile_v3000_block = get_mol_block_v3000(last_smiles_string, cdk_plugin)
    new_molfile_v3000_block = molfile_v3000_block.replace("CONNECT=HT ", "")

    mol = Chem.MolFromMolBlock(new_molfile_v3000_block, removeHs=False) ##仅仅是个画图的工具人
    if (is_wave_line is False) and (random.random()>0.5):
        mol = abbretion_replace(mol)

    try:
        Chem.Kekulize(mol)
    except Chem.rdchem.KekulizeException:
        return None, None
    if mol is None:
        return None, None
    
    image, coords_list = draw_molecule(copy.deepcopy(mol), is_wave_line=is_wave_line)
    graph = get_graph(mol, coords_list)
    
    graph["smiles"] = Chem.MolToSmiles(mol, canonical=False)
    graph["extention"] = extention_string
    graph["molblock"] = Chem.MolToMolBlock(mol, forceV3000=True)

    import ipdb
    ipdb.set_trace()

    return image, graph


def polymer(mol, cdk_plugin, debug=False):
    is_wave_line = False

    end_atoms_idx_list = get_end_atoms_idx(mol)
    ## 有0.8的概率走这一条路
    if len(end_atoms_idx_list)>1 and random.random()<0.8:
        # if len(end_atoms_idx_list) == 1:
        #     image, graph = get_polymer_result(mol, end_atoms_idx_list)

        if len(end_atoms_idx_list) == 2 and len(mol.GetAtoms())>=3:
            if random.random()<0.5:
                for atom_idx in end_atoms_idx_list:
                    atom = mol.GetAtomWithIdx(atom_idx)
                    atom.SetAtomicNum(Chem.GetPeriodicTable().GetAtomicNumber(dummy_symbol))

                if random.random()<0.5:
                    is_wave_line = True

            image, graph = get_polymer_result(mol, end_atoms_idx_list, cdk_plugin, is_wave_line)
            # if random.random()<0.7:
            #     image, graph = get_polymer_result(mol, end_atoms_idx_list)

            ## 有0.3的概率将末端原子考虑进来
            # else:
            #     end_atoms_idx_list = random.sample(end_atoms_idx_list, 1)
            #     image, graph = get_polymer_result(mol, end_atoms_idx_list)
            
        else:
            smiles = Chem.MolToSmiles(mol)
            # 提取化合物的骨架
            scaffold = MurckoScaffold.MurckoScaffoldSmilesFromSmiles(smiles)

            # 创建骨架的 RDKit 分子对象
            scaffold_mol = Chem.MolFromSmiles(scaffold)

            mol = Chem.MolFromSmiles(Chem.MolToSmiles(scaffold_mol))
            end_atoms_idx_list = get_end_atoms_idx(mol)

            ### 只考虑一个原子和两个原子的情况
            # if len(end_atoms_idx_list) == 1:
            #     image, graph = get_polymer_result(mol, end_atoms_idx_list)

            if len(end_atoms_idx_list) == 2 and len(mol.GetAtoms())>=3:
                if random.random()<0.7:
                    if random.random()>0.5:
                        is_wave_line = True
                    image, graph = get_polymer_result(mol, end_atoms_idx_list, cdk_plugin, is_wave_line)
                ## 有0.3的概率将末端原子考虑进来
                else:
                    if random.random()>0.5:
                        is_wave_line = True
                    end_atoms_idx_list = random.sample(end_atoms_idx_list, 1)
                    image, graph = get_polymer_result(mol, end_atoms_idx_list, cdk_plugin, is_wave_line)
            
            else:
                return None, None
    
    ## 找到所有只有两个单键相连的原子
    else:
        # 遍历每个原子
        two_single_bond_list = []
        for atom in mol.GetAtoms():
            atom_index = atom.GetIdx()
            neighbor_bonds = atom.GetBonds()

            # 计数单键相连的数量
            single_bond_count = 0
            for bond in neighbor_bonds:
                if bond.GetBondType() == rdkit.Chem.rdchem.BondType.SINGLE:
                    single_bond_count += 1
                else:
                    single_bond_count += 999

            # 检查单键相连的数量是否为 2
            if single_bond_count == 2:
                # print("Atom index:", atom_index)
                two_single_bond_list.append(atom_index)
        
        if len(two_single_bond_list)>0:
            random_atom_idx = random.sample(two_single_bond_list, 1)[0]
            extention_string = "Sg:n:"
            sub = random.sample(["m", "n", ""], 1)[0]
            if sub != "":
                if random.random()>0.5:
                    randint = random.randint(1,2)
                    if randint == 1:
                        random_num = random.randint(1, 10)
                        sub += f"={random_num}"
                    else:
                        random_num_list = []
                        while len(random_num_list)<2:
                            random_num = random.randint(1, 10)
                            if random_num not in random_num_list:
                                random_num_list.append(random_num)
                        random_num_list.sort()
                        sub += f"={random_num_list[0]}-{random_num_list[1]}"
            else:
                if random.random()>0.5:
                    randint = random.randint(1,2)
                    if randint > 1:
                        random_num_list = []
                        while len(random_num_list)<2:
                            random_num = random.randint(1, 10)
                            if random_num not in random_num_list:
                                random_num_list.append(random_num)
                        random_num_list.sort()
                        sub += f"{random_num_list[0]}-{random_num_list[1]}"

            ## 写入结果
            smiles = Chem.MolToSmiles(mol, canonical=False)
            extention_string += f"{random_atom_idx}"+":"+sub+":ht"
            last_smiles_string = smiles + " |" + extention_string + "|"
            molfile_v3000_block = get_mol_block_v3000(last_smiles_string, cdk_plugin)
            molfile_v3000_block = molfile_v3000_block.replace("CONNECT=HT ", "")

            mol = Chem.MolFromMolBlock(molfile_v3000_block, removeHs=False) ##仅仅是个画图的工具人

            if random.random()>0.5:
                mol = abbretion_replace(mol)
            Chem.Kekulize(mol)
            if mol is None:
                return None, None
            image, coords_list = draw_molecule(copy.deepcopy(mol))
            graph = get_graph(mol, coords_list)

            graph["smiles"] = Chem.MolToSmiles(mol, canonical=False)
            graph["extention"] = extention_string
            graph["molblock"] = Chem.MolToMolBlock(mol, forceV3000=True)
            
        
        else:
            return None, None
    
    if image is not None:
        # image.save("temp.png")
        # print((graph["smiles"]+" |" + graph["extention"]+"|"))
        if debug:
            image.save("temp.png")
            print((graph["smiles"]+" |" + graph["extention"]+"|"))

        new_smiles, success = get_result_smiles(graph["symbols"],  graph["rdkit_smiles"])
        if success:
            return image, graph
        else:
            return None, None
    else:
        return None, None

def get_canonical_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles) #一定要设置`sanitize=False`, 不然会改变输入smiles的原子顺序, 但是可能会存在画坐标失败
        mol = AllChem.RemoveHs(mol) ## 去除自由基
        canonical_smiles = Chem.MolToSmiles(mol)
        return canonical_smiles
    
    except Exception as e:
        print(e)
        return ""

def random_generate_rgroup_type_2(idx:int=0, 
                                  original_smiles:str=None, 
                                  cdk_plugin:dict={}, 
                                  **kwargs):
    """_summary_

    Args:
        idx (int, optional): index of smiles. Defaults to 0.
        original_smiles (str, optional): smiles of molecule. Defaults to None.
        cdk_plugin (dict, optional): cdk_plugin. Defaults to {}.

    Returns:
        image (PIL.PngImagePlugin.PngImageFile) : image of smiles 
        result_dict (dict): result of molecule
    """
    if original_smiles is None:
        original_smiles = "CN(C)C\C=C\C(=O)Nc3cc1c(Nc(cc2Cl)ccc2F)ncnc1cc3OC4COCC4"
    ## keep the largest component in molecule
    original_smiles = keep_largest_component_fn(original_smiles)
    ## Canonilize the mole cule
    canonical_smiles = get_canonical_smiles(original_smiles)
    mol = Chem.MolFromSmiles(canonical_smiles, sanitize=False)
    ## large molecule is too crowded in picture
    if len(mol.GetAtoms())>45:
        return None, None
    
    ## generate markush molecule
    image, graph = None, None
    count = 0
    markush_type = "RGroup"
    while count < 100:
        count = count + 1
        image, graph = RGroup_2(mol, cdk_plugin, kwargs)

        ## invalid molecule during drawing
        if (graph is None) or (image is None):
            continue
        elif graph["smiles"] == "" or graph["num_atoms"] <= 0:
            continue
        elif image is not None:
            if len(np.array(image).shape) == 3:
                break
        else:
            break
    
    result_dict = {
                "%d"%(idx):{
                "smiles":graph["smiles"], #已经被覆盖了
                "original_smiles":original_smiles,
                "graph":graph,
                "markush_type":markush_type,
                }
            }

    return image, result_dict 


def random_generate_normal_mol(idx:int=0, 
                               original_smiles:str=None, 
                               cdk_plugin:dict={},
                               **kwargs):
    """_summary_

    Args:
        idx (int, optional): index of smiles. Defaults to 0.
        original_smiles (str, optional): smiles of molecule. Defaults to None.
        cdk_plugin (dict, optional): cdk_plugin. Defaults to {}.

    Returns:
        image (PIL.PngImagePlugin.PngImageFile) : image of smiles 
        result_dict (dict): result of molecule
    """
    
    if original_smiles is None:
        original_smiles = "CN(C)C\C=C\C(=O)Nc3cc1c(Nc(cc2Cl)ccc2F)ncnc1cc3OC4COCC4"
    ## keep the largest component in molecule
    original_smiles = keep_largest_component_fn(original_smiles)
    ## Canonilize the mole cule
    canonical_smiles = get_canonical_smiles(original_smiles)
    mol = Chem.MolFromSmiles(canonical_smiles, sanitize=False)
    ## large molecule is too crowded in picture
    if len(mol.GetAtoms())>45:
        return None, None
    
    ## generate markush molecule
    image, graph = None, None
    count = 0
    markush_type = "Normal"
    while count < 100:
        count = count + 1
        image, graph = Normal_mol(mol, cdk_plugin)

        ## invalid molecule during drawing
        if (graph is None) or (image is None):
            continue
        elif graph["smiles"] == "" or graph["num_atoms"] <= 0:
            continue
        elif image is not None:
            if len(np.array(image).shape) == 3:
                break
        else:
            break
    
    result_dict = {
                "%d"%(idx):{
                "smiles":graph["smiles"], #已经被覆盖了
                "original_smiles":original_smiles,
                "graph":graph,
                "markush_type":markush_type,
                }
            }

    return image, result_dict 

def random_generate_uncertainty_position_v1(idx:int=0, 
                                            original_smiles:str=None, 
                                            cdk_plugin:dict={}, 
                                            is_indigo:bool=False):
    """_summary_

    Args:
        idx (int, optional): index of smiles. Defaults to 0.
        original_smiles (str, optional): smiles of molecule. Defaults to None.
        cdk_plugin (dict, optional): cdk_plugin. Defaults to {}.
        is_indigo (bool, optional)
    Returns:
        image (PIL.PngImagePlugin.PngImageFile) : image of smiles 
        result_dict (dict): result of molecule
    """
    if original_smiles is None:
        original_smiles = "CN(C)C\C=C\C(=O)Nc3cc1c(Nc(cc2Cl)ccc2F)ncnc1cc3OC4COCC4"
    ## keep the largest component in molecule
    original_smiles = keep_largest_component_fn(original_smiles)
    ## Canonilize the mole cule
    canonical_smiles = get_canonical_smiles(original_smiles)
    mol = Chem.MolFromSmiles(canonical_smiles, sanitize=False)
    ## large molecule is too crowded in picture
    if len(mol.GetAtoms())>45:
        return None, None
    
    ## generate markush molecule
    image, graph = None, None
    count = 0
    markush_type = "uncertainty_position"
    while count < 100:
        count = count + 1
        image, graph = uncertainty_position_v1(mol, cdk_plugin, is_indigo=is_indigo)

        ## invalid molecule during drawing
        if (graph is None) or (image is None):
            continue
        elif graph["smiles"] == "" or graph["num_atoms"] <= 0:
            continue
        elif image is not None:
            if len(np.array(image).shape) == 3:
                break
        else:
            break
    
    result_dict = {
                "%d"%(idx):{
                "smiles":graph["smiles"], #已经被覆盖了
                "original_smiles":original_smiles,
                "graph":graph,
                "markush_type":markush_type,
                }
            }

    return image, result_dict 

def random_generate_uncertainty_position_v2(idx:int=0, original_smiles:str=None, cdk_plugin:dict={}, is_indigo=False):
    if original_smiles is None:
        original_smiles = "CN(C)C\C=C\C(=O)Nc3cc1c(Nc(cc2Cl)ccc2F)ncnc1cc3OC4COCC4"
    ## keep the largest component in molecule
    original_smiles = keep_largest_component_fn(original_smiles)
    ## Canonilize the mole cule
    canonical_smiles = get_canonical_smiles(original_smiles)
    mol = Chem.MolFromSmiles(canonical_smiles, sanitize=False)
    ## large molecule is too crowded in picture
    if len(mol.GetAtoms())>45:
        return None, None
    
    ## generate markush molecule
    image, graph = None, None
    count = 0
    markush_type = "uncertainty_position"
    while count < 100:
        count = count + 1
        image, graph = uncertainty_position_v2(mol, cdk_plugin, is_indigo=is_indigo)

        ## invalid molecule during drawing
        if (graph is None) or (image is None):
            continue
        elif graph["smiles"] == "" or graph["num_atoms"] <= 0:
            continue
        elif image is not None:
            if len(np.array(image).shape) == 3:
                break
        else:
            break
    
    result_dict = {
                "%d"%(idx):{
                "smiles":graph["smiles"], #已经被覆盖了
                "original_smiles":original_smiles,
                "graph":graph,
                "markush_type":markush_type,
                }
            }
    return image, result_dict 


def random_generate_polymer(idx:int=0, original_smiles:str=None, cdk_plugin:dict={}):
    if original_smiles is None:
        original_smiles = "CN(C)C\C=C\C(=O)Nc3cc1c(Nc(cc2Cl)ccc2F)ncnc1cc3OC4COCC4"
    ## keep the largest component in molecule
    original_smiles = keep_largest_component_fn(original_smiles)
    ## Canonilize the mole cule
    canonical_smiles = get_canonical_smiles(original_smiles)
    mol = Chem.MolFromSmiles(canonical_smiles, sanitize=False)
    ## large molecule is too crowded in picture
    if len(mol.GetAtoms())>45:
        return None, None
    
    ## generate markush molecule
    image, graph = None, None
    count = 0
    markush_type = "polymer"
    while count < 100:
        count = count + 1
        image, graph = polymer(mol, cdk_plugin)

        ## invalid molecule during drawing
        if (graph is None) or (image is None):
            continue
        elif graph["smiles"] == "" or graph["num_atoms"] <= 0:
            continue
        elif image is not None:
            if len(np.array(image).shape) == 3:
                break
        else:
            break
    
    result_dict = {
                "%d"%(idx):{
                "smiles":graph["smiles"], #已经被覆盖了
                "original_smiles":original_smiles,
                "graph":graph,
                "markush_type":markush_type,
                }
            }
    return image, result_dict 


if __name__ == "__main__":
    ## get cdk pulgin for prepare MolFile with V300
    idx = 0
    cdk_plugin = get_cdk_plugin()
    original_smiles = "CN(C)C\C=C\C(=O)Nc3cc1c(Nc(cc2Cl)ccc2F)ncnc1cc3OC4COCC4"
    kwargs = {
        "is_dash" : False, 
        "is_wave_line" : False, 
        "dash_line" : False
    }
    

    # image, result_dict = random_generate_rgroup_type_2(
    #                                                     idx = idx,
    #                                                     original_smiles = original_smiles,
    #                                                     cdk_plugin = cdk_plugin,
    #                                                     **kwargs
    #                                                     )
    # import ipdb
    # ipdb.set_trace()

    image, result_dict = random_generate_normal_mol(
                                                    idx = idx,
                                                    original_smiles = original_smiles,
                                                    cdk_plugin = cdk_plugin,
                                                    )

    # image, result_dict = random_generate_uncertainty_position_v1(
    #                                                 idx = idx,
    #                                                 original_smiles = original_smiles,
    #                                                 cdk_plugin = cdk_plugin,
    #                                                 )

    # image, result_dict = random_generate_uncertainty_position_v1(
    #                                                 idx = idx,
    #                                                 original_smiles = original_smiles,
    #                                                 cdk_plugin = cdk_plugin,
    #                                                 is_indigo=True
    #                                                 )

    # image, result_dict = random_generate_uncertainty_position_v2(
    #                                                 idx = idx,
    #                                                 original_smiles = original_smiles,
    #                                                 cdk_plugin = cdk_plugin,
    #                                                 )

    # image, result_dict = random_generate_uncertainty_position_v2(
    #                                                 idx = idx,
    #                                                 original_smiles = original_smiles,
    #                                                 cdk_plugin = cdk_plugin,
    #                                                 is_indigo=True
    #                                                 )

    image, result_dict = random_generate_polymer(
                                                    idx = idx,
                                                    original_smiles = original_smiles,
                                                    cdk_plugin = cdk_plugin,
                                                    )
    