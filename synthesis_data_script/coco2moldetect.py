import os
import json
from tqdm import tqdm
import argparse

## standardized the imagePath in labelme
def standard_data(file_path):
    with open(file_path, "r") as f:
        train_data = json.loads(f.read())
    ## 定义license
    train_data["licenses"] = [{'name': '', 'id': 0, 'url': ''}]
    ## 调整原始的images为annotations，调整原始的annotations为_images，删除images
    train_data["annotations"], train_data["_images"] = train_data["images"], train_data["annotations"]
    train_data["images"] = []

    ## 类别索引映射
    train_data_cat_dict = {}
    for _ in train_data["categories"]:
        train_data_cat_dict[_["id"]] = _["name"]
    
    print(train_data_cat_dict)
    
    ## 重建images
    result_list = []
    for temp in tqdm(train_data["annotations"]):
        result_dict = {}
        result_dict["file_name"] = temp["file_name"]
        result_dict["id"]  = temp["id"]
        result_dict["width"] = temp["width"]
        result_dict["height"]  = temp["height"]
        bboxes = []
        corefs = []
        for _ in train_data["_images"]:
            if _["image_id"] == result_dict["id"]:
                bboxes.append({
                    'id': len(bboxes), 'bbox': _["bbox"], 'category_id': int(train_data_cat_dict[_["category_id"]])
                })
        
        temp_list = []
        for _ in bboxes:
            if _["category_id"] == 1:
                if len(temp_list)>0:
                    corefs.append(temp_list)
                temp_list = []
                temp_list.append(_["id"])
            else:
                temp_list.append(_["id"])
        
        if len(temp_list)>0:
            corefs.append(temp_list)
        temp_list = []

        result_dict["bboxes"] = bboxes
        result_dict["corefs"] = corefs

        result_list.append(result_dict)
    
    train_data["images"] = result_list
    
    base_name = os.path.basename(file_path)
    if "." in base_name:
        base_name_list = base_name.split(".")
        base_name_list[-2] = base_name_list[-2] + "_fixed"
    else:
        base_name_list = base_name.split(".")
        base_name_list[-1] = base_name_list[-1] + "_fixed"
    
    ## 定义类别
    train_data["categories"] = [{'id': 1, 'name': 'structure'},
                                {'id': 2, 'name': 'text'},
                                {'id': 3, 'name': 'identifier'},
                                {'id': 4, 'name': 'supplement'},
                                {'id': 5, 'name': 'none'},
                                {'id': 6, 'name': 'r-group'},
                                {'id': 7, 'name': 'identifier-text'},
                                {'id': 8, 'name': 'iupac'}]

    save_path = os.path.join(os.path.dirname(file_path),".".join(base_name_list))
    with open(save_path, "w") as f:
        f.write(json.dumps(train_data))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_data_path', default="/mnt/d/clean_data_bak_1219/label_0313/dataset.json",type=str, help="input: coco format(json)")
    args = parser.parse_args()
    standard_data(args.coco_data_path)