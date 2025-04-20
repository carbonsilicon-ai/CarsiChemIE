## file
```bash
├── data                # Directory containing data files
│   ├── img            # Subdirectory for images
│   ├── label          # Subdirectory for label files
│   └── test           # Subdirectory for predction data
├── readme.md          # Markdown file containing project documentation
└── td_test.py         # Python script for testing the model
```

## Evalation
```bash
python your_script_name.py --annFile path/to/your/annotations.json --resFile path/to/your/results.json --category_idx 0
```
## result

| Model | Precision@50% | Recall@50% | Precision@75% | Recall@75% | Precision@90% | Recall@90% |
|--------------|-----------|--------|-----------|--------|-----------|--------|
| DETR  | 0.807     | 0.818  | 0.665     | 0.712  | 0.112     | 0.246  |
| yolo9c(Ours)  | **0.961**     | **0.996**  | **0.926**     | **0.975**  | **0.762**     | **0.881**  |