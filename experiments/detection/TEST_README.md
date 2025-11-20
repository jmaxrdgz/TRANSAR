# Detection Model Testing

This directory contains a script for testing trained Faster R-CNN detection models on test data.

## Script: `test_detection.py`

Evaluates a trained detection model on the test set, computes quantitative metrics, and generates visualizations of best and worst performing images.

### Features

- **Quantitative Metrics**: Computes same metrics as training/validation
  - mAP@0.5, mAP@0.75, mAP@0.5:0.95
  - F1 Score, Precision, Recall

- **Per-Image Analysis**: Computes metrics for each test image individually

- **Visualization**: Generates comparison plots showing:
  - Ground truth boxes (green solid lines)
  - Predicted boxes (red dashed lines)
  - Confidence scores
  - Per-image metrics

- **Ranking**: Identifies best and worst performing images by multiple metrics:
  - F1 Score
  - mAP@0.5:0.95
  - Average IoU

### Usage

#### Basic Usage

```bash
python experiments/detection/test_detection.py \
  --checkpoint path/to/checkpoint.ckpt \
  --config path/to/config.yaml \
  --output_dir results/test_run_1
```

#### Full Options

```bash
python experiments/detection/test_detection.py \
  --checkpoint logs/detection_experiments/faster_rcnn_detection/version_0/checkpoints/best.ckpt \
  --config experiments/detection/configs/config_experiment.yaml \
  --data_path dataset/supervised/synthetic_yolo \
  --output_dir test_results \
  --batch_size 4 \
  --num_workers 4 \
  --ranking_metric all \
  --num_visualize 5 \
  --device cuda
```

#### Arguments

- `--checkpoint` (required): Path to trained model checkpoint (.ckpt file)
- `--config`: Path to config file (if not specified, tries to find from checkpoint directory)
- `--data_path`: Override dataset path from config
- `--output_dir`: Directory to save results (default: `test_results`)
- `--batch_size`: Batch size for inference (default: 4)
- `--num_workers`: Number of data loading workers (default: 4)
- `--ranking_metric`: Metric for ranking images (choices: `f1`, `map`, `iou`, `all`; default: `all`)
- `--num_visualize`: Number of best/worst images to visualize per metric (default: 5)
- `--device`: Device for inference (default: auto-detect CUDA)

### Output

The script generates the following files in the output directory:

1. **`test_results.json`**: Complete test results including:
   - Overall metrics (mAP, F1, precision, recall)
   - Per-image metrics for all test images

2. **Visualization Images**:
   - `best_f1.png`: Top 5 images by F1 score
   - `worst_f1.png`: Bottom 5 images by F1 score
   - `best_map_50_95.png`: Top 5 images by mAP@0.5:0.95
   - `worst_map_50_95.png`: Bottom 5 images by mAP@0.5:0.95
   - `best_avg_iou.png`: Top 5 images by average IoU
   - `worst_avg_iou.png`: Bottom 5 images by average IoU

### Example Output

```
================================================================================
OVERALL TEST RESULTS
================================================================================
mAP@0.5:        0.7234
mAP@0.75:       0.5123
mAP@0.5:0.95:   0.4567
F1 Score:       0.7891
Precision:      0.8123
Recall:         0.7678
================================================================================
```

### Dataset Requirements

The test script expects the dataset to have a `test` split in YOLO format:

```
dataset/supervised/synthetic_yolo/
├── images/
│   └── test/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
└── labels/
    └── test/
        ├── image1.txt
        ├── image2.txt
        └── ...
```

Each label file contains boxes in YOLO format:
```
class_id x_center y_center width height
```
(normalized coordinates in [0, 1])

### Troubleshooting

**Error: "Test directory not found"**
- Ensure the dataset has a `test` split directory
- Check that `DATA.DATA_PATH` in config points to the correct location

**Error: "Config file not found"**
- Specify config explicitly with `--config` argument
- Ensure the config file exists and is valid YAML

**Out of Memory**
- Reduce `--batch_size` (try 1 or 2)
- Use CPU inference with `--device cpu` (slower but uses less memory)

### Notes

- The script reuses the same metric computation functions as training/validation for consistency
- SAR-specific normalization is applied based on config settings
- Visualizations show the first channel of SAR images in grayscale
- Per-image mAP computation is simplified for individual images (see code for details)
