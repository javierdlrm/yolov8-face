# YOLOv8-Face

The **YOLOv8-Face** repository provides pre-trained models designed specifically for face detection. The models have
been pre-trained by Lindevs from scratch.


## Pre-trained Models

The models have been trained on [WIDERFace](http://shuoyang1213.me/WIDERFACE/) dataset using NVIDIA RTX 4090.
[YOLOv8 models](https://github.com/ultralytics/ultralytics#models) were used as initial weights for training.

| Name         | Image Size<br>(pixels) | mAP<sup>val<br>50-95 | Params   | GFLOPs |
|--------------|------------------------|----------------------|----------|--------|
| YOLOv8n-Face | 640                    | 37.5                 | 3005843  | 8.1    |
| YOLOv8s-Face | 640                    | 40.6                 | 11125971 | 28.4   |
| YOLOv8m-Face | 640                    | 41.7                 | 25840339 | 78.7   |
| YOLOv8l-Face | 640                    | 42.8                 | 43607379 | 164.8  |
| YOLOv8x-Face | 640                    | 43.3                 | 68124531 | 257.4  |


* Training results:

| Name         | Training Time | Epochs | Batch Size | Non-default<br>parameters                        | Link                                                  |
|--------------|---------------|--------|------------|--------------------------------------------------|-------------------------------------------------------|
| YOLOv8n-Face | 2.75 hours    | 300    | 16         | -                                                | [results.txt](results/train/yolov8n-face/results.txt) |
| YOLOv8s-Face | 2.68 hours    | 200    | 16         | -                                                | [results.txt](results/train/yolov8s-face/results.txt) |
| YOLOv8m-Face | 3.01 hours    | 120    | 16         | -                                                | [results.txt](results/train/yolov8m-face/results.txt) |
| YOLOv8l-Face | 3.97 hours    | 110    | 16         | -                                                | [results.txt](results/train/yolov8l-face/results.txt) |
| YOLOv8x-Face | 13.65 hours   | 240    | 16         | optimizer='SGD'<br>lrf=1e-5<br>weight_decay=5e-3 | [results.txt](results/train/yolov8x-face/results.txt) |

* Evaluation results on WIDERFace dataset:

| Name         | Easy  | Medium | Hard  |
|--------------|-------|--------|-------|
| YOLOv8n-Face | 93.79 | 91.82  | 79.38 |
| YOLOv8s-Face | 95.13 | 93.62  | 82.90 |
| YOLOv8m-Face | 95.73 | 94.47  | 84.55 |
| YOLOv8l-Face | 96.26 | 95.03  | 85.43 |
| YOLOv8x-Face | 96.33 | 95.16  | 85.80 |

# Instructions


## Training in Notebooks

Training runs as the `train_yolo` platform job (1 CPU, 10000 MB, 1 GPU, environment `yolov8`),
registered and launched from the notebooks. The Jupyter server itself needs no GPU. This folder
must be inside the project filesystem (for example cloned into the `Jupyter` dataset): it becomes
the job's application path.

- To try the pretrained weights locally, create a console in Jupyter and run the following

```shell
    bash
    cd yolov8-face
    python predictsave.py --weights weights/yolov8n-face-lindevs.pt --source data/images/bus.jpg
```

Now check in the 'results' directory for your bus png file with detected faces

***

### Run notebooks in the following order

Create feature groups for all training images
```shell
    1-create-feature-groups.ipynb
```

Register the `train_yolo` job and run it to fine-tune the pretrained model. The same job is
reused later for drift-triggered retraining.
```shell
    2-fine-tune.ipynb
```

Now check in the model registry for your trained model


## Training from your IDE

You need to set the following environment variables

* HOPSWORKS_HOST=saab.dev-cloud.hopsworks.ai
* HOPSWORKS_PROJECT=groupX
* HOPSWORKS_API_KEY=

```shell
python run_job.py create [drop]
python run_job.py train [drop]
python run_job.py retrain [drop]
```

`create` registers and runs the `create_fgs` job, `train` fine-tunes from the prepared snapshot,
`retrain` rebuilds the training data from the `wider_face_files` feature group first. `drop`
re-registers the job configuration before running.

### Extra notebooks 

Exploratory data analysis of the images/bounding-boxes in feature groups.
```shell
    3-eda-fgs.ipynb
```

Build vector index for similarity search in a feature group (requires a GPU). Takes a few mins.
```shell
    4-similarity-search-index.ipynb
```

UI with gradio to do similarity search:
```shell
    5-similarity-search-gradio.ipynb
```

Feature/model monitoring on the embedding feature with drift-triggered retraining:
```shell
    6-monitoring-and-retraining.ipynb
```
This notebook adds feature monitoring on the CLIP `embedding` (centroid distance + norm PSI) and on
`num_bboxes`. It then creates a logging-enabled feature view over the embeddings, runs `train_yolo`
in feature-store-driven mode to register a `facerecognition` version linked to the feature view and
a baseline training dataset, redeploys the similarity service with a predictor that logs query
embeddings under that version, and configures model monitoring that re-runs `train_yolo` after
consecutive embedding-drift shifts.

# This has already been done for your projects

## Installation

```shell
pip install -r requirements.txt
```


## Dataset Preparation

* Download WIDERFace dataset and annotations:

```shell
python download.py
```

* Convert annotations to YOLO format:

```shell
python annotations.py
```


