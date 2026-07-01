from fedn.utils.helpers.helpers import get_helper
from ultralytics import YOLO
import torch
import collections
import numpy as np
import hopsworks
import os
from PIL import Image
import shutil
import zipfile


HELPER_MODULE = "numpyhelper"
helper = get_helper(HELPER_MODULE)

def get_best_device():
    """ Get the best device available.

    :return: The best device available.
    :rtype: str
    """
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return device

def compile_model():
    """Compile the YOLO model.

    :return: The compiled YOLO model.
    :rtype: torch.nn.Module
    """
    device = get_best_device()
    return YOLO("model.yaml").to(device)

def save_parameters(model, out_path):
    """Save model parameters to file.

    :param model: The model to serialize.
    :type model: torch.nn.Module
    :param out_path: The path to save to.
    :type out_path: str
    """
    parameters_np = [
    val.cpu().numpy().astype(np.float32) if val.dtype.is_floating_point else val.cpu().numpy()
    for _, val in model.model.model.state_dict().items()
    ]
    helper.save(parameters_np, out_path)

def load_parameters(model_path):
    """Load model parameters from a file.

    :param model_path: The path to the model parameters file.
    :type model_path: str
    :return: The YOLO model with loaded parameters.
    :rtype: YOLO
    """
    parameters_np = helper.load(model_path)

    yolo_model = compile_model()
    torch_model = yolo_model.model.model

    keys = list(torch_model.state_dict().keys())
    if len(parameters_np) != len(keys):
        raise ValueError(f"Mismatch: {len(parameters_np)} parameters vs {len(keys)} model keys")

    state_dict = collections.OrderedDict({
        key: torch.tensor(val) for key, val in zip(keys, parameters_np)
    })
    torch_model.load_state_dict(state_dict, strict=False)

    yolo_model.ckpt = {"model": torch_model}
    return yolo_model

def copy_to_local_dir_training_data():
    print("Copying training data to local directory.")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(base_dir, "data/widerface.zip")
    dst = "/tmp/widerface.zip"
    shutil.copy2(src, dst)
    with zipfile.ZipFile(dst, 'r') as zip_ref:
        zip_ref.extractall("/tmp")
    print("Finished copying.")


def build_training_data_from_feature_store(fs, out_dir="/tmp/widerface"):
    """Rebuild the YOLO dataset from the current `wider_face_files` feature group.

    Mirrors the layout produced by annotations.py (images and YOLO-format `.txt` labels
    colocated under `<out_dir>/<split>/`), but from the rows currently in the feature
    store so a retrain incorporates newly ingested images rather than a static snapshot.

    :param fs: Hopsworks feature store handle.
    :param out_dir: Output dataset directory referenced by data/widerface.yaml.
    """
    import json
    import cv2

    df = fs.get_feature_group("wider_face_files", version=1).read()
    for split in ("train", "val"):
        os.makedirs(os.path.join(out_dir, split), exist_ok=True)

    written = 0
    for _, row in df.iterrows():
        img_path = row["file_path"]
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        height, width = img.shape[:2]

        dst_dir = os.path.join(out_dir, row["label"])
        shutil.copy2(img_path, dst_dir)

        labels = []
        for bbox in json.loads(row["bboxes"]):
            bbox = [max(0, int(v)) for v in bbox]
            x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
            if x > width or y > height or w > width or h > height:
                continue
            cx = (x + w / 2) / width
            cy = (y + h / 2) / height
            labels.append(f"0 {cx} {cy} {w / width} {h / height}")

        base = os.path.splitext(os.path.basename(img_path))[0]
        with open(os.path.join(dst_dir, base + ".txt"), "w") as f:
            f.write("\n".join(labels) + ("\n" if labels else ""))
        written += 1

    print(f"Built YOLO dataset from feature store: {written} images into {out_dir}")
    return out_dir


if __name__ == '__main__':
    import sys

    # "retrain" is passed as the execution argument by the drift-triggered model-monitoring
    # job (model_retraining_job_execution_args="retrain"); a bare run defaults to "train".
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    print(f"Running YOLO training job in '{mode}' mode")

    project = hopsworks.login()
    mr = project.get_model_registry()
    fs = project.get_feature_store()

    # Prepare the training data. A retrain rebuilds the dataset from the current
    # `wider_face_files` feature group so newly ingested images are included; a bare
    # train run uses the prepared snapshot zip.
    if mode == "retrain":
        build_training_data_from_feature_store(fs)
    else:
        copy_to_local_dir_training_data()

    try:
        num_images = fs.get_feature_group("wider_face_files", version=1).read().shape[0]
        print(f"Current wider_face_files size: {num_images} rows")
    except Exception as e:
        num_images = None
        print(f"Could not read wider_face_files: {e}")

    model = load_parameters("weights/face_finder_best.npz")
    params = {
        'data': os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/widerface.yaml"),
        'epochs': 1,
        'batch': 32,
        'imgsz': 640,
        'device': 0,
        'resume': False,
        'workers': 0,
        'cache': "ram",
        'amp': True,
    }

    print(params)

    model.train(**params)


    model_dir = "mr_model"
    os.makedirs(f"{model_dir}/images", exist_ok=True)
    save_parameters(model, f"./{model_dir}/fine-tuned-model.npz")

    img_path = "data/images/bus.jpg"
    results = model.predict(
        img_path,
        imgsz=640,
        conf=0.75,
        iou=0.7,
        device=0,
        verbose=False
    )
    
    img = results[0].plot()  # BGR numpy array
    img = Image.fromarray(img[..., ::-1])  # Convert to RGB for PIL
    
    base, _ = os.path.splitext(os.path.basename(img_path))
    output_filename = f"./{model_dir}/images/{base}-faces-detected.png"
    output_path = os.path.abspath(output_filename)
    img.save(output_path, format="PNG")
    
    metrics = {
        "epochs": params['epochs'],
        "batch": params['batch'],
    }
    if num_images is not None:
        metrics["num_images"] = num_images

    faces_model = mr.python.create_model(
        name="facerecognition",
        metrics=metrics,
        description=f"Yolo-v8 face recognition model ({mode})",
    )

    # Save the model to the specified directory
    faces_model.save(model_dir)
    print(f"Registered facerecognition v{faces_model.version} ({mode} mode)")
