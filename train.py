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


if __name__ == '__main__':
    copy_to_local_dir_training_data()
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
    
    
    mr = hopsworks.login().get_model_registry()
    
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
    
    faces_model = mr.python.create_model(
        name="facerecognition", 
        metrics=metrics,
        description="Yolo-v8 face recognition model", 
    )
    
    # Save the model to the specified directory
    faces_model.save(model_dir)
