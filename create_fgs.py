import os
import json
import pandas as pd
from datetime import datetime
import hopsworks
from hsfs.statistics_config import StatisticsConfig

def parse_annotation_file(annotation_file, label, base_dir_images=""):
    """
    Parse a WIDER Face-style annotation file.

    Returns a DataFrame with columns:
      - file_path (str): absolute path to image file
      - file_timestamp (datetime or None)
      - file_size_mb (float or None)
      - num_bboxes (int)
      - bboxes (str): JSON-encoded list[list[int]]
      - label (str): 'train' | 'val'
      - ingested_at (datetime)
    """
    rows = []
    with open(annotation_file, "r") as f:
        # Keep only non-empty lines
        lines = [line.strip() for line in f if line.strip()]

    i = 0
    now = datetime.now()
    while i < len(lines):
        filename = lines[i]
        num_bboxes = int(lines[i + 1])
        bbox_lines = lines[i + 2 : i + 2 + num_bboxes]

        # Convert bbox lines into list[list[int]]
        bboxes_list = [list(map(int, l.split())) for l in bbox_lines]

        # Build full absolute path for the image file
        # If filename is already absolute, abspath will keep it.
        full_path = os.path.abspath(os.path.join(base_dir_images, filename))

        # File metadata (graceful if file doesn't exist on disk)
        if os.path.exists(full_path):
            size_mb = os.path.getsize(full_path) / (1024 * 1024)
            ts = datetime.fromtimestamp(os.path.getmtime(full_path))
        else:
            size_mb = None
            ts = None

        rows.append(
            {
                "file_path": full_path,
                "file_timestamp": ts,
                "file_size_mb": size_mb,
                "num_bboxes": num_bboxes,
                "bboxes": json.dumps(bboxes_list), # Store as JSON string 
                "label": label,
                "ingested_at": now,
            }
        )

        i += 2 + num_bboxes

    return pd.DataFrame(rows)

def build_dataset_and_write_to_hopsworks(
    df_train,
    df_val,
    feature_group_name="wider_face_files",
    feature_group_version=1,
    primary_key=["file_path"],
    event_time="file_timestamp",
    description="WIDER Face annotations with file metadata and bounding boxes (JSON).",
):
    """
    Parse train/val files, combine, and write to a Hopsworks Feature Group.
    """

    df = pd.concat([df_train, df_val], ignore_index=True)

    project = hopsworks.login()
    fs = project.get_feature_store()

    fg = fs.get_or_create_feature_group(
        name=feature_group_name,
        version=feature_group_version,
        primary_key=primary_key,
        event_time=event_time,
        description=description,
        online_enabled=False,
        # kll=True persists per-commit KLL sketches, required by the rolling-window
        # distribution monitoring configured in notebook 6.
        statistics_config=StatisticsConfig(kll=True),
    )

    print(f"Inserting {len(df)} rows into feature group '{feature_group_name}' v{feature_group_version}...")
    fg.insert(df)
    return df, fg



if __name__ == '__main__':
    # Resolve paths relative to this script's directory so it runs from any container folder
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Source annotation files
    TRAIN_FILE = os.path.join(BASE_DIR, "data/wider_face_split/wider_face_train_bbx_gt.txt")
    VAL_FILE   = os.path.join(BASE_DIR, "data/wider_face_split/wider_face_val_bbx_gt.txt")

    # Base directory where the actual image files live (so we can compute size/timestamp).
    # If the paths in the files are already absolute, leave this empty string.
    TRAIN_DIR_IMAGES = os.path.join(BASE_DIR, "data/WIDER_train/images")
    VAL_DIR_IMAGES = os.path.join(BASE_DIR, "data/WIDER_val/images")
    
    df_train = parse_annotation_file(TRAIN_FILE, label="train", base_dir_images=TRAIN_DIR_IMAGES)
    df_val = parse_annotation_file(VAL_FILE, label="val", base_dir_images=VAL_DIR_IMAGES)
    
    df, fg = build_dataset_and_write_to_hopsworks(
        df_train,
        df_val
    )
    
    print(f"Wrote {len(df)} rows to Feature Group: {fg.name}, v{fg.version}")
    print(df.head())

    