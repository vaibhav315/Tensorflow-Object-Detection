#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow Object Detection Project (Kaggle: andrewmvd/face-mask-detection)

What this script does
---------------------
1) Downloads the Kaggle dataset "Face Mask Detection" (andrewmvd/face-mask-detection)
2) Unzips and parses Pascal VOC XML annotations into a clean DataFrame
3) Builds a tf.data pipeline that yields images + bounding boxes for KerasCV
4) Trains a RetinaNet detector (TensorFlow + KerasCV) on the dataset
5) Evaluates and runs inference, saving sample predictions to disk

Why this dataset?
-----------------
- Public Kaggle dataset with 3 classes: 'with_mask', 'without_mask', 'mask_weared_incorrect'
- VOC XML annotations are easy to parse
- Great small-scale demo for end-to-end object detection

Environment & Dependencies
--------------------------
- Python 3.10+ recommended
- TensorFlow 2.12+ (or newer 2.x)
- KerasCV >= 0.6.0
- Kaggle CLI (for dataset download)
- lxml, pandas, Pillow, tqdm

Install (example):
------------------
pip install --upgrade pip
pip install tensorflow keras-cv pandas lxml pillow tqdm matplotlib
pip install kaggle

# Configure Kaggle API (only once):
# 1) Go to https://www.kaggle.com/ -> Account -> Create New API Token
# 2) This downloads kaggle.json. Place it at: ~/.kaggle/kaggle.json (Linux/Mac)
#    or %USERPROFILE%\\.kaggle\\kaggle.json (Windows)
# 3) Ensure permissions: chmod 600 ~/.kaggle/kaggle.json

Run:
----
python tf_object_detection_kaggle.py --epochs 10 --batch 4 --img-size 512 --limit 0

Arguments:
----------
--data-dir   : Where to store data (default: ./data/face-mask-detection)
--out-dir    : Where to store outputs (default: ./runs)
--epochs     : Training epochs (default: 10)
--batch      : Batch size (default: 4)
--img-size   : Square image size for resizing/padding (default: 512)
--lr         : Learning rate (default: 0.001)
--limit      : Optional: limit number of training images for a quick demo (0 means all)
--val-split  : Validation split fraction (default: 0.2)
--seed       : Random seed (default: 1337)

Notes:
------
- KerasCV RetinaNet expects a dict with keys {"images", "bounding_boxes"}.
- "bounding_boxes" itself is a dict: {"boxes": Tensor[N, 4], "classes": Tensor[N]}
- Set bounding_box_format consistently (we use "xywh" in *absolute pixel* units).
- This script includes robust parsing, visualization, and simple training/eval loops.
"""

import argparse
import os
import random
import shutil
import subprocess
import sys
import zipfile
from glob import glob
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from lxml import etree
from tqdm import tqdm

import tensorflow as tf

# KerasCV is built on top of Keras (TF 2.x). We use its RetinaNet implementation.
import keras_cv
from tensorflow import keras

# ---------------------------
# Utility
# ---------------------------

CLASSES = ["with_mask", "without_mask", "mask_weared_incorrect"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def run_cmd(cmd: List[str]):
    print(f"[cmd] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

# ---------------------------
# Kaggle Download
# ---------------------------

def download_kaggle_dataset(data_dir: Path):
    \"\"\"Download and unzip the Kaggle dataset if not already present.\"\"\"
    images_dir = data_dir / "images"
    annotations_dir = data_dir / "annotations"
    if images_dir.exists() and annotations_dir.exists():
        print(\"Dataset already present. Skipping download.\")
        return

    ensure_dir(data_dir)
    zip_path = data_dir / "face-mask-detection.zip"

    # Requires Kaggle CLI + kaggle.json configured
    print(\"Downloading Kaggle dataset andrewmvd/face-mask-detection ...\")
    run_cmd([sys.executable, \"-m\", \"kaggle\", \"datasets\", \"download\",
             \"-d\", \"andrewmvd/face-mask-detection\", \"-p\", str(data_dir)])

    print(\"Unzipping ...\")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(data_dir)

    # Dataset structure after unzip:
    # - annotations/: Pascal VOC XML files
    # - images/: JPEGs
    if not (data_dir / \"images\").exists() or not (data_dir / \"annotations\").exists():
        # Some Kaggle datasets unzip into nested folder; try to locate.
        print(\"Trying to locate 'images' and 'annotations'...\" )
        for root, dirs, files in os.walk(data_dir):
            if \"images\" in dirs and \"annotations\" in dirs:
                print(f\"Found nested dataset at: {root}\")
                nested = Path(root)
                shutil.move(str(nested/\"images\"), str(data_dir/\"images\"))
                shutil.move(str(nested/\"annotations\"), str(data_dir/\"annotations\"))
                break

    assert (data_dir / \"images\").exists(), \"images/ folder not found after unzip\"
    assert (data_dir / \"annotations\").exists(), \"annotations/ folder not found after unzip\"
    print(\"Download + unzip complete.\")

# ---------------------------
# Parse VOC XML to DataFrame
# ---------------------------

def parse_voc_xml(xml_path: Path) -> pd.DataFrame:
    \"\"\"Parse a single Pascal VOC XML into a DataFrame with one row per box.\"\"\"
    tree = etree.parse(str(xml_path))
    root = tree.getroot()

    file_name = root.findtext(\"filename\")
    size = root.find(\"size\")
    width = int(size.findtext(\"width\"))
    height = int(size.findtext(\"height\"))

    rows = []
    for obj in root.findall(\"object\"):
        cls = obj.findtext(\"name\")
        if cls not in CLASS_TO_ID:
            # Skip unknown classes, if any
            continue
        bnd = obj.find(\"bndbox\")
        xmin = int(float(bnd.findtext(\"xmin\")))
        ymin = int(float(bnd.findtext(\"ymin\")))
        xmax = int(float(bnd.findtext(\"xmax\")))
        ymax = int(float(bnd.findtext(\"ymax\")))

        w = max(0, xmax - xmin)
        h = max(0, ymax - ymin)
        if w <= 1 or h <= 1:
            # filter tiny/invalid boxes
            continue

        rows.append({
            \"image\": file_name,
            \"width\": width,
            \"height\": height,
            \"class\": cls,
            \"class_id\": CLASS_TO_ID[cls],
            \"xmin\": xmin,
            \"ymin\": ymin,
            \"xmax\": xmax,
            \"ymax\": ymax,
            \"x\": xmin,
            \"y\": ymin,
            \"w\": w,
            \"h\": h,
        })

    return pd.DataFrame(rows)

def build_annotations_df(ann_dir: Path) -> pd.DataFrame:
    xmls = sorted(glob(str(ann_dir / \"*.xml\")))
    dfs = []
    for x in tqdm(xmls, desc=\"Parsing VOC annotations\"):
        df = parse_voc_xml(Path(x))
        if len(df):
            dfs.append(df)
    assert len(dfs) > 0, \"No valid annotations parsed.\"
    full = pd.concat(dfs, ignore_index=True)
    return full

# ---------------------------
# Train/Val split
# ---------------------------

def train_val_split(df: pd.DataFrame, val_split: float = 0.2, seed: int = 1337) -> Tuple[pd.DataFrame, pd.DataFrame]:
    images = df[\"image\"].unique().tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(images)
    n_val = max(1, int(len(images) * val_split))
    val_images = set(images[:n_val])
    train_df = df[~df[\"image\"].isin(val_images)].copy()
    val_df = df[df[\"image\"].isin(val_images)].copy()
    return train_df, val_df

# ---------------------------
# Dataset pipeline helpers
# ---------------------------

def load_image(path: tf.Tensor) -> tf.Tensor:
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    return img

def resize_and_pad_image(img: tf.Tensor, boxes: tf.Tensor, target_size: int) -> Tuple[tf.Tensor, tf.Tensor]:
    \"\"\"Resize the image keeping aspect ratio, then pad to square; adjust boxes (xywh ABS).\"\"\"
    h = tf.cast(tf.shape(img)[0], tf.float32)
    w = tf.cast(tf.shape(img)[1], tf.float32)

    scale = tf.minimum(target_size / h, target_size / w)
    nh = tf.cast(tf.round(h * scale), tf.int32)
    nw = tf.cast(tf.round(w * scale), tf.int32)
    img = tf.image.resize(img, (nh, nw))

    # pad to square
    pad_h = target_size - nh
    pad_w = target_size - nw
    img = tf.image.pad_to_bounding_box(img, 0, 0, target_size, target_size)

    # boxes are xywh in absolute pixels; scale and no offset since padded at top-left
    scale_vec = tf.stack([scale, scale, scale, scale])
    boxes = boxes * scale_vec
    return img, boxes

def df_to_grouped(df: pd.DataFrame, images_dir: Path) -> List[Dict]:
    \"\"\"Group annotations by image and convert to a list of records.\"\"\"
    grouped = []
    for img_name, sub in df.groupby(\"image\"):
        boxes = sub[[\"x\", \"y\", \"w\", \"h\"]].values.astype(\"float32\")
        classes = sub[\"class_id\"].values.astype(\"int32\")
        grouped.append({
            \"path\": str(images_dir / img_name),
            \"boxes\": boxes,
            \"classes\": classes
        })
    return grouped

def make_tf_dataset(records: List[Dict], img_size: int, batch_size: int, shuffle: bool, augment: bool) -> tf.data.Dataset:
    box_format = \"xywh\"  # absolute pixels

    def gen():
        for r in records:
            yield r[\"path\"], r[\"boxes\"], r[\"classes\"]

    def _load(path, boxes, classes):
        img = load_image(path)
        img, boxes = resize_and_pad_image(img, boxes, img_size)
        img = tf.cast(img, tf.float32) / 255.0
        # Pack into KerasCV expected dict
        sample = {
            \"images\": img,
            \"bounding_boxes\": {
                \"boxes\": boxes,
                \"classes\": classes,
                \"bounding_box_format\": box_format,
            }
        }
        return sample

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
        )
    )
    if shuffle:
        ds = ds.shuffle(1024, reshuffle_each_iteration=True)
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        # Simple horizontal flip augmentation using KerasCV layer
        ds = ds.map(
            keras_cv.layers.RandomFlip(mode=\"horizontal\", bounding_box_format=\"xywh\"),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    ds = ds.padded_batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# ---------------------------
# Visualization
# ---------------------------

def draw_boxes_on_image(image_pil: Image.Image, boxes_xywh: np.ndarray, classes: np.ndarray, save_path: Path):
    draw = ImageDraw.Draw(image_pil)
    try:
        font = ImageFont.truetype(\"DejaVuSans.ttf\", 16)
    except Exception:
        font = ImageFont.load_default()
    for box, cls in zip(boxes_xywh, classes):
        x, y, w, h = box
        x2, y2 = x + w, y + h
        draw.rectangle([x, y, x2, y2], outline=(255, 0, 0), width=2)
        label = CLASSES[int(cls)]
        draw.text((x + 3, y + 3), label, fill=(255, 0, 0), font=font)
    image_pil.save(save_path)

# ---------------------------
# Build / Train Model
# ---------------------------

def build_model(num_classes: int, img_size: int) -> keras.Model:
    # KerasCV RetinaNet: anchor-based detector
    # Using ResNet50 backbone; include_rescaling=False since we already scale images to [0,1]
    model = keras_cv.models.RetinaNet(
        classes=num_classes,
        bounding_box_format=\"xywh\",
        backbone=\"resnet50\",
        include_rescaling=False,
    )
    # Compile with standard losses/optimizer; KerasCV provides built-ins internally.
    optimizer = keras.optimizers.SGD(learning_rate=args.lr, momentum=0.9, global_clipnorm=10.0)
    model.compile(
        optimizer=optimizer,
        classification_loss=\"focal\",
        box_loss=\"smoothl1\"
    )
    return model

# ---------------------------
# Main
# ---------------------------

def main(args):
    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    # 1) Download dataset
    download_kaggle_dataset(data_dir)

    images_dir = data_dir / \"images\"
    ann_dir = data_dir / \"annotations\"

    # 2) Build annotations dataframe
    df = build_annotations_df(ann_dir)

    # Optional: limit dataset for quick demo
    if args.limit and args.limit > 0:
        keep_imgs = set(df[\"image\"].unique()[:args.limit])
        df = df[df[\"image\"].isin(keep_imgs)].copy()

    # 3) Train/Val split
    train_df, val_df = train_val_split(df, val_split=args.val_split, seed=args.seed)
    print(f\"Train images: {train_df['image'].nunique()} | Val images: {val_df['image'].nunique()}\" )

    # 4) Build tf.data datasets
    train_recs = df_to_grouped(train_df, images_dir)
    val_recs = df_to_grouped(val_df, images_dir)

    train_ds = make_tf_dataset(train_recs, args.img_size, args.batch, shuffle=True, augment=True)
    val_ds = make_tf_dataset(val_recs, args.img_size, args.batch, shuffle=False, augment=False)

    # 5) Visual sanity-check: save 3 samples with boxes
    vis_dir = out_dir / \"viz_gt\"
    ensure_dir(vis_dir)
    for i, rec in enumerate(train_recs[:3]):
        img = Image.open(rec[\"path\"]).convert(\"RGB\")
        draw_boxes_on_image(img.copy(), np.array(rec[\"boxes\"]), np.array(rec[\"classes\"]), vis_dir / f\"gt_{i}.jpg\")

    # 6) Build model
    model = build_model(num_classes=len(CLASSES), img_size=args.img_size)
    model.summary()

    # 7) Train
    ckpt_path = out_dir / \"checkpoints\" / \"retinanet.weights.h5\"
    ensure_dir(ckpt_path.parent)
    callbacks = [
        keras.callbacks.ModelCheckpoint(str(ckpt_path), save_weights_only=True, save_best_only=True, monitor=\"val_loss\"),
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor=\"val_loss\"),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks
    )

    # 8) Evaluate (simple: use val_loss) and run inference on a few images
    pred_dir = out_dir / \"predictions\"
    ensure_dir(pred_dir)

    # KerasCV RetinaNet predict() returns bounding boxes in the configured format.
    # We'll run inference per-image for nicer visualization.
    for i, rec in enumerate(val_recs[:10]):
        img = Image.open(rec[\"path\"]).convert(\"RGB\")
        img_np = np.array(img)
        # Prepare single image similarly to the dataset pipeline
        img_tf = tf.convert_to_tensor(img_np)
        img_tf, boxes_tf = resize_and_pad_image(img_tf, tf.convert_to_tensor(rec[\"boxes\"], tf.float32), args.img_size)
        img_tf = tf.cast(img_tf, tf.float32) / 255.0
        sample = {
            \"images\": tf.expand_dims(img_tf, 0),
            \"bounding_boxes\": {
                \"boxes\": tf.expand_dims(boxes_tf, 0),
                \"classes\": tf.expand_dims(tf.convert_to_tensor(rec[\"classes\"], tf.int32), 0),
                \"bounding_box_format\": \"xywh\",
            }
        }
        # Model inference
        preds = model.predict(sample[\"images\"], verbose=0)
        # preds is a dict with 'boxes', 'classes', 'confidence' as a RaggedTensor per image
        # KerasCV standardizes the output; handle ragged to numpy
        boxes = preds[\"boxes\"][0].numpy()
        classes = preds[\"classes\"][0].numpy().astype(int)
        conf = preds[\"confidence\"][0].numpy()

        # Filter low-confidence
        keep = conf >= 0.3
        boxes = boxes[keep]
        classes = classes[keep]

        # Draw and save
        out_img = img.copy()
        draw_boxes_on_image(out_img, boxes, classes, pred_dir / f\"pred_{i}.jpg\")

    print(\"Done. Artifacts saved to:\", out_dir)


if __name__ == \"__main__\":
    parser = argparse.ArgumentParser()
    parser.add_argument(\"--data-dir\", type=str, default=\"./data/face-mask-detection\")
    parser.add_argument(\"--out-dir\", type=str, default=\"./runs\")
    parser.add_argument(\"--epochs\", type=int, default=10)
    parser.add_argument(\"--batch\", type=int, default=4)
    parser.add_argument(\"--img-size\", type=int, default=512)
    parser.add_argument(\"--lr\", type=float, default=1e-3)
    parser.add_argument(\"--limit\", type=int, default=0, help=\"Limit number of images (0 = use all)\" )
    parser.add_argument(\"--val-split\", type=float, default=0.2)
    parser.add_argument(\"--seed\", type=int, default=1337)
    args = parser.parse_args()

    # Basic CUDA/CPU info
    print(\"TensorFlow version:\", tf.__version__)
    print(\"KerasCV version:\", keras_cv.__version__)
    print(\"Num GPUs available:\", len(tf.config.experimental.list_physical_devices('GPU')))

    main(args)
