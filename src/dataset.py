"""
Data loading, class balancing, and stratified splitting for
Parkinson's MRI classification.
"""

import os
import random
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42


def build_balanced_filelist(data_dir, balance=True, max_per_class=None):
    """
    Reads data_dir/normal and data_dir/parkinson, returns (filepaths, labels).
    If balance=True, undersamples the majority class to match the minority
    class (mirrors the balanced 221/221-style split used in published work
    on this dataset).
    """
    data_dir = Path(data_dir)
    normal_files = list((data_dir / "normal").glob("*.*"))
    pd_files = list((data_dir / "parkinson").glob("*.*"))

    random.seed(SEED)
    random.shuffle(normal_files)
    random.shuffle(pd_files)

    if balance:
        n = min(len(normal_files), len(pd_files))
        if max_per_class:
            n = min(n, max_per_class)
        normal_files = normal_files[:n]
        pd_files = pd_files[:n]

    filepaths = [str(f) for f in normal_files] + [str(f) for f in pd_files]
    labels = [0] * len(normal_files) + [1] * len(pd_files)

    print(f"Normal: {len(normal_files)} | Parkinson: {len(pd_files)} | Total: {len(filepaths)}")
    return filepaths, labels


def stratified_split(filepaths, labels, val_size=0.15, test_size=0.15):
    """80/... split done as: first carve out test, then val from remainder."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        filepaths, labels, test_size=(val_size + test_size),
        stratify=labels, random_state=SEED
    )
    relative_test = test_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test,
        stratify=y_temp, random_state=SEED
    )
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def _load_and_preprocess(filepath, label, augment=False):
    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)

    if augment:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=0.1)
        img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
        # small random rotation via image ops
        img = tf.image.rot90(img, k=tf.random.uniform([], 0, 1, dtype=tf.int32) * 0)

    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img, label


def make_dataset(filepaths, labels, augment=False, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(filepaths), seed=SEED)
    ds = ds.map(
        lambda fp, lb: _load_and_preprocess(fp, lb, augment=augment),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


def get_class_weights(labels):
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return dict(zip(classes, weights))


if __name__ == "__main__":
    # Quick sanity check when run directly
    DATA_DIR = "data"
    fps, lbs = build_balanced_filelist(DATA_DIR, balance=True)
    (Xtr, ytr), (Xv, yv), (Xte, yte) = stratified_split(fps, lbs)
    print("Class weights:", get_class_weights(ytr))
