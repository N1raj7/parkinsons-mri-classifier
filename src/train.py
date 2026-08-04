"""
Training entrypoint. Run: python src/train.py --data_dir data --epochs 25
"""

import argparse
import os

import tensorflow as tf

from dataset import (build_balanced_filelist, stratified_split,
                      make_dataset, get_class_weights)
from model import build_transfer_model, compile_model


def main(args):
    filepaths, labels = build_balanced_filelist(args.data_dir, balance=True)
    (Xtr, ytr), (Xv, yv), (Xte, yte) = stratified_split(filepaths, labels)

    train_ds = make_dataset(Xtr, ytr, augment=True, shuffle=True)
    val_ds = make_dataset(Xv, yv, augment=False, shuffle=False)
    test_ds = make_dataset(Xte, yte, augment=False, shuffle=False)

    class_weights = get_class_weights(ytr)
    print("Class weights:", class_weights)

    model, base = build_transfer_model(fine_tune_at=args.fine_tune_at)
    model = compile_model(model, lr=args.lr)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=6, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            args.output, monitor="val_auc", mode="max", save_best_only=True
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    print("\nEvaluating on held-out test set:")
    results = model.evaluate(test_ds, return_dict=True)
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    # Save test split filenames/labels for evaluate.py / gradcam.py reuse
    with open("test_split.txt", "w") as f:
        for fp, lb in zip(Xte, yte):
            f.write(f"{fp},{lb}\n")

    print(f"\nBest model saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--fine_tune_at", type=int, default=100,
                         help="Layer index below which EfficientNetB0 stays frozen")
    parser.add_argument("--output", type=str, default="model.h5")
    args = parser.parse_args()
    main(args)
