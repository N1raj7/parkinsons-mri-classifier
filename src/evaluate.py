"""
Full evaluation: accuracy, precision, recall, F1, ROC-AUC, confusion matrix.
Run after train.py: python src/evaluate.py --model model.h5
"""

import argparse

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_curve, auc, ConfusionMatrixDisplay)

from dataset import make_dataset


def load_test_split(path="test_split.txt"):
    filepaths, labels = [], []
    with open(path) as f:
        for line in f:
            fp, lb = line.strip().rsplit(",", 1)
            filepaths.append(fp)
            labels.append(int(lb))
    return filepaths, labels


def main(args):
    filepaths, labels = load_test_split(args.test_split)
    test_ds = make_dataset(filepaths, labels, augment=False, shuffle=False)

    model = tf.keras.models.load_model(args.model)

    y_true = np.array(labels)
    y_prob = model.predict(test_ds).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Parkinson's"]))

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Parkinson's"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix — Test Set")
    plt.savefig("confusion_matrix.png", bbox_inches="tight")
    print("Saved confusion_matrix.png")

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig("roc_curve.png", bbox_inches="tight")
    print("Saved roc_curve.png")
    print(f"\nFinal ROC-AUC: {roc_auc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="model.h5")
    parser.add_argument("--test_split", type=str, default="test_split.txt")
    args = parser.parse_args()
    main(args)
