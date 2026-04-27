from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


def evaluate(y_true, y_pred):
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print()
    print("Classification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    classes = np.unique(y_true)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()

    plt.xticks(range(len(classes)), classes)
    plt.yticks(range(len(classes)), classes)

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()