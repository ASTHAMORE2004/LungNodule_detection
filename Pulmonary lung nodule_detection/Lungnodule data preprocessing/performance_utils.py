# performance_utils.py
import matplotlib.pyplot as plt
import itertools
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# ─────────────────────────────────────────────────────────
# 1.  PLOT TRAINING CURVES
# ─────────────────────────────────────────────────────────
def model_performance(history, save_path: str | None = None):
    """Plot training / validation loss and accuracy.

    Args:
        history: Keras History object from model.fit().
        save_path: Optional filepath to save the figure (PNG). If None, just show.
    """
    acc      = history.history['accuracy']
    val_acc  = history.history['val_accuracy']
    loss     = history.history['loss']
    val_loss = history.history['val_loss']
    epochs   = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 6))

    # Loss curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss,     'r', label='Training Loss')
    plt.plot(epochs, val_loss, 'g', label='Validation Loss')
    plt.title("Loss");  plt.legend();  plt.xlabel("Epoch")

    # Accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc,     'r', label='Training Acc')
    plt.plot(epochs, val_acc, 'g', label='Validation Acc')
    plt.title("Accuracy");  plt.legend();  plt.xlabel("Epoch")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()
    plt.close()


# ─────────────────────────────────────────────────────────
# 2.  EVALUATE MODEL ON 3 DATA SPLITS
# ─────────────────────────────────────────────────────────
def evaluate_model(model, train_gen, valid_gen, test_gen):
    """Print loss & accuracy on train / val / test generators."""
    print("Train  :", model.evaluate(train_gen , verbose=0))
    print("Val    :", model.evaluate(valid_gen, verbose=0))
    print("Test   :", model.evaluate(test_gen , verbose=0))
    print('-'*40)


# ─────────────────────────────────────────────────────────
# 3.  GET NUMERICAL PREDICTIONS
# ─────────────────────────────────────────────────────────
def get_predictions(model, test_gen):
    """Return class indices predicted for the entire test generator."""
    preds = model.predict(test_gen, verbose=0)
    return np.argmax(preds, axis=1)


# ─────────────────────────────────────────────────────────
# 4.  PLOT CONFUSION MATRIX
# ─────────────────────────────────────────────────────────
def plot_conf_matrix(y_true, y_pred, class_names, save_path: str | None = None):
    """Plot and optionally save a confusion‑matrix heat‑map."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()

    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45)
    plt.yticks(ticks, class_names)

    thresh = cm.max() / 2
    for i, j in itertools.product(range(len(cm)), range(len(cm))):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()
    plt.close()


# ─────────────────────────────────────────────────────────
# 5.  OPTIONAL: CLASSIFICATION REPORT
# ─────────────────────────────────────────────────────────
def print_classification_report(y_true, y_pred, class_names):
    """Pretty‑print precision / recall / F1 per class."""
    print(classification_report(y_true, y_pred, target_names=class_names))
