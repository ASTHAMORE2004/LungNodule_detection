# ------------------------------------------------------------------
# 0.  Imports & paths
# ------------------------------------------------------------------
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from data_loader import loading_the_data, change_label_names, get_generators
from config      import DATA_DIR, IMG_SIZE          # already in your repo

cnn_path   = "lung_cnn_model.h5"        # ← your saved CNN
eff_path   = "lung_cancer_model.h5"     # ← your saved EfficientNetB3

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd

def plot_roc_auc(y_true_cnn, y_prob_cnn, y_true_eff, y_prob_eff, class_names):
    y_true_cnn_1h = pd.get_dummies(y_true_cnn).values
    y_true_eff_1h = pd.get_dummies(y_true_eff).values

    plt.figure(figsize=(12, 5))

    # CNN ROC Curve
    plt.subplot(1, 2, 1)
    for i in range(len(class_names)):
        fpr, tpr, _ = roc_curve(y_true_cnn_1h[:, i], y_prob_cnn[:, i])
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={auc_score:.2f})")
    plt.plot([0,1],[0,1],'k--'); plt.title("CNN ROC Curve")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend()

    # EfficientNet ROC Curve
    plt.subplot(1, 2, 2)
    for i in range(len(class_names)):
        fpr, tpr, _ = roc_curve(y_true_eff_1h[:, i], y_prob_eff[:, i])
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={auc_score:.2f})")
    plt.plot([0,1],[0,1],'k--'); plt.title("EfficientNetB3 ROC Curve")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend()

    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------------
# 1.  Load models
# ------------------------------------------------------------------
cnn_model = load_model(cnn_path)
eff_model = load_model(eff_path)

# ------------------------------------------------------------------
# 2.  Build test generator (exact same one you used while training)
# ------------------------------------------------------------------
df = loading_the_data(DATA_DIR)
change_label_names(df)
_, _, test_gen = get_generators(df)     # keep shuffle=False in get_generators

n_classes   = len(test_gen.class_indices)
class_names = list(test_gen.class_indices.keys())

# ------------------------------------------------------------------
# 3.  Run inference on the ENTIRE test set  (1 500 images in LC25000)
#     Keras generators are iterable; model.predict will loop over it.
# ------------------------------------------------------------------
y_prob_cnn = cnn_model.predict(test_gen, verbose=1)
cnn_model.reset_metrics()              # just hygiene

y_prob_eff = eff_model.predict(test_gen, verbose=1)
eff_model.reset_metrics()

# IMPORTANT:  test_gen.classes always gives labels **in the same order**
y_true = test_gen.classes              # shape (N,)

# ------------------------------------------------------------------
# 4.  Plot ROC–AUC curves
# ------------------------------------------------------------------
plot_roc_auc(
    y_true_cnn = y_true,
    y_prob_cnn = y_prob_cnn,
    y_true_eff = y_true,               # same labels
    y_prob_eff = y_prob_eff,
    class_names = class_names
)
