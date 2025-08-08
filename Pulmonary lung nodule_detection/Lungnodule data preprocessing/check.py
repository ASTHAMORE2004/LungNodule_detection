import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from data_loader import get_generators, loading_the_data, change_label_names
from performance_utils import evaluate_model, get_predictions, plot_conf_matrix, print_classification_report
from config import DATA_DIR

# ──────────────────────────────────────────────────────
# 1. Load trained model
# ──────────────────────────────────────────────────────
model = load_model("lung_cancer_model.h5")

# ──────────────────────────────────────────────────────
# 2. Load and preprocess data
# ──────────────────────────────────────────────────────
df = loading_the_data(DATA_DIR)
change_label_names(df)
train_gen, valid_gen, test_gen = get_generators(df)
class_names = list(test_gen.class_indices.keys())

# ──────────────────────────────────────────────────────
# 3. Evaluate performance
# ──────────────────────────────────────────────────────
evaluate_model(model, train_gen, valid_gen, test_gen)
y_pred = get_predictions(model, test_gen)
plot_conf_matrix(test_gen.classes, y_pred, class_names)
print_classification_report(test_gen.classes, y_pred, class_names)

st.subheader("🧩 Available Layers")
for layer in model.layers:
    st.text(layer.name)