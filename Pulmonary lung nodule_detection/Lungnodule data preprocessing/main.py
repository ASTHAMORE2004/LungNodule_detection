from tensorflow.keras.models import load_model
from data_loader import loading_the_data, change_label_names, get_generators
from config import DATA_DIR
import numpy as np

# Load data
df = loading_the_data(DATA_DIR); change_label_names(df)
train_gen, val_gen, test_gen = get_generators(df)

# Load models
cnn = load_model("lung_cnn_model.h5")
eff = load_model("lung_cancer_model.h5")  # EfficientNetB3 model

# Get predictions
y_prob_cnn = cnn.predict(test_gen, verbose=1)
y_prob_eff = eff.predict(test_gen, verbose=1)

y_true_cnn = test_gen.classes  # Same for both
y_true_eff = test_gen.classes

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class_names = ["Adeno", "SCC", "Benign"]

# Ground truth and predicted class indices
y_true_cnn = y_true_eff = test_gen.classes
y_pred_cnn = np.argmax(y_prob_cnn, axis=1)
y_pred_eff = np.argmax(y_prob_eff, axis=1)

# Confusion matrices
cm_cnn = confusion_matrix(y_true_cnn, y_pred_cnn)
cm_eff = confusion_matrix(y_true_eff, y_pred_eff)

# Plot confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(cm_cnn, annot=True, fmt="d", cmap="Blues", ax=axes[0])
axes[0].set_title("CNN Confusion Matrix")
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")

sns.heatmap(cm_eff, annot=True, fmt="d", cmap="Greens", ax=axes[1])
axes[1].set_title("EfficientNetB3 Confusion Matrix")
axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.show()

# Print classification reports
print("📌 CNN Classification Report")
print(classification_report(y_true_cnn, y_pred_cnn, target_names=class_names))

print("\n📌 EfficientNetB3 Classification Report")
print(classification_report(y_true_eff, y_pred_eff, target_names=class_names))
