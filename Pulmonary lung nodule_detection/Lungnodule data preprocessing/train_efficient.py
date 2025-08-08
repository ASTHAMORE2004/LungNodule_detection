"""
train_efficient.py – EfficientNetB3 with 10% flip/aug for adenocarcinoma
"""

import os, pickle, math
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from data_loader import loading_the_data, change_label_names
from model_utils import build_efficientnetb3_model
from performance_utils import (
    model_performance, evaluate_model,
    get_predictions, plot_conf_matrix
)
from config import DATA_DIR, IMG_SIZE, EPOCHS, BATCH_SIZE

# ─────────────────────────────────────────────────────────
# 1. Load and split dataframe
# ─────────────────────────────────────────────────────────
df = loading_the_data(DATA_DIR)
change_label_names(df)

from sklearn.model_selection import train_test_split
train_df, tmp_df  = train_test_split(df, train_size=0.8, shuffle=True, random_state=42)
valid_df, test_df = train_test_split(tmp_df, train_size=0.5, shuffle=True, random_state=42)

# ─────────────────────────────────────────────────────────
# 2. Create Data Generators
# ─────────────────────────────────────────────────────────

base_gen = ImageDataGenerator(rescale=1./255)

# Split train_df into adenocarcinoma and others
aca_df    = train_df[train_df.labels == 'Lung_adenocarcinoma']
other_df  = train_df[train_df.labels != 'Lung_adenocarcinoma']

# (A) Generator for unaugmented "other" classes
# Get consistent class list
all_classes = sorted(df['labels'].unique())

# Other generator (non-adenocarcinoma)
other_gen = base_gen.flow_from_dataframe(
    other_df, x_col='filepaths', y_col='labels',
    classes=all_classes,  # force same label map
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=True)

# Augmented adenocarcinoma generator
aug_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
).flow_from_dataframe(
    aca_df, x_col='filepaths', y_col='labels',
    classes=all_classes,  # force same label map
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=True)

# Calculate training steps
steps_other = math.ceil(len(other_df) / BATCH_SIZE)
extra_steps = max(1, int(0.10 * len(train_df) / BATCH_SIZE))  # 10% synthetic boost

# (C) Combined generator
def combined_generator():
    while True:
        x_other, y_other = next(other_gen)
        yield x_other, y_other
        for _ in range(extra_steps):
            x_aug, y_aug = next(aug_gen)
            yield x_aug, y_aug

train_steps = steps_other + extra_steps

# ─────────────────────────────────────────────────────────
# 3. Validation and Test generators
# ─────────────────────────────────────────────────────────

valid_gen = base_gen.flow_from_dataframe(
    valid_df, x_col='filepaths', y_col='labels',
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False)

test_gen = base_gen.flow_from_dataframe(
    test_df, x_col='filepaths', y_col='labels',
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False)

# ─────────────────────────────────────────────────────────
# 4. Build & Train the Model
# ─────────────────────────────────────────────────────────

input_shape = (*IMG_SIZE, 3)
num_classes = len(valid_gen.class_indices)

model = build_efficientnetb3_model(input_shape, num_classes)

history = model.fit(
    combined_generator(),
    steps_per_epoch=train_steps,
    epochs=EPOCHS,
    validation_data=valid_gen
)

# ─────────────────────────────────────────────────────────
# 5. Save History
# ─────────────────────────────────────────────────────────

with open("history.pkl", "wb") as f:
    pickle.dump(history.history, f)

# ─────────────────────────────────────────────────────────
# 6. Evaluate Model
# ─────────────────────────────────────────────────────────

model_performance(history)
evaluate_model(model, valid_gen, valid_gen, test_gen)

y_pred = get_predictions(model, test_gen)
plot_conf_matrix(test_gen.classes, y_pred, list(valid_gen.class_indices.keys()))

model.save("lung_cancer_model.h5")
print("✅ Model and history saved.")
