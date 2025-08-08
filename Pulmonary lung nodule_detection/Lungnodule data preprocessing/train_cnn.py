from model_utils   import build_cnn_model
from data_loader   import loading_the_data, change_label_names, get_generators
from config        import DATA_DIR, IMG_SIZE

df = loading_the_data(DATA_DIR); change_label_names(df)
train_gen, val_gen, test_gen = get_generators(df)

cnn = build_cnn_model((*IMG_SIZE, 3), len(train_gen.class_indices))
history = cnn.fit(train_gen, epochs=5, validation_data=val_gen)
cnn.save("lung_cnn_model.h5")

import pickle, gzip
with gzip.open("cnn_history.pkl", "wb") as f:
    pickle.dump(history.history, f)
