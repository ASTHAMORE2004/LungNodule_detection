import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import DATA_DIR, IMG_SIZE, BATCH_SIZE, SEED

def loading_the_data(data_dir):
    filepaths, labels = [], []
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        for file in os.listdir(folder_path):
            filepaths.append(os.path.join(folder_path, file))
            labels.append(folder)
    return pd.DataFrame({'filepaths': filepaths, 'labels': labels})

def change_label_names(df):
    mapping = {
        'lung_aca': 'Lung_adenocarcinoma',
        'lung_n': 'Lung_benign_tissue',
        'lung_scc': 'Lung squamous_cell_carcinoma'
    }
    df['labels'] = df['labels'].replace(mapping)

def get_generators(df):
    train_df, temp_df = train_test_split(df, train_size=0.8, shuffle=True, random_state=SEED)
    valid_df, test_df = train_test_split(temp_df, train_size=0.5, shuffle=True, random_state=SEED)

    tr_gen = ImageDataGenerator(rescale=1./255)
    ts_gen = ImageDataGenerator(rescale=1./255)

    train_gen = tr_gen.flow_from_dataframe(train_df, x_col='filepaths', y_col='labels',
                                           target_size=IMG_SIZE, class_mode='categorical',
                                           batch_size=BATCH_SIZE, shuffle=True)

    valid_gen = ts_gen.flow_from_dataframe(valid_df, x_col='filepaths', y_col='labels',
                                           target_size=IMG_SIZE, class_mode='categorical',
                                           batch_size=BATCH_SIZE, shuffle=True)

    test_gen = ts_gen.flow_from_dataframe(test_df, x_col='filepaths', y_col='labels',
                                          target_size=IMG_SIZE, class_mode='categorical',
                                          batch_size=BATCH_SIZE, shuffle=False)

    return train_gen, valid_gen, test_gen
