import os
import zipfile
from ast import literal_eval
from typing import List, Tuple, Dict

import nltk
import numpy as np
import pandas as pd
import requests
import structlog
import matplotlib.pyplot as plt
from pandarallel import pandarallel
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from tensorflow import keras, one_hot
from tensorflow.keras import layers
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.layers import ReLU

plt.rcParams.update({'figure.figsize': (16.0, 12.0)})
_LOGGER = structlog.get_logger(__file__)
pandarallel.initialize()
HEADER_COLUMN = 12
LABEL_COLUMN = 'False Warning'


def download_file(url: str, local_dir: str = '.', local_filename: str = '') -> str:
    """
    Downloads a file from a provided url to a local directory
    :param url: URL to download the file from
    :param local_dir: Local directory to download the file to (created if it does not exist)
    :param local_filename: What to name the file when saved
     (if empty or none, assume the name of the original name of the file)
    :return: the name of the file which was saved
    """
    os.makedirs(f'{local_dir}', exist_ok=True)
    local_filename = local_filename if local_filename else url.split('/')[-1]
    if os.path.exists(f'{local_dir}/{local_filename}'):
        _LOGGER.info(f'{local_dir}/{local_filename} already exists. Skipping download.')
    else:
        _LOGGER.info(f"Downloading file from {url} to {local_dir}/{local_filename}.")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(f'./{local_dir}/{local_filename}', 'wb') as f:
                for chunk in r.iter_content(chunk_size=128):
                    f.write(chunk)
        _LOGGER.info(f"Finished saving file from {url} to {local_dir}/{local_filename}.")
    return f'{local_dir}/{local_filename}'


def unzip_file(path_to_zip_file: str, dir_to_extract_to: str) -> str:
    """
    Unzips a zip file to a provided directory
    :param path_to_file: path to zip file
    :param dir_to_extract_to: directory to extract zip file
    :return: full path to unzipped file (assuming there is only one)
    """
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(dir_to_extract_to)
        return f'{dir_to_extract_to}/{zip_ref.namelist()[0]}'


def load_data(path_to_file: str) -> pd.DataFrame:
    """
    Loads excel data from a supplied path into a Pandas dataframe
    :param path_to_file: path to excel file
    :return: Pandas dataframe containing contents of excel spreadsheet
    """
    _LOGGER.info(f"Started loading the excel data from {path_to_file} into a dataframe - this may take a while. "
                 f"You may want to grab a coffee.")
    df = pd.read_excel(path_to_file, engine='openpyxl', header=HEADER_COLUMN)
    _LOGGER.info(f"Finished loading the excel data from {path_to_file} into a dataframe.")
    return df


def sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans up column names and elements in a dataframe by stripping out whitespaces/delimiters,
    dropping rows with null text, etc.
    :param df: dataframe we're operating on/cleaning
    :return: dataframe with cleaned up columns and elements
    """
    _LOGGER.info("Cleaning up dataframe")
    df = df.dropna(subset=['Text'])
    df = df.rename(columns=lambda x: x.strip() if isinstance(x, str) else x)
    df = df.parallel_applymap(lambda x: x.strip() if isinstance(x, str) else x)
    return df


def vectorize(df: pd.DataFrame, **kwargs) -> Tuple[np.array, List[str]]:
    _LOGGER.info("Converting text to feature matrix")
    vectorizer = TfidfVectorizer(**kwargs)
    sparse_matrix = vectorizer.fit_transform(df['Text'])
    feature_matrix = sparse_matrix.todense()
    return feature_matrix, vectorizer.get_feature_names()


def extract_and_encode_labels(df: pd.DataFrame) -> Tuple[np.array, Dict[str, int]]:
    label_mapping = dict((label, i) for i, label in enumerate(df[LABEL_COLUMN].unique()))
    labels = list(df[LABEL_COLUMN].map(label_mapping))
    labels = one_hot(np.array(labels), len(label_mapping))
    return np.array(labels), label_mapping


if __name__ == "__main__":
    local_dir = './data'

    compute_features = not os.path.exists(f'{local_dir}/feature_data.csv')
    model_type = "knn"

    if compute_features:
        # download the file
        path_to_downloaded_zip_file = download_file(
            'https://www.fire.tc.faa.gov/zip/MasterModelVersion3DDeliverable.zip',
            local_dir)
        # unzip the file
        path_to_file = unzip_file(path_to_downloaded_zip_file, local_dir)

        # load the file into a Pandas dataframe
        df = load_data(path_to_file)

        # clean up the dataframe (remove whitespace from columns and entries, remove rows with no data, etc.)
        df = sanitize_df(df)

        # save preprocessed data to save time for future runs
        df.to_csv(f'{local_dir}/feature_data.csv')
    else:
        # don't go through the hassle of preprocessing if we already have the preprocessed data saved
        df = pd.read_csv(f'{local_dir}/feature_data.csv')

    count_of_null_labels = len(df[df[LABEL_COLUMN].isnull()])
    df = df.dropna(subset=[LABEL_COLUMN])
    _LOGGER.info(f"Dropped {count_of_null_labels} records because {LABEL_COLUMN} was null or NaN")

    # create a sparse feature matrix of size n x m,
    # where n = number of documents, m = number of words in vocabulary
    feature_matrix, feature_names = vectorize(df, min_df=0.001, use_idf=False, norm='l1')

    labels, label_mapping = extract_and_encode_labels(df)
    num_labels = len(label_mapping)
    num_features = feature_matrix.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(feature_matrix, labels, test_size=0.05, random_state=1)

    _LOGGER.info(f"Training on {X_train.shape[0]} samples, validating on {X_test.shape[0]} samples.")
    _LOGGER.info(f"Number of features: {num_features}")

    if model_type == "mlp":
        inputs = keras.Input(shape=(num_features,))
        layer_1 = layers.Dense(8192, activation=ReLU())(inputs)
        layer_2 = layers.Dense(2048, activation=ReLU())(layer_1)
        layer_3 = layers.Dense(512, activation=ReLU())(layer_2)
        layer_4 = layers.Dense(128, activation=ReLU())(layer_3)
        layer_5 = layers.Dense(32, activation=ReLU())(layer_4)
        layer_6 = layers.Dense(8, activation=ReLU())(layer_5)
        outputs = layers.Dense(num_labels, activation="softmax")(layer_6)

        model = keras.Model(inputs=inputs, outputs=outputs)
        _LOGGER.info(model.summary())
        model.compile(
            optimizer=keras.optimizers.Adamax(),  # Optimizer
            loss=keras.losses.CategoricalCrossentropy(),  # Loss function to minimize
            metrics=[keras.metrics.Accuracy()]  # List of metrics to monitor
        )
        model.fit(X_train, y_train,
                  validation_data=(X_test, y_test), shuffle=True, epochs=200, batch_size=64,
                  callbacks=[CSVLogger('./results.csv')])
        model.save('model')
    elif model_type == "rf":
        rf = RandomForestClassifier(n_jobs=-1)
        y_train = np.argmax(y_train, axis=1)
        y_test = np.argmax(y_test, axis=1)
        rf.fit(X_train, y_train)
        training_acc = rf.score(X_train, y_train)
        validation_acc = rf.score(X_test, y_test)
        _LOGGER.info(f"Training accuracy with Random Forest: {training_acc}")
        _LOGGER.info(f"Validation accuracy with Random Forest: {validation_acc}")
    elif model_type == "knn":
        knn = KNeighborsClassifier(n_jobs=-1)
        y_train = np.argmax(y_train, axis=1)
        y_test = np.argmax(y_test, axis=1)
        knn.fit(X_train, y_train)
        training_acc = knn.score(X_train, y_train)
        validation_acc = knn.score(X_test, y_test)
        _LOGGER.info(f"Training accuracy with kNN: {training_acc}")
        _LOGGER.info(f"Validation accuracy with kNN: {validation_acc}")








