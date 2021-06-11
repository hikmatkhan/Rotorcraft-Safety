import os
import zipfile
from ast import literal_eval
from collections import Counter
import random
from typing import List, Tuple, Dict

import nltk
import numpy as np
import pandas as pd
import requests
import structlog
from nltk import RegexpTokenizer, WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.decomposition import PCA
from sklearn.feature_extraction import DictVectorizer
from pandarallel import pandarallel
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from tensorflow import keras, one_hot
from tensorflow.keras import layers
from tensorflow.keras.callbacks import CSVLogger

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


def tokenize(df: pd.DataFrame) -> pd.DataFrame:
    _LOGGER.info("Tokenizing text")
    tokenizer = RegexpTokenizer(r'\w+')
    df['tokenized'] = df['Text'].apply(lambda x: tokenizer.tokenize(x))
    return df


def lemmatize(df: pd.DataFrame) -> pd.DataFrame:
    _LOGGER.info("Lemmatizing text")
    nltk.download('wordnet')
    lemmatiser = WordNetLemmatizer()
    df['lemmatized'] = df['tokenized'].apply(lambda tokens:
                                             [lemmatiser.lemmatize(token.lower(), pos='v') for token in tokens])
    return df


def remove_stop_words(df: pd.DataFrame) -> pd.DataFrame:
    _LOGGER.info("Removing stop words from text")
    nltk.download('stopwords')
    df['lemmatized_filtered'] = df['lemmatized'].parallel_apply(lambda lemmas:
                                                                [lemma for lemma in lemmas if
                                                                 lemma not in stopwords.words('english')])
    return df


def vectorize(df: pd.DataFrame) -> Tuple[np.array, List[str]]:
    _LOGGER.info("Converting text to feature matrix")
    df['counter'] = df['lemmatized_filtered'].parallel_apply(lambda x: Counter(x))
    vectorizer = DictVectorizer(sparse=False)
    sparse_matrix = vectorizer.fit_transform(df['counter'])
    return sparse_matrix, vectorizer.get_feature_names()


def shuffle_and_split_into_training_and_validation_sets(data: List[Tuple[np.array, int]],
                                                        ratio: float = 0.9) -> Tuple[List[np.array], List[np.array]]:
    random.shuffle(data)
    number_of_samples = len(data)
    train_samples = int(ratio * number_of_samples)
    return data[:train_samples], data[train_samples:]


def compute_term_frequency_inverse_document_frequency(feature_matrix: np.array) -> np.array:
    transformer = TfidfTransformer(smooth_idf=False)
    ifd_matrix = transformer.fit_transform(feature_matrix)
    ifd_matrix = np.squeeze(np.array([x.toarray() for x in ifd_matrix]))
    return ifd_matrix


def extract_and_encode_labels(df: pd.DataFrame) -> Tuple[np.array, Dict[str, int]]:
    label_mapping = dict((label, i) for i, label in enumerate(df[LABEL_COLUMN].unique()))
    labels = list(df[LABEL_COLUMN].map(label_mapping))
    labels = one_hot(np.array(labels), len(label_mapping))
    return np.array(labels), label_mapping


if __name__ == "__main__":
    local_dir = './data'

    compute_features = not os.path.exists(f'{local_dir}/feature_data.csv')
    dimensionality_reduction = False

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
        sanitized_df = sanitize_df(df)

        # tokenize the text column in the dataframe
        tokenized_df = tokenize(sanitized_df)

        # lemmatize the tokens in the dataframe
        lemmatized_df = lemmatize(tokenized_df)

        # remove english stopwords from lemmatized tokens
        filtered_df = remove_stop_words(lemmatized_df)

        # save preprocessed data to save time for future runs
        filtered_df.to_csv(f'{local_dir}/feature_data.csv')
    else:
        # don't go through the hassle of of preprocessing if we already have the preprocessed data saved
        filtered_df = pd.read_csv(f'{local_dir}/feature_data.csv')
        # we have to do this because otherwise, this column is loaded in as a string :(
        filtered_df['lemmatized_filtered'] = filtered_df['lemmatized_filtered'].parallel_apply(literal_eval)

    count_of_null_labels = len(filtered_df[filtered_df[LABEL_COLUMN].isnull()])
    filtered_df = filtered_df.dropna(subset=[LABEL_COLUMN])
    _LOGGER.info(f"Dropped {count_of_null_labels} records because {LABEL_COLUMN} was null or NaN")

    # create a sparse feature matrix of size n x m,
    # where n = number of documents, m = number of words in vocabulary
    feature_matrix, feature_names = vectorize(filtered_df)

    # apply term frequency–inverse document frequency (tf-idf)
    feature_matrix = compute_term_frequency_inverse_document_frequency(feature_matrix)

    # optionally apply dimensionality reduction (PCA)
    if dimensionality_reduction:
        num_components = 500
        _LOGGER.info(f"Performing dimensionality reduction using PCA. Reducing to {num_components}.")
        pca = PCA(n_components=num_components)
        feature_matrix = pca.fit_transform(feature_matrix)

    labels, label_mapping = extract_and_encode_labels(filtered_df)
    num_labels = len(label_mapping)
    num_features = feature_matrix.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(feature_matrix, labels, test_size=0.1, random_state=42)

    inputs = keras.Input(shape=(num_features,))
    layer_1 = layers.Dense(512, activation="relu")(inputs)
    layer_2 = layers.Dense(256, activation="relu")(layer_1)
    layer_3 = layers.Dense(128, activation="relu")(layer_2)
    layer_4 = layers.Dense(64, activation="relu")(layer_3)
    outputs = layers.Dense(num_labels, activation="softmax")(layer_4)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(),  # Optimizer
        # Loss function to minimize
        loss=keras.losses.CategoricalCrossentropy(),
        # List of metrics to monitor
        metrics=[keras.metrics.Accuracy()],
    )
    model.fit(X_train, y_train,
              validation_data=(X_test, y_test), shuffle=True, epochs=200, batch_size=64,
              callbacks=[CSVLogger('./results.csv')])
    model.save('model')







