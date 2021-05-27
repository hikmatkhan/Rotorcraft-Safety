import os
import zipfile
from ast import literal_eval
from collections import Counter
import random
from typing import List, Tuple

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
from sklearn.metrics.pairwise import linear_kernel
from sklearn.neighbors import KNeighborsClassifier

_LOGGER = structlog.get_logger(__file__)
pandarallel.initialize()
HEADER_COLUMN = 12


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


def vectorize(df: pd.DataFrame) -> Tuple[np.array, List[str], List[int]]:
    _LOGGER.info("Converting text to feature matrix")
    df['counter'] = df['lemmatized_filtered'].parallel_apply(lambda x: Counter(x))
    vectorizer = DictVectorizer(sparse=False)
    sparse_matrix = vectorizer.fit_transform(df['counter'])
    return sparse_matrix, vectorizer.get_feature_names(), df['Index No.\n (Do not alter or delete)']


def convert_to_vector_id_pairs(feature_matrix: np.array, report_ids: List[int]) -> List[Tuple[np.array, int]]:
    feature_vectors = list(feature_matrix)
    data = list(zip(feature_vectors, report_ids))
    return data


def shuffle_and_split_into_sets(data: List[Tuple[np.array, int]],
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

    # create a sparse feature matrix of size n x m,
    # where n = number of documents, m = number of words in vocabulary
    feature_matrix, feature_names, report_ids = vectorize(filtered_df)

    # apply term frequency–inverse document frequency (tf-idf)
    feature_matrix = compute_term_frequency_inverse_document_frequency(feature_matrix)

    # optionally apply dimensionality reduction (PCA)
    if dimensionality_reduction:
        num_components = 500
        _LOGGER.info(f"Performing dimensionality reduction using PCA. Reducing to {num_components}.")
        pca = PCA(n_components=num_components)
        sparse_matrix = pca.fit_transform(feature_matrix)

    # split the data up into (feature_vector, report_id) pairs
    pairs = convert_to_vector_id_pairs(feature_matrix, report_ids)

    # shuffle and split the data
    train, validation = shuffle_and_split_into_sets(pairs)

    # classification step
    train_data, train_report_ids = list(zip(*train))
    train_data = np.array(train_data)
    for feature_vector, report_id_of_document in validation:
        pairwise_distances = linear_kernel(np.expand_dims(feature_vector, 0), np.array(train_data)).flatten()
        sorted_distances = np.argsort(pairwise_distances)
        minimum_distance = sorted_distances[-1]
        report_id_of_closest_document = train_report_ids[minimum_distance]
        # for debugging
        keys_of_interest = ['Text', 'Location on sircraft of the defective or malfunctioning part',
                            'Text reflecting condition of failed part',
                            '1 A = Air Carrier   G = General Aviation',
                            'Segment Code 1 = aircraft, 2 = engine, 3 = propeller, 4 = component',
                            'Precautionary Measures Taken']
        actual_record = filtered_df[filtered_df['Index No.\n (Do not alter or delete)'] == report_id_of_document][keys_of_interest].to_dict(orient='records')
        closest_record = filtered_df[
            filtered_df['Index No.\n (Do not alter or delete)'] == report_id_of_closest_document][keys_of_interest].to_dict(orient='records')

        print(actual_record)
        print(closest_record)
