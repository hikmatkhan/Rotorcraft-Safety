import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# For mutliple devices (GPUs: 4, 5, 6, 7)
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import os
import zipfile
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
print("NP:", np.__version__)
print("PD:", pd.__version__)
print("Request:", requests.__version__)
print("SKLearn:", sklearn.__version__)

plt.rcParams.update({'figure.figsize': (16.0, 12.0)})
HEADER_COLUMN = 12
LABEL_COLUMN = 'False Warning'
TEXT_COLUMN = 'Text'

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
        print("{0}/{1} already exists. Skipping download.".format(local_dir, local_filename))
    else:
        print("Downloading file from {0} to {1}/{2}.".format(url, local_dir, local_filename))
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(f'./{local_dir}/{local_filename}', 'wb') as f:
                for chunk in r.iter_content(chunk_size=128):
                    f.write(chunk)
        print("Finished saving file from {0} to {1}/{2}.".format(url, local_dir, local_filename))
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
    print("Started loading the excel data from {0} into a datafram - this may take a while. You may want to grab a coffee.".format(path_to_file))
    df = pd.read_excel(path_to_file, engine='openpyxl', header=HEADER_COLUMN)
    print("Finished loading the excel data from {0} into a dataframe.".format(path_to_file))
    return df


def vectorize(df: pd.DataFrame, **kwargs) -> Tuple[np.array, List[str]]:
    print("Converting text to feature matrix")
    vectorizer = TfidfVectorizer(**kwargs)
    sparse_matrix = vectorizer.fit_transform(df[TEXT_COLUMN])
    feature_matrix = sparse_matrix.todense()
    return feature_matrix, vectorizer.get_feature_names()


def extract_and_encode_labels(df: pd.DataFrame) -> Tuple[np.array, Dict[str, int]]:
    label_mapping = dict((label, i) for i, label in enumerate(df[LABEL_COLUMN].unique()))
    labels = list(df[LABEL_COLUMN].map(label_mapping))
    return np.array(labels), label_mapping

def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements


if __name__ == "__main__":
    local_dir = './data'

    compute_features = not os.path.exists(f'{local_dir}/feature_data.csv')
    model_type = "mlp" #"{knn", "mlp", "rf"}  

    if compute_features:
        # download the file
        path_to_downloaded_zip_file = download_file(
            'https://www.fire.tc.faa.gov/zip/MasterModelVersion3DDeliverable.zip',
            local_dir)
        # unzip the file
        path_to_file = unzip_file(path_to_downloaded_zip_file, local_dir)

        # load the file into a Pandas dataframe
        df = load_data(path_to_file)

        # save preprocessed data to save time for future runs
        df.to_csv(f'{local_dir}/feature_data.csv')
    else:
        # don't go through the hassle of preprocessing if we already have the preprocessed data saved
        df = pd.read_csv(f'{local_dir}/feature_data.csv')

    count_of_no_text = len(df[df[TEXT_COLUMN].isnull()])
    df = df.dropna(subset=[TEXT_COLUMN])
    print("Dropped {0} records because {1} was null or NaN".format(count_of_no_text, TEXT_COLUMN))

    count_of_null_labels = len(df[df[LABEL_COLUMN].isnull()])
    df = df.dropna(subset=[LABEL_COLUMN])
    print("Dropped {0} records because {1} was null or NaN".format(count_of_null_labels, LABEL_COLUMN))

    # create a sparse feature matrix of size n x m,
    # where n = number of documents, m = number of words in vocabulary
    feature_matrix, feature_names = vectorize(df, min_df=0.001)

    labels, label_mapping = extract_and_encode_labels(df)
    num_labels = len(label_mapping)
    num_features = feature_matrix.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(feature_matrix, labels, test_size=0.05, random_state=1)

    print("Training on {0} sample, validating on {1} samples".format(X_train.shape[0], X_test.shape[0]))
    print("Number of features: {0}".format(num_features))

    if model_type == "mlp":
        print("MLP training: This may take a while. You may want to grab a coffee.")
        model = MLPClassifier(hidden_layer_sizes=(num_features, 256, 128, 64, 32, num_labels),
                              #(8192, 2048, 512, 128, 32, 8, num_labels), 
                              activation='relu', solver='adam', verbose=1, learning_rate_init=0.01, batch_size=4096,
                              max_iter=25)
        model.fit(X_train, y_train)
        predict_train = model.predict(X_train)
        predict_test = model.predict(X_test)
        training_acc = confusion_matrix(predict_train, y_train)
        validation_acc = confusion_matrix(predict_test, y_test)
        print("Training accuracy with MLP: {0}".format(accuracy(training_acc)))
        print("Validation accuracy with MLP: {0}".format(accuracy(validation_acc)))
    elif model_type == "rf":
        rf = RandomForestClassifier(n_jobs=-1)
        rf.fit(X_train, y_train)
        training_acc = rf.score(X_train, y_train)
        validation_acc = rf.score(X_test, y_test)
        print("Training accuracy with Random Forest: {0}".format(training_acc))
        print("Validation accuracy with Random Forest: {0}".format(validation_acc))
    elif model_type == "knn":
        knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        knn.fit(X_train, y_train)
        training_acc = knn.score(X_train, y_train)
        validation_acc = knn.score(X_test, y_test)
        print("Training accuracy with kNN: {0}".format(training_acc))
        print("Validation accuracy with kNN: {0}".format(validation_acc))