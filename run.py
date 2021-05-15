import os
import zipfile

import pandas as pd
import requests
import structlog

_LOGGER = structlog.get_logger(__file__)


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
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(f'./{local_dir}/{local_filename}', 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        _LOGGER.info(f'Successfully saved file to {local_dir}/{local_filename}')
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
    _LOGGER.info(f"Loading the excel data from {path_to_file} into a dataframe - this may take a while.")
    return pd.read_excel(path_to_file, engine='openpyxl')


if __name__ == "__main__":
    local_dir = './data'
    # download the file
    path_to_downloaded_zip_file = download_file('https://www.fire.tc.faa.gov/zip/MasterModelVersion3DDeliverable.zip', local_dir)
    # unzip the file
    path_to_file = unzip_file(path_to_downloaded_zip_file, local_dir)
    print(path_to_file)
    load_data(path_to_file)
