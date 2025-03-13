# Install gdown if not already installed
# !pip install gdown

import gdown
import zipfile
import os
import glob
import json
import requests
from tqdm import tqdm


def download_and_unzip(file_id, output_dir=None):
    """
    Downloads a zipped file from Google Drive using its file ID and unzips it to a specified directory.

    Parameters:
    - file_id (str): The file ID of the Google Drive file.
    - output_dir (str): The directory where the file should be extracted. Defaults to the current working directory.

    Returns:
    - str: The path to the extracted file.
    """
    if output_dir is None:
        output_dir = os.getcwd()
        
    os.makedirs(output_dir, exist_ok=True)
    url = f'https://drive.google.com/uc?id={file_id}'
    output = os.path.join(output_dir, 'temp.zip')
    gdown.download(url, output, quiet=False)

    with zipfile.ZipFile(output, 'r') as zip_ref:
        original_name = zip_ref.namelist()[0]
        zip_ref.extractall(output_dir)

    # Remove the temporary zip file
    os.remove(output)

    # The path to the extracted file
    extracted_file = os.path.join(output_dir, original_name)

    print(f"File extracted as: {extracted_file}, saved to {output_dir}")
    return extracted_file


def download_file_from_url(url: str, fname: str, chunk_size=1024):
    """
    Download a file from a given url
    """
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


def download_TinyStories(data_dir):
    """
    Downloads the TinyStories dataset to data_dir
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # download the TinyStories dataset
    data_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
    TinyStories_raw = os.path.join(data_dir, "TinyStories_all_data.tar.gz")
    if not os.path.exists(TinyStories_raw):
        print(f"Downloading {data_url} to {TinyStories_raw}...")
        download_file_from_url(data_url, TinyStories_raw)
    else:
        print(f"{TinyStories_raw} already exists, skipping download...")

    # unpack the tar.gz file into all the data shards (json files)
    TinyStories_unpack = os.path.join(data_dir, "TinyStories_all_data")
    if not os.path.exists(TinyStories_unpack):
        os.makedirs(TinyStories_unpack, exist_ok=True)
        print(f"Unpacking {TinyStories_raw}...")
        os.system(f"tar -xzf {TinyStories_raw} -C {TinyStories_unpack}")
    else:
        print(f"{TinyStories_unpack} already exists, skipping unpacking...")

    print("Download done.")
