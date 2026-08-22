import os
import json
import requests
from urllib.parse import urljoin

# Conditional imports with user-friendly warnings
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    import tensorflow as tf
except ImportError:
    tf = None


def download_file(url, destination, backup_url=None):
    """
    Downloads a file from a URL to a destination path, with support for a backup URL.
    """
    def _attempt_download(download_url):
        response = requests.get(download_url, stream=True, timeout=60)
        response.raise_for_status()

        file_size = int(response.headers.get("Content-Length", 0))

        # Check if file exists and has same size
        if os.path.exists(destination):
            file_size_local = os.path.getsize(destination)
            if file_size and file_size == file_size_local:
                print(f"File already exists and is up-to-date: {destination}")
                return True

        block_size = 1024  # 1 KB
        desc = os.path.basename(download_url)
        
        # Use progress bar if tqdm is installed, otherwise fallback to simple messaging
        if tqdm is not None:
            with tqdm(total=file_size, unit="iB", unit_scale=True, desc=desc) as progress_bar:
                with open(destination, "wb") as file:
                    for chunk in response.iter_content(chunk_size=block_size):
                        if chunk:
                            file.write(chunk)
                            progress_bar.update(len(chunk))
        else:
            print(f"Downloading {desc} (size: {file_size / (1024*1024):.2f} MB)...")
            with open(destination, "wb") as file:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        file.write(chunk)
            print(f"Finished downloading {desc}.")
        return True

    try:
        if _attempt_download(url):
            return
    except requests.exceptions.RequestException as e:
        if backup_url is not None:
            print(f"Primary URL ({url}) failed. Attempting backup URL: {backup_url}")
            try:
                if _attempt_download(backup_url):
                    return
            except requests.exceptions.RequestException:
                pass

        error_message = (
            f"Failed to download from both primary URL ({url})\n"
            f"{'and backup URL (' + backup_url + ')' if backup_url else ''}.\n"
            "Check your internet connection or the file availability.\n"
            "For help, visit: https://github.com/rasbt/LLMs-from-scratch/discussions/273"
        )
        print(error_message)
        raise e
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise e


def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    """
    Loads GPT-2 parameters from a TensorFlow checkpoint path into a PyTorch-compatible shape dict.
    Requires tensorflow and numpy to be installed.
    """
    if tf is None or np is None:
        raise ImportError(
            "Both 'tensorflow' and 'numpy' must be installed to load parameters from a TensorFlow checkpoint.\n"
            "Please run: pip install tensorflow numpy"
        )

    # Initialize parameters dictionary with empty blocks for each layer
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # Iterate over each variable in the checkpoint
    for name, _ in tf.train.list_variables(ckpt_path):
        # Load the variable and remove singleton dimensions
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # Process the variable name to extract relevant parts
        variable_name_parts = name.split("/")[1:]  # Skip the 'model/' prefix

        # Identify the target dictionary for the variable
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        # Recursively access or create nested dictionaries
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        # Assign the variable array to the last key
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params


def download_and_load_gpt2(model_size="124M", models_dir="gpt2"):
    """
    Downloads GPT-2 model weights from OpenAI's blob storage (or a backup URL)
    and loads settings and parameters from the TensorFlow checkpoint.
    """
    # Validate model size
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    # Define paths
    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    backup_base_url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/gpt2"
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    # Download files
    os.makedirs(model_dir, exist_ok=True)
    for filename in filenames:
        # Use urljoin or standard string formatting/joining for url paths
        file_url = f"{base_url}/{model_size}/{filename}"
        backup_url = f"{backup_base_url}/{model_size}/{filename}"
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path, backup_url)

    # Load settings and params
    # Note: tf.train.latest_checkpoint relies on tensorflow.
    if tf is not None:
        tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    else:
        # Fallback to manual path construction if tensorflow is missing for check
        tf_ckpt_path = os.path.join(model_dir, "model.ckpt")

    hparams_path = os.path.join(model_dir, "hparams.json")
    with open(hparams_path, "r", encoding="utf-8") as f:
        settings = json.load(f)

    try:
        params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)
    except ImportError as e:
        print(f"\n[Warning] Weights were downloaded successfully to '{model_dir}', but:")
        print(e)
        print("Returning settings only. Parameters cannot be parsed without dependency packages.")
        params = None

    return settings, params


if __name__ == "__main__":
    # Example execution to download the 124M model weights
    print("Initiating download for GPT-2 124M parameters...")
    settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
    print("Download completed structure successfully.")
    print("Settings:", settings)
