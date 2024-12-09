import os
import urllib.request

# import requests
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm


def retrieve_and_extract_gpt2(model_variant, storage_directory):
    # Validate model size
    permitted_variants = ("124M", "355M", "774M", "1558M")
    if model_variant not in permitted_variants:
        raise ValueError(f"Model variant not in {permitted_variants}")

    # Define paths
    variant_directory = os.path.join(storage_directory, model_variant)
    remote_base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    resource_names = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    # Download files
    os.makedirs(variant_directory, exist_ok=True)
    for resource in resource_names:
        resource_url = os.path.join(remote_base_url, model_variant, resource)
        resource_path = os.path.join(variant_directory, resource)
        fetch_resource(resource_url, resource_path)

    # Load settings and params
    tf_checkpoint_route = tf.train.latest_checkpoint(variant_directory)
    configuration = json.load(open(os.path.join(variant_directory, "hparams.json")))
    model_parameters = extract_gpt2_params_from_checkpoint(tf_checkpoint_route, configuration)

    return configuration, model_parameters


def fetch_resource(resource_url, destination):
    # Send a GET request to download the file
    with urllib.request.urlopen(resource_url) as response:
        # Get the total file size from headers, defaulting to 0 if not present
        resource_size = int(response.headers.get("Content-Length", 0))

        # Check if file exists and has the same size
        if os.path.exists(destination):
            local_resource_size = os.path.getsize(destination)
            if resource_size == local_resource_size:
                print(f"Resource already exists and is up-to-date: {destination}")
                return

        # Define the block size for reading the file
        chunk_dimension = 1024  # 1 Kilobyte

        # Initialize the progress bar with total file size
        progress_descriptor = os.path.basename(resource_url)  # Extract filename from URL
        with tqdm(total=resource_size, unit="iB", unit_scale=True, desc=progress_descriptor) as progress_tracker:
            # Open the destination file in binary write mode
            with open(destination, "wb") as file:
                # Read the file in chunks and write to destination
                while True:
                    data_chunk = response.read(chunk_dimension)
                    if not data_chunk:
                        break
                    file.write(data_chunk)
                    progress_tracker.update(len(data_chunk))  # Update progress bar


def extract_gpt2_params_from_checkpoint(checkpoint_route, configuration):
    # Initialize parameters dictionary with empty blocks for each layer
    model_structure = {"blocks": [{} for _ in range(configuration["n_layer"])]}

    # Iterate over each variable in the checkpoint
    for identifier, _ in tf.train.list_variables(checkpoint_route):
        # Load the variable and remove singleton dimensions
        parameter_matrix = np.squeeze(tf.train.load_variable(checkpoint_route, identifier))

        # Process the variable name to extract relevant parts
        parameter_components = identifier.split("/")[1:]  # Skip the 'model/' prefix

        # Identify the target dictionary for the variable
        destination_container = model_structure
        if parameter_components[0].startswith("h"):
            layer_index = int(parameter_components[0][1:])
            destination_container = model_structure["blocks"][layer_index]

        # Recursively access or create nested dictionaries
        for key in parameter_components[1:-1]:
            destination_container = destination_container.setdefault(key, {})

        # Assign the variable array to the last key
        final_key = parameter_components[-1]
        destination_container[final_key] = parameter_matrix

    return model_structure