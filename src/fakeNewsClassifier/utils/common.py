import os
from box.exceptions import BoxValueError
import yaml
from fakeNewsClassifier.logging import logger
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads a YAML file and returns its content as a ConfigBox object.

    Args:
        path_to_yaml (Path): Path to the YAML file.

    Raises:
        ValueError: If the YAML file is empty.
        e: If any other exception occurs.

    Returns:
        ConfigBox: A ConfigBox object containing the YAML content.
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            if not content:
                raise ValueError("YAML file is empty")
            logger.info(f"YAML file '{path_to_yaml}' loaded successfully.")
            return ConfigBox(content)
    except (BoxValueError, ValueError) as e:
        # Raise a specific error for empty or invalid YAML
        raise ValueError(f"Error reading YAML file at {path_to_yaml}: {e}")
    except Exception as e:
        # Catch any other unexpected errors
        raise e

@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """
    Creates a list of directories.

    Args:
        path_to_directories (list): A list of paths of directories to be created.
        verbose (bool, optional): If True, logs the creation of each directory. Defaults to True.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at: {path}")

@ensure_annotations
def get_size(path: Path) -> str:
    """
    Gets the size of a file in kilobytes.

    Args:
        path (Path): Path of the file.

    Returns:
        str: The size of the file in KB as a formatted string.
    """
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_in_kb} KB"