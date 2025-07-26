from dataclasses import dataclass
from pathlib import Path

# Using dataclasses to define the structure for data ingestion configuration.
# This ensures type safety and makes the configuration object-oriented.

@dataclass(frozen=True)
class DataIngestionConfig:
    """
    Configuration for the Data Ingestion component.
    
    Attributes:
        root_dir (Path): The root directory where data ingestion artifacts will be stored.
        source_data_path (Path): The path to the raw source data.
        local_data_file (Path): The path where the combined data will be saved locally.
    """
    root_dir: Path
    source_data_path: Path
    local_data_file: Path


@dataclass(frozen=True)
class DataTransformationConfig:
    """
    Configuration for the Data Transformation component.
    
    Attributes:
        root_dir (Path): The root directory for transformation artifacts.
        transformed_data_path (Path): Path to save the training data.
        test_data_path (Path): Path to save the testing data.
    """
    root_dir: Path
    transformed_data_path: Path
    test_data_path: Path


@dataclass(frozen=True)
class ModelTrainerConfig:
    """
    Configuration for the Model Trainer component.
    
    Attributes:
        root_dir (Path): Root directory for model training artifacts.
        trained_model_file_path (Path): Path to save the trained model (.pkl).
        vectorizer_file_path (Path): Path to save the TF-IDF vectorizer (.pkl).
    """
    root_dir: Path
    trained_model_file_path: Path
    vectorizer_file_path: Path