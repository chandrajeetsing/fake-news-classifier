from fakeNewsClassifier.constants import *
from pathlib import Path
from fakeNewsClassifier.utils.common import read_yaml, create_directories
from fakeNewsClassifier.entity.config_entity import (DataIngestionConfig,
                                                      DataTransformationConfig,
                                                      ModelTrainerConfig)

class ConfigurationManager:
    """
    Manages the application's configuration by reading from YAML files
    and providing structured configuration objects.
    """
    def __init__(self, config_filepath=CONFIG_FILE_PATH):
        """
        Initializes the ConfigurationManager.

        Args:
            config_filepath (Path, optional): Path to the main config file.
                                              Defaults to CONFIG_FILE_PATH.
        """
        self.config = read_yaml(config_filepath)
        # Create the main artifacts directory
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Retrieves the data ingestion configuration.

        Returns:
            DataIngestionConfig: A dataclass object with data ingestion settings.
        """
        config = self.config.data_ingestion
        
        # Create the root directory for data ingestion artifacts
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_data_path=Path(config.source_data_path),
            local_data_file=Path(config.local_data_file)
        )

        return data_ingestion_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        """
        Retrieves the data transformation configuration.

        Returns:
            DataTransformationConfig: A dataclass object with data transformation settings.
        """
        config = self.config.data_transformation
        
        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=Path(config.root_dir),
            transformed_data_path=Path(config.transformed_data_path),
            test_data_path=Path(config.test_data_path)
        )

        return data_transformation_config

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        """
        Retrieves the model trainer configuration.

        Returns:
            ModelTrainerConfig: A dataclass object with model training settings.
        """
        config = self.config.model_trainer
        
        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=Path(config.root_dir),
            trained_model_file_path=Path(config.trained_model_file_path),
            vectorizer_file_path=Path(config.vectorizer_file_path)
        )

        return model_trainer_config