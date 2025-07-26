import os
import pandas as pd
from fakeNewsClassifier.logging import logger
from fakeNewsClassifier.entity.config_entity import DataIngestionConfig

class DataIngestion:
    """
    Handles the ingestion of data from source files.
    
    This component reads the raw 'True.csv' and 'Fake.csv' files,
    adds a 'label' column to distinguish between them, combines them
    into a single DataFrame, shuffles the data, and saves it to a
    specified location for further processing.
    """
    def __init__(self, config: DataIngestionConfig):
        """
        Initializes the DataIngestion component with its configuration.

        Args:
            config (DataIngestionConfig): A dataclass object containing paths and settings
                                          for data ingestion.
        """
        self.config = config

    def ingest_data(self):
        """
        Executes the data ingestion process.
        """
        logger.info("Starting data ingestion process.")
        
        try:
            # Define paths to the raw data files
            true_data_path = os.path.join(self.config.source_data_path, 'True.csv')
            fake_data_path = os.path.join(self.config.source_data_path, 'Fake.csv')

            # Read the datasets
            logger.info(f"Reading real news data from: {true_data_path}")
            df_true = pd.read_csv(true_data_path)
            
            logger.info(f"Reading fake news data from: {fake_data_path}")
            df_fake = pd.read_csv(fake_data_path)

            # Add label column: 0 for real, 1 for fake
            df_true['label'] = 0
            df_fake['label'] = 1
            
            logger.info("Added 'label' column to both dataframes.")

            # Combine the dataframes
            df_combined = pd.concat([df_true, df_fake], ignore_index=True)
            logger.info("Combined real and fake news dataframes.")

            # Shuffle the dataset to mix the data
            df_shuffled = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
            logger.info("Shuffled the combined dataset.")

            # Save the combined and shuffled data
            output_path = self.config.local_data_file
            df_shuffled.to_csv(output_path, index=False)
            logger.info(f"Data ingestion successful. Saved combined data to: {output_path}")

        except FileNotFoundError as e:
            logger.error(f"Error: One of the source files was not found. Please check the path in config.yaml. Details: {e}")
            raise e
        except Exception as e:
            logger.error(f"An unexpected error occurred during data ingestion: {e}")
            raise e