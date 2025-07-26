import os
import pandas as pd
from fakeNewsClassifier.logging import logger
from fakeNewsClassifier.entity.config_entity import DataIngestionConfig
from pathlib import Path

class DataIngestion:
    """
    Handles the ingestion of data from the BBC News dataset structure.
    
    This component reads text files from category-named subdirectories,
    assigns the category as a label, combines them into a single DataFrame,
    and saves it to a specified location for further processing.
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
        Executes the data ingestion process for the BBC dataset.
        """
        logger.info("Starting data ingestion process for BBC dataset.")
        
        try:
            # The root directory of the BBC dataset
            bbc_data_path = Path(self.config.source_data_path) / 'bbc'
            
            if not bbc_data_path.exists():
                logger.error(f"Source data directory not found at: {bbc_data_path}")
                raise FileNotFoundError(f"Source data directory not found at: {bbc_data_path}")

            all_articles = []
            categories = [d for d in bbc_data_path.iterdir() if d.is_dir()]
            
            logger.info(f"Found categories: {[cat.name for cat in categories]}")

            for category_path in categories:
                category_name = category_path.name
                logger.info(f"Processing category: {category_name}")
                for text_file in category_path.glob('*.txt'):
                    try:
                        with open(text_file, 'r', encoding='utf-8', errors='ignore') as f:
                            text_content = f.read()
                            all_articles.append({
                                'text': text_content,
                                'category': category_name
                            })
                    except Exception as e:
                        logger.warning(f"Could not read file {text_file}: {e}")

            # Convert the list of dictionaries to a DataFrame
            df = pd.DataFrame(all_articles)
            logger.info(f"Successfully read {len(df)} articles.")
            
            # Shuffle the dataset
            df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
            logger.info("Shuffled the combined dataset.")

            # Save the combined data
            output_path = self.config.local_data_file
            df_shuffled.to_csv(output_path, index=False)
            logger.info(f"Data ingestion successful. Saved combined data to: {output_path}")

        except Exception as e:
            logger.error(f"An unexpected error occurred during data ingestion: {e}")
            raise e