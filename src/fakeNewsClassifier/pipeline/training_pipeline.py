from fakeNewsClassifier.config.configuration import ConfigurationManager
from fakeNewsClassifier.components.data_ingestion import DataIngestion
from fakeNewsClassifier.components.data_transformation import DataTransformation
from fakeNewsClassifier.components.model_trainer import ModelTrainer
from fakeNewsClassifier.logging import logger

class TrainPipeline:
    """
    Orchestrates the entire model training workflow.
    
    This pipeline sequentially runs all the necessary components for training,
    starting from data ingestion.
    """
    def __init__(self):
        """
        Initializes the training pipeline.
        """
        self.config_manager = ConfigurationManager()

    def main(self):
        """
        The main entry point to run the training pipeline.
        """
        try:
            logger.info("Starting the training pipeline.")
            
            # # --- Data Ingestion Step ---
            # logger.info("Executing Data Ingestion component.")
            # data_ingestion_config = self.config_manager.get_data_ingestion_config()
            # data_ingestion = DataIngestion(config=data_ingestion_config)
            # data_ingestion.ingest_data()
            # logger.info("Data Ingestion component finished successfully.")


            # # --- Data Transformation Step ---
            # logger.info("Executing Data Transformation component.")
            data_transformation_config = self.config_manager.get_data_transformation_config()
            # data_transformation = DataTransformation(config=data_transformation_config)
            # # Pass the path of the ingested data to the transformation component
            # data_transformation.transform_data(data_path=data_ingestion_config.local_data_file)
            # logger.info("Data Transformation component finished successfully.")

            # --- Model Training Step ---
            logger.info("Executing Model Trainer component.")
            model_trainer_config = self.config_manager.get_model_trainer_config()
            model_trainer = ModelTrainer(config=model_trainer_config)
            model_trainer.train(
                train_data_path=data_transformation_config.transformed_data_path,
                test_data_path=data_transformation_config.test_data_path
            )
            logger.info("Model Trainer component finished successfully.")

            

            logger.info("Training pipeline finished successfully.")

        except Exception as e:
            logger.error(f"Training pipeline failed with error: {e}")
            raise e

# This block allows the script to be run directly
if __name__ == '__main__':
    pipeline = TrainPipeline()
    pipeline.main()