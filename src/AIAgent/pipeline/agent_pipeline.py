# src/AIAgent/pipeline/agent_pipeline.py
from AIAgent.config.configuration import ConfigurationManager
from AIAgent.components.data_ingestion import DataIngestion
from AIAgent.components.content_processing import ContentProcessing
from AIAgent.logging import logger

class AgentPipeline:
    def __init__(self):
        pass

    def run(self):
        logger.info("Starting Agent Pipeline...")
        config_manager = ConfigurationManager()
        
        # Data Ingestion
        data_ingestion_config = config_manager.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.fetch_and_save_data()

        # Content Processing
        content_processing_config = config_manager.get_content_processing_config()
        content_processor = ContentProcessing(
            processing_config=content_processing_config,
            ingestion_config=data_ingestion_config # It needs to know where the raw data is
        )
        content_processor.process_content() # <-- RUN THE NEW COMPONENT
        
        logger.info("Agent Pipeline finished.")