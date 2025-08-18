# src/AIAgent/config/configuration.py
from AIAgent.utils.common import read_yaml, create_directories
from AIAgent.entity.config_entity import DataIngestionConfig
from AIAgent.constants import PARAMS_FILE_PATH
from pathlib import Path
from AIAgent.entity.config_entity import ContentProcessingConfig

class ConfigurationManager:
    def __init__(self, params_filepath=PARAMS_FILE_PATH):
        self.params = read_yaml(params_filepath)
        create_directories([self.params['artifacts_root']]) # <-- FIX

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.params['data_ingestion'] # <-- FIX
        
        create_directories([config['root_dir']]) # <-- FIX
        
        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config['root_dir']), # <-- FIX
            news_api_key=config['news_api_key'], # <-- FIX
            topics=config['topics'], # <-- FIX
            source_file_path=Path(config['root_dir']) / config['source_file_name'] # <-- FIX
        )
        return data_ingestion_config
    

    def get_content_processing_config(self) -> ContentProcessingConfig:
        config = self.params['content_processing']
        
        create_directories([config['root_dir']])
        
        content_processing_config = ContentProcessingConfig(
            root_dir=Path(config['root_dir']),
            processed_file_path=Path(config['root_dir']) / config['processed_file_name'],
            model_name=config['model_name'],
            google_api_key=config['google_api_key']
        )
        return content_processing_config