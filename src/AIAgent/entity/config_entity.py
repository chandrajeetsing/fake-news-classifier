# src/AIAgent/entity/config_entity.py
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    news_api_key: str
    topics: list
    source_file_path: Path

# Add this class to the end of the file
@dataclass(frozen=True)
class ContentProcessingConfig:
    root_dir: Path
    processed_file_path: Path
    model_name: str
    google_api_key: str

# Define ContentProcessingConfig and ContentPublishingConfig here later