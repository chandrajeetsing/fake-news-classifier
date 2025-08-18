# src/AIAgent/components/data_ingestion.py
import requests
import json
from AIAgent.entity.config_entity import DataIngestionConfig
from AIAgent.logging import logger

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        """
        Initializes the DataIngestion component with its configuration.
        """
        self.config = config

    def fetch_and_save_data(self):
        """
        Fetches articles from NewsAPI for each topic and saves them to a single JSON file.
        """
        logger.info(f"Starting data ingestion for topics: {self.config.topics}")
        all_articles = []
        
        for topic in self.config.topics:
            logger.info(f"Fetching articles for topic: '{topic}'")
            try:
                # Construct the API URL
                url = f"https://newsapi.org/v2/everything?q={topic}&apiKey={self.config.news_api_key}"
                
                # Make the API request
                response = requests.get(url)
                response.raise_for_status()  # This will raise an HTTPError for bad responses (4xx or 5xx)

                # Parse the JSON response and add articles to the list
                articles = response.json().get('articles', [])
                logger.info(f"Found {len(articles)} articles for topic: '{topic}'")
                all_articles.extend(articles)

            except requests.exceptions.RequestException as e:
                logger.error(f"API request failed for topic '{topic}': {e}")
                continue # Continue to the next topic even if one fails
        
        # Save all collected articles to the specified file
        try:
            with open(self.config.source_file_path, 'w', encoding='utf-8') as f:
                json.dump(all_articles, f, indent=4, ensure_ascii=False)
            logger.info(f"Successfully saved a total of {len(all_articles)} articles to {self.config.source_file_path}")
        except IOError as e:
            logger.error(f"Failed to write articles to file: {e}")