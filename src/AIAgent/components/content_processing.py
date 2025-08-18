# src/AIAgent/components/content_processing.py
import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from AIAgent.entity.config_entity import ContentProcessingConfig, DataIngestionConfig
from AIAgent.logging import logger

class ContentProcessing:
    def __init__(self, processing_config: ContentProcessingConfig, ingestion_config: DataIngestionConfig):
        self.processing_config = processing_config
        self.ingestion_config = ingestion_config
        # self.llm = ChatOpenAI(model=self.processing_config.model_name)
        self.llm = ChatGoogleGenerativeAI(
            model=self.processing_config.model_name,
            google_api_key=self.processing_config.google_api_key,
            convert_system_message_to_human=True # Helps with some models
        )

    def process_content(self):
        logger.info("Starting content processing...")
        
        # 1. Read the raw data
        try:
            with open(self.ingestion_config.source_file_path, 'r') as f:
                articles = json.load(f)
        except FileNotFoundError:
            logger.error(f"Raw data file not found at {self.ingestion_config.source_file_path}. Aborting.")
            return

        processed_articles = []

        # 2. Define the AI prompt and parser
        parser = JsonOutputParser()
        prompt = ChatPromptTemplate.from_template(
            """
            Analyze the following news article content. Your goal is to act as an expert news editor.
            Extract the key information and provide it strictly in the following JSON format:
            {{
                "region": "The main city or district discussed (e.g., Lucknow, Noida, Varanasi)",
                "category": "The primary category (e.g., Politics, Crime, Business, Sports, Technology, Other)",
                "summary": "A concise, neutral summary of the article in 2-3 sentences.",
                "seo_title": "An engaging, SEO-friendly title under 70 characters.",
                "tags": ["A list of 3-5 relevant lowercase keywords"]
            }}

            ARTICLE CONTENT:
            ```{content}```
            """
        )
        chain = prompt | self.llm | parser

        # 3. Loop through articles and process them
        for i, article in enumerate(articles):
            # We only process the first 5 for this example to save on API calls during testing
            if i >= 5:
                logger.info("Processed the first 5 articles for testing. Stopping.")
                break
            
            logger.info(f"Processing article {i+1}/{len(articles)}: {article.get('title')}")
            try:
                content = article.get('content') or article.get('description', '')
                if not content:
                    logger.warning("Article has no content or description. Skipping.")
                    continue
                
                ai_response = chain.invoke({"content": content})
                
                # Combine original data with AI response
                final_article_data = {
                    "original_url": article.get('url'),
                    "original_title": article.get('title'),
                    **ai_response # Unpack the AI's JSON response here
                }
                processed_articles.append(final_article_data)

            except Exception as e:
                logger.error(f"Failed to process article {article.get('title')}: {e}")

        # 4. Save the processed data
        with open(self.processing_config.processed_file_path, 'w') as f:
            json.dump(processed_articles, f, indent=4)
        
        logger.info(f"Successfully processed and saved {len(processed_articles)} articles.")