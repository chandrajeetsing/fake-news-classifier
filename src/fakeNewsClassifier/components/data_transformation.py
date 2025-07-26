import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from fakeNewsClassifier.logging import logger
from fakeNewsClassifier.entity.config_entity import DataTransformationConfig

class DataTransformation:
    """
    Transforms the raw text data into a format suitable for model training.
    
    This includes cleaning the text (removing stopwords, punctuation, etc.) and
    splitting the data into training and testing sets.
    """
    def __init__(self, config: DataTransformationConfig):
        """
        Initializes the DataTransformation component.

        Args:
            config (DataTransformationConfig): Configuration for data transformation.
        """
        self.config = config
        # Download NLTK resources if not already present
        self._download_nltk_resources()

    def _download_nltk_resources(self):
        """Downloads necessary NLTK data files."""
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            logger.info("Downloading NLTK stopwords...")
            nltk.download('stopwords')
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            logger.info("Downloading NLTK wordnet...")
            nltk.download('wordnet')

    def _preprocess_text(self, text: str) -> str:
        """
        Cleans and preprocesses a single piece of text.

        Args:
            text (str): The input text to clean.

        Returns:
            str: The cleaned text.
        """
        if not isinstance(text, str):
            return ""
        
        lemmatizer = WordNetLemmatizer()
        # Remove non-alphabetic characters
        review = re.sub('[^a-zA-Z]', ' ', text)
        review = review.lower()
        review = review.split()
        
        # Lemmatize and remove stopwords
        review = [lemmatizer.lemmatize(word) for word in review if not word in stopwords.words('english')]
        return ' '.join(review)

    def transform_data(self, data_path: str):
        """
        Main method to execute the data transformation process.

        Args:
            data_path (str): Path to the input data CSV file.
        """
        logger.info("Starting data transformation process.")
        try:
            df = pd.read_csv(data_path)
            
            # Drop rows with missing values in 'text' or 'title'
            df.dropna(subset=['text', 'title'], inplace=True)
            logger.info("Dropped rows with missing text or title.")
            
            # Combine title and text for a richer feature set
            df['content'] = df['title'] + ' ' + df['text']
            logger.info("Combined 'title' and 'text' into 'content' column.")

            # Apply preprocessing
            logger.info("Applying text preprocessing to the 'content' column...")
            df['content'] = df['content'].apply(self._preprocess_text)
            logger.info("Text preprocessing complete.")

            # Splitting the data
            X = df['content']
            y = df['label']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            logger.info("Split data into training and testing sets.")

            # Save the transformed data
            train_df = pd.DataFrame({'content': X_train, 'label': y_train})
            test_df = pd.DataFrame({'content': X_test, 'label': y_test})

            train_df.to_csv(self.config.transformed_data_path, index=False)
            test_df.to_csv(self.config.test_data_path, index=False)
            
            logger.info(f"Saved training data to: {self.config.transformed_data_path}")
            logger.info(f"Saved testing data to: {self.config.test_data_path}")
            logger.info("Data transformation process finished successfully.")

        except Exception as e:
            logger.error(f"An error occurred during data transformation: {e}")
            raise e