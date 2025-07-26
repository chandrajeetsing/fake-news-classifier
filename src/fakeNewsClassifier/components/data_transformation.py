import pandas as pd
import re
import nltk
import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from fakeNewsClassifier.logging import logger
from fakeNewsClassifier.entity.config_entity import DataTransformationConfig

class DataTransformation:
    """
    Transforms the raw text data into a format suitable for model training.
    
    This includes cleaning the text, encoding the categorical labels, and
    splitting the data into training and testing sets.
    """
    def __init__(self, config: DataTransformationConfig):
        """
        Initializes the DataTransformation component.

        Args:
            config (DataTransformationConfig): Configuration for data transformation.
        """
        self.config = config
        self._download_nltk_resources()

    def _download_nltk_resources(self):
        """Downloads necessary NLTK data files if they don't exist."""
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
        review = re.sub('[^a-zA-Z]', ' ', text)
        review = review.lower()
        review = review.split()
        review = [lemmatizer.lemmatize(word) for word in review if not word in stopwords.words('english')]
        return ' '.join(review)

    def transform_data(self, data_path: str):
        """
        Main method to execute the data transformation process.

        Args:
            data_path (str): Path to the input data CSV file.
        """
        logger.info("Starting data transformation process for BBC data.")
        try:
            df = pd.read_csv(data_path)
            
            df.dropna(subset=['text', 'category'], inplace=True)
            logger.info("Dropped rows with missing text or category.")
            
            # Apply text preprocessing
            logger.info("Applying text preprocessing to the 'text' column...")
            df['text'] = df['text'].apply(self._preprocess_text)
            logger.info("Text preprocessing complete.")

            # Encode the 'category' column
            encoder = LabelEncoder()
            df['category_encoded'] = encoder.fit_transform(df['category'])
            logger.info("Applied Label Encoding to the 'category' column.")
            
            # Save the fitted LabelEncoder
            joblib.dump(encoder, self.config.label_encoder_path)
            logger.info(f"Saved LabelEncoder to: {self.config.label_encoder_path}")

            # Splitting the data
            X = df['text']
            y = df['category_encoded']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            logger.info("Split data into training and testing sets.")

            # Save the transformed data
            train_df = pd.DataFrame({'text': X_train, 'label': y_train})
            test_df = pd.DataFrame({'text': X_test, 'label': y_test})

            train_df.to_csv(self.config.transformed_data_path, index=False)
            test_df.to_csv(self.config.test_data_path, index=False)
            
            logger.info(f"Saved training data to: {self.config.transformed_data_path}")
            logger.info(f"Saved testing data to: {self.config.test_data_path}")
            logger.info("Data transformation process finished successfully.")

        except Exception as e:
            logger.error(f"An error occurred during data transformation: {e}")
            raise e