import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import joblib
from fakeNewsClassifier.logging import logger
from fakeNewsClassifier.entity.config_entity import ModelTrainerConfig

class ModelTrainer:
    """
    Trains the machine learning model and saves it.
    
    This component handles the training of the text classification model.
    It uses TF-IDF to vectorize the text data and trains a Passive-Aggressive
    Classifier. The trained model and the vectorizer are then saved to disk.
    """
    def __init__(self, config: ModelTrainerConfig):
        """
        Initializes the ModelTrainer component.

        Args:
            config (ModelTrainerConfig): Configuration for model training.
        """
        self.config = config

    def train(self, train_data_path: str, test_data_path: str):
        """
        Executes the model training process.

        Args:
            train_data_path (str): Path to the training data CSV.
            test_data_path (str): Path to the testing data CSV.
        """
        try:
            logger.info("Starting model training process.")
            
            # Load the datasets
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)
            logger.info("Loaded training and testing data.")

            # Drop rows with missing content
            train_df.dropna(subset=['content'], inplace=True)
            test_df.dropna(subset=['content'], inplace=True)

            # Prepare the data
            X_train = train_df['content']
            y_train = train_df['label']
            X_test = test_df['content']
            y_test = test_df['label']
            
            # Initialize TF-IDF Vectorizer
            tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
            
            # Fit and transform the training data, transform the test data
            tfidf_train = tfidf_vectorizer.fit_transform(X_train)
            tfidf_test = tfidf_vectorizer.transform(X_test)
            logger.info("Applied TF-IDF vectorization to the data.")
            
            # Initialize and train the Passive-Aggressive Classifier
            pac = PassiveAggressiveClassifier(max_iter=50)
            pac.fit(tfidf_train, y_train)
            logger.info("Model training complete.")
            
            # Save the trained model and the vectorizer
            joblib.dump(pac, self.config.trained_model_file_path)
            joblib.dump(tfidf_vectorizer, self.config.vectorizer_file_path)
            logger.info(f"Saved trained model to: {self.config.trained_model_file_path}")
            logger.info(f"Saved TF-IDF vectorizer to: {self.config.vectorizer_file_path}")
            
            logger.info("Model training process finished successfully.")

        except Exception as e:
            logger.error(f"An error occurred during model training: {e}")
            raise e