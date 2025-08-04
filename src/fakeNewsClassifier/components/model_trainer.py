import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
from pathlib import Path
from fakeNewsClassifier.logging import logger
from fakeNewsClassifier.entity.config_entity import ModelTrainerConfig, DataTransformationConfig

class ModelTrainer:
    """
    Trains the machine learning model for multi-class text classification.
    
    This component uses TF-IDF to vectorize the text data and trains a
    Logistic Regression model. The trained model and the vectorizer are
    then saved to disk.
    """
    def __init__(self, config: ModelTrainerConfig):
        """
        Initializes the ModelTrainer component.

        Args:
            config (ModelTrainerConfig): Configuration for model training.
        """
        self.config = config

    def train(self, train_data_path: str, test_data_path: str, data_transformation_config: DataTransformationConfig):
        """
        Executes the model training process.

        Args:
            train_data_path (str): Path to the training data CSV.
            test_data_path (str): Path to the testing data CSV.
        """
        try:
            logger.info("Starting model training process for BBC data.")
            
            # Load the datasets
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)
            logger.info("Loaded training and testing data.")

            # Drop rows with missing content
            train_df.dropna(subset=['text', 'label'], inplace=True)
            test_df.dropna(subset=['text', 'label'], inplace=True)

            # Prepare the data
            X_train = train_df['text']
            y_train = train_df['label']
            X_test = test_df['text']
            y_test = test_df['label']
            
            # Initialize TF-IDF Vectorizer
            # Using sublinear_tf=True can be effective for text data
            tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.8, sublinear_tf=True)
            
            # Fit and transform the training data, transform the test data
            tfidf_train = tfidf_vectorizer.fit_transform(X_train)
            tfidf_test = tfidf_vectorizer.transform(X_test)
            logger.info("Applied TF-IDF vectorization to the data.")
            
            # Initialize and train the Logistic Regression model
            # Multi-class is handled automatically by LogisticRegression
            lr_model = LogisticRegression(random_state=42, solver='liblinear')
            lr_model.fit(tfidf_train, y_train)
            logger.info("Model training complete.")
            
            # Save the trained model and the vectorizer
            joblib.dump(lr_model, self.config.trained_model_file_path)
            joblib.dump(tfidf_vectorizer, self.config.vectorizer_file_path)
            logger.info(f"Saved trained model to: {self.config.trained_model_file_path}")
            logger.info(f"Saved TF-IDF vectorizer to: {self.config.vectorizer_file_path}")
            
            logger.info("Model training process finished successfully.")


            # --- Copy the Label Encoder ---
            # Load the encoder from the data transformation artifacts
            encoder_to_copy = joblib.load(data_transformation_config.label_encoder_path)
            # Define the destination path within the model_trainer artifacts directory
            final_encoder_path = Path(self.config.root_dir) / "label_encoder.pkl"
            # Save a copy to the model_trainer directory
            joblib.dump(encoder_to_copy, final_encoder_path)
            logger.info(f"Copied label encoder to: {final_encoder_path}")

        except Exception as e:
            logger.error(f"An error occurred during model training: {e}")
            raise e