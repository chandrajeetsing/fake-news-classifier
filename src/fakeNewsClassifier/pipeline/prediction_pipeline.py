import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pathlib import Path

class PredictionPipeline:
    """
    A pipeline for making predictions on new, unseen text data.
    
    This class loads the trained model and vectorizer, preprocesses the input text
    in the same way as the training data, and returns a prediction.
    """
    def __init__(self):
        """
        Initializes the PredictionPipeline by loading the trained model
        and TF-IDF vectorizer from their saved paths.
        """
        # Define paths to the saved model and vectorizer
        model_path = Path('artifacts/model_trainer/model.pkl')
        vectorizer_path = Path('artifacts/model_trainer/tfidf_vectorizer.pkl')

        # Load the model and vectorizer
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        
        # Ensure NLTK resources are available
        self._download_nltk_resources()

    def _download_nltk_resources(self):
        """Downloads necessary NLTK data files if they don't exist."""
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')

    def _preprocess_text(self, text: str) -> str:
        """
        Cleans and preprocesses a single piece of text.
        This method MUST be identical to the one in DataTransformation.

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

    def predict(self, text: str) -> str:
        """
        Makes a prediction on a single piece of input text.

        Args:
            text (str): The raw news article text.

        Returns:
            str: The prediction, either "FAKE" or "REAL".
        """
        # Preprocess the input text
        processed_text = self._preprocess_text(text)
        
        # Vectorize the processed text using the loaded vectorizer
        vectorized_text = self.vectorizer.transform([processed_text])
        
        # Make a prediction using the loaded model
        prediction = self.model.predict(vectorized_text)
        
        # Return the human-readable result
        if prediction[0] == 1:
            return "FAKE"
        else:
            return "REAL"