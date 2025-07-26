import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pathlib import Path

class PredictionPipeline:
    """
    A pipeline for making predictions on the BBC News dataset.
    
    This class loads the trained model, vectorizer, and label encoder,
    preprocesses input text, and returns a predicted category name.
    """
    def __init__(self):
        """
        Initializes the PredictionPipeline by loading the trained model,
        TF-IDF vectorizer, and LabelEncoder from their saved paths.
        """
        self.model = joblib.load(Path('artifacts/model_trainer/model.pkl'))
        self.vectorizer = joblib.load(Path('artifacts/model_trainer/tfidf_vectorizer.pkl'))
        self.label_encoder = joblib.load(Path('artifacts/model_trainer/label_encoder.pkl'))
        
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
        """
        if not isinstance(text, str):
            return ""
        
        lemmatizer = WordNetLemmatizer()
        review = re.sub('[^a-zA-Z]', ' ', text)
        review = review.lower()
        review = review.split()
        review = [lemmatizer.lemmatize(word) for word in review if not word in stopwords.words('english')]
        return ' '.join(review)

    def predict(self, text: str) -> str:
        """
        Makes a prediction on a single piece of input text.

        Args:
            text (str): The raw news article text.

        Returns:
            str: The predicted category name (e.g., "tech", "sport").
        """
        processed_text = self._preprocess_text(text)
        vectorized_text = self.vectorizer.transform([processed_text])
        prediction_numeric = self.model.predict(vectorized_text)
        
        # Decode the numeric prediction back to its string label
        prediction_label = self.label_encoder.inverse_transform(prediction_numeric)[0]
        
        return prediction_label.capitalize()
