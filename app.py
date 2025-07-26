from flask import Flask, render_template, request
import os
from fakeNewsClassifier.pipeline.prediction_pipeline import PredictionPipeline

# Initialize the Flask application
app = Flask(__name__, template_folder='app/templates', static_folder='app/static')

# Route for the home page
@app.route('/', methods=['GET'])
def home():
    """Renders the home page."""
    return render_template('index.html')

# Route for handling the prediction
@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives text from the form, uses the prediction pipeline to get a result,
    and renders the result on the index page.
    """
    try:
        # Get the text from the form
        text = request.form['text']
        
        # Initialize the prediction pipeline
        prediction_pipeline = PredictionPipeline()
        
        # Get the prediction
        result = prediction_pipeline.predict(text)
        
        # Render the page with the prediction result
        return render_template('index.html', prediction=str(result))

    except Exception as e:
        # In case of an error, render the page with an error message
        return render_template('index.html', prediction=f"Error: {e}")

# This block allows the app to be run directly
if __name__ == "__main__":
    # Use 0.0.0.0 to make it accessible on your local network
    app.run(host="0.0.0.0", port=8080, debug=True)