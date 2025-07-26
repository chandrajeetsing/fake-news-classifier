from flask import Flask, render_template
from collections import Counter
from fakeNewsClassifier.pipeline.prediction_pipeline import PredictionPipeline
from fakeNewsClassifier.components.web_scraper import WebScraper
from fakeNewsClassifier.logging import logger

# Initialize the Flask application
app = Flask(__name__, template_folder='app/templates', static_folder='app/static')

@app.route('/', methods=['GET'])
def home():
    """
    Scrapes the latest news, classifies each headline, and displays a summary.
    """
    try:
        logger.info("Request received for home page.")
        
        # 1. Scrape latest headlines
        scraper = WebScraper()
        headlines_data = scraper.get_latest_headlines(limit=20)
        
        if not headlines_data:
            return render_template('index.html', error="Could not scrape any headlines. The website layout may have changed.")

        # 2. Classify each headline
        prediction_pipeline = PredictionPipeline()
        classified_articles = []
        all_categories = []

        for item in headlines_data:
            headline = item['headline']
            predicted_category = prediction_pipeline.predict(headline)
            classified_articles.append({'headline': headline, 'category': predicted_category})
            all_categories.append(predicted_category)
        
        # 3. Summarize the topics
        topic_summary = Counter(all_categories)
        
        logger.info(f"Topic summary: {topic_summary}")

        return render_template('index.html', articles=classified_articles, summary=topic_summary)

    except Exception as e:
        logger.error(f"An error occurred on the home page: {e}")
        return render_template('index.html', error=f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)