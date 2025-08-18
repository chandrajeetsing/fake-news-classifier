# app.py
import json
from flask import Flask, render_template
from pathlib import Path
from apscheduler.schedulers.background import BackgroundScheduler
from AIAgent.pipeline.agent_pipeline import AgentPipeline
from AIAgent.logging import logger
import atexit

# Initialize the Flask app
app = Flask(__name__)

def run_agent_job():
    """Function to run the AI agent pipeline."""
    logger.info("Scheduler is running the agent pipeline...")
    try:
        pipeline = AgentPipeline()
        pipeline.run()
        logger.info("Agent pipeline finished its scheduled run.")
    except Exception as e:
        logger.exception(f"Error during scheduled agent run: {e}")

@app.route('/')
def homepage():
    # ... (this function stays exactly the same)
    processed_data_path = Path("artifacts/content_processing/processed_data.json")
    articles = []
    if processed_data_path.exists():
        with open(processed_data_path, 'r', encoding='utf-8') as f:
            articles = json.load(f)
    return render_template('index.html', articles=articles)

if __name__ == "__main__":
    # Run the agent once on startup
    run_agent_job()

    # Set up the scheduler to run the agent every hour
    scheduler = BackgroundScheduler()
    scheduler.add_job(func=run_agent_job, trigger="interval", hours=1)
    scheduler.start()

    # Shut down the scheduler when the app exits
    atexit.register(lambda: scheduler.shutdown())

    # Run the web server
    app.run(host="0.0.0.0", port=5000)