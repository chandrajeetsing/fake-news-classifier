import os
import sys
import logging

# Define the logging format string
# Example: [ 2023-07-26 10:30:00,123 ] INFO - module_name: log message
logging_str = "[ %(asctime)s ] %(levelname)s - %(module)s: %(message)s"

# Define the directory and file path for logs
log_dir = "logs"
log_filepath = os.path.join(log_dir, "running_logs.log")
os.makedirs(log_dir, exist_ok=True)

# Basic configuration for the logging system
logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[
        logging.FileHandler(log_filepath), # Log to a file
        logging.StreamHandler(sys.stdout)  # Log to the console
    ]
)

# Create a logger instance that can be imported and used across the project
logger = logging.getLogger("fakeNewsClassifierLogger")
