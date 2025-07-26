# Use an official Python runtime as a parent image
# Using a slim version to keep the image size smaller
FROM python:3.8-slim

# Set the working directory in the container to /app
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir ensures that pip does not store the downloaded packages, reducing image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code from your local machine to the container at /app
# This includes the 'src' directory, 'app.py', 'artifacts', etc.
COPY . .

# Make port 8080 available to the world outside this container
# This is the port our Flask app will run on
EXPOSE 8080

# Define the command to run your application
# This command runs the Flask development server.
# "0.0.0.0" means it will be accessible from outside the container.
CMD ["python", "app.py"]