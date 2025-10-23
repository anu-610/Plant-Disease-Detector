# Use a lightweight official Python image suitable for production
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED 1
ENV PORT 8000 # Azure typically uses port 8000 for web apps

# Set the working directory inside the container
WORKDIR /app

# 1. Copy requirements and install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy the rest of your application code
# Your app.py contains the logic to download the model
COPY . /app/
RUN mkdir -p /app/uploads

# 3. Define the command to run the application using Gunicorn
# This tells Azure to run the 'app' object found in 'app.py'
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]