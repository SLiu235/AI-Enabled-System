# Use an official Python runtime as the base image
FROM python:3.8-slim

# Set the working directory 
WORKDIR /app

# Copy the current directory 
COPY . /app

# Copy the requirements
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5001 available to the world outside this container
EXPOSE 5001

# Run app.py when the container launches
CMD ["python", "app.py"]