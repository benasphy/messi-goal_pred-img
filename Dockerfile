# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the application code
COPY main1.py ./

# Install the required libraries
RUN pip install numpy --progress-bar off


# Run the application
CMD ["python", "main1.py"]