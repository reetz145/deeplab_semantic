# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install Streamlit, PyTorch, and other dependencies
RUN pip install --no-cache-dir streamlit pandas numpy torch

# Define environment variables to prevent Streamlit from opening a browser
ENV STREAMLIT_SERVER_HEADLESS=true

# Run a simple script to check if Streamlit is installed correctly
RUN python -c "import streamlit; print('Streamlit version:', streamlit.__version__)"

# Define a default command that will be overridden in the docker-compose.yml
CMD ["streamlit", "run"]
