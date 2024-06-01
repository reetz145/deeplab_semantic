# DeeplabV3 GeoTIFF Segmentation with Streamlit

This project uses a pre-trained DeeplabV3 model to perform image segmentation on GeoTIFF imagery through a Streamlit web application.

## Requirements

- Docker
- Git

## Setup Instructions

### Clone the Repository

```sh
git clone https://github.com/reetz145/deeplab_semantic

### Set Up Docker Image
docker build -t deeplabv3-streamlit

### Run the Docker Container

docker run -p 8501:8502 deeplabv3-streamlit
