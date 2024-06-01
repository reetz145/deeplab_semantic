# DeeplabV3 GeoTIFF Segmentation with Streamlit

This project uses a pre-trained DeeplabV3 model to perform image segmentation on GeoTIFF imagery through a Streamlit web application.

## Requirements

- Docker
- Git

## Setup Instructions

### Clone the Repository and Run the Docker container

```sh
git clone https://github.com/reetz145/deeplab_semantic

docker build -t deeplabv3-streamlit

docker run -p 8501:8502 deeplabv3-streamlit
