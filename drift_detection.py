import numpy as np
import cv2
from skimage import io
from scipy.stats import entropy
import matplotlib.pyplot as plt
import streamlit as st

# Function to load and preprocess an image
def load_and_preprocess_image(filepath):
    image = io.imread(filepath)
    if len(image.shape) == 3 and image.shape[2] == 4:  # If RGBA, convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image_normalized = image_gray / 255.0
    return image_normalized

# Function to calculate cross-entropy
def cross_entropy(image1, image2):
    epsilon = 1e-12  # To avoid log(0)
    image1 = np.clip(image1, epsilon, 1.0 - epsilon)
    image2 = np.clip(image2, epsilon, 1.0 - epsilon)
    return -np.sum(image1 * np.log(image2))

# Function to calculate KL divergence
def kl_divergence(image1, image2):
    image1_flat = image1.flatten()
    image2_flat = image2.flatten()
    return entropy(image1_flat, image2_flat)

# Streamlit app
st.title("Image Drift Detection")

# Upload training image
training_image_file = st.file_uploader("Upload Training Image", type=["tiff", "jpeg", "jpg", "png"])
# Upload incoming image
incoming_image_file = st.file_uploader("Upload Incoming Image", type=["tiff", "jpeg", "jpg", "png"])

if training_image_file and incoming_image_file:
    # Load and preprocess the images
    training_image = load_and_preprocess_image(training_image_file)
    incoming_image = load_and_preprocess_image(incoming_image_file)
    
    # Resize incoming image to match training image dimensions
    incoming_image_resized = cv2.resize(incoming_image, (training_image.shape[1], training_image.shape[0]))
    
    # Calculate drift metrics
    cross_entropy_value = cross_entropy(training_image, incoming_image_resized)
    kl_divergence_value = kl_divergence(training_image, incoming_image_resized)
    
    # Display drift metrics
    st.write(f"Cross-Entropy: {cross_entropy_value}")
    st.write(f"KL Divergence: {kl_divergence_value}")
    
    # Display images
    st.subheader("Training Image")
    st.image(training_image, caption='Training Image', use_column_width=True, channels="GRAY")
    
    st.subheader("Incoming Image")
    st.image(incoming_image_resized, caption='Incoming Image', use_column_width=True, channels="GRAY")
    
    # Check thresholds and alert
    CROSS_ENTROPY_THRESHOLD = 0.5
    KL_DIVERGENCE_THRESHOLD = 0.1
    
    if cross_entropy_value > CROSS_ENTROPY_THRESHOLD:
        st.error("Alert: Significant image drift detected based on Cross-Entropy!")
    
    if kl_divergence_value > KL_DIVERGENCE_THRESHOLD:
        st.error("Alert: Significant image drift detected based on KL Divergence!")

# Run the Streamlit app
if __name__ == '__main__':
    st.write("Upload the images to start drift detection.")
