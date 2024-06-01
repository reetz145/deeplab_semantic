import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

# Function to perform segmentation
def perform_segmentation(image, model):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        model.eval()
        outputs = model(input_tensor)
        outputs = torch.argmax(outputs, dim=1).squeeze().cpu().numpy()

    return outputs

# Load your pre-trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "deeplabv3_checkpoint.pt"
model = smp.DeepLabV3(encoder_name="resnet50", encoder_weights="imagenet", classes=5).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

st.title('Image Segmentation')

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png", "tiff", "tif"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Perform image segmentation
    segmented_output = perform_segmentation(image, model)

    # Convert the segmented output tensor to a numpy array
    cmap = plt.cm.get_cmap("tab20", segmented_output.max() + 1)  # Create a colormap with enough colors
    segmented_image = cmap(segmented_output)[:, :, :3]  # Remove alpha channel

    # Display the segmented image
    st.image(segmented_image, caption='Segmented Image.', use_column_width=True)
