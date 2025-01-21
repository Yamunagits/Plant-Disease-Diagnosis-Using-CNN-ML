import streamlit as st
from PIL import Image, ImageOps
import tensorflow as tf
import numpy as np

# Load the pre-trained plant disease detection model
model = tf.keras.models.load_model("keras_model.h5")

# Define the classes for plant diseases and their corresponding pesticides
class_names = ['Bacterial', 'Fungal', 'Rust', 'Virus']
disease_to_pesticide = {
    'Bacterial': 'Pseudomonas fluorescens',
    'Fungal': 'sulfur-based fungicides',
    'Rust': 'Propiconazole + Difenoconazole',
    'Virus': 'Neonicotinoids'
}

st.title('Plant Disease Detection and Pesticide Recommendation')

img_file = st.file_uploader('Upload plant image', type=['png', 'jpg', 'jpeg'])

def load_img(img):
    img = Image.open(img)
    return img

if img_file is not None:
    file_details = {}
    file_details['name'] = img_file.name
    file_details['size'] = img_file.size
    file_details['type'] = img_file.type
    st.write(file_details)
    st.image(load_img(img_file), width=255)

    # Save the uploaded image
    with open('uploads/plant_img.jpg', 'wb') as f:
        f.write(img_file.getbuffer())

    st.success('Image Saved')

    # Load the saved image
    image = Image.open('uploads/plant_img.jpg').convert("RGB")

    # Preprocess the image
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 255.0)  # Normalize to [0,1]
    data = np.expand_dims(normalized_image_array, axis=0)

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]

    # Display the predicted plant disease
    st.success(f"Predicted Plant Disease: {class_name}")

    # Recommend pesticide based on predicted disease
    if class_name in disease_to_pesticide:
        pesticide = disease_to_pesticide[class_name]
        st.info(f"Recommended Pesticide: {pesticide}")
    else:
        st.warning("Pesticide recommendation not available for this disease.")
