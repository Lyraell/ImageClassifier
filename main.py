#IN ORDER TO RUN THIS YOU NEED TO HAVE uv INSTALLED
# IN THE TERMINAL CO TO THE MAIN.PY FILE PATH AND THEN SAY 'RUN STREAMLIT RUN MAIN.PY'
#THAT WILL RUN THE PROJECT IN AN EXTERNAL BROWSER WINDOW

import cv2
import numpy as np
import streamlit as st
# from PIL.ImageFont import Layout # This import is not used and can be removed
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import os

# --- Configuration for finding classes ---
# This path should point to your 'dataset' directory created by prepare_dataset.py
DATASET_BASE_DIR = 'C:/Users/Jessi W/PycharmProjects/ImageClassifier/dataset'

# --- Dynamically load your breed classes ---
def get_dog_breed_classes(dataset_base_dir):
    """
    Reads the folder names from the 'training' subdirectory to get the class names.
    These should be the WordNet IDs (e.g., 'n02085936').
    """
    training_dir = os.path.join(dataset_base_dir, 'training')
    if not os.path.isdir(training_dir):
        # We can't use st.error here directly as it might violate set_page_config rule
        # but the logic that calls this function will handle the UI display
        return []

    classes = sorted([d for d in os.listdir(training_dir) if os.path.isdir(os.path.join(training_dir, d))])

    if not classes:
        # Same here, UI display will be handled by the caller
        return []

    return classes

# We will now load CLASS_NAMES inside main() after set_page_config

def load_custom_model():
    # Load your own trained model file
    # Ensure this model is trained on the classes derived from the Stanford Dogs dataset
    model = load_model("breed_classifier_model.h5")
    return model

def preprocess_image(image):
    img = np.array(image)
    # Ensure the image has 3 channels (RGB)
    if img.ndim == 2: # Handle grayscale images
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4: # RGBA to RGB
        img = img[..., :3]
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

def classify_image(model, image, class_names): # Pass class_names as argument
    try:
        if not class_names:
            st.error("Breed classes not loaded. Cannot classify.")
            return None

        processed_image = preprocess_image(image)
        raw_predictions = model.predict(processed_image)[0] # Get the prediction array

        predictions = []
        for i, score in enumerate(raw_predictions):
            if i < len(class_names):
                predictions.append((class_names[i], float(score)))
            else:
                st.warning(f"Warning: Prediction index {i} out of bounds for CLASS_NAMES (len {len(class_names)}).")
                break

        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions

    except Exception as e:
        st.error(f"Error classifying image: {str(e)}")
        st.error("Please ensure your model is trained with the same number of output classes as your `CLASS_NAMES` list.")
        return None

def main():
    # st.set_page_config MUST be the first Streamlit command called
    st.set_page_config(page_title="Dog Breed Classifier", page_icon="🐾", layout="centered")

    st.title("🐾 Dog Breed Classifier")
    st.write("Upload an image of a dog to identify its breed.")

    # --- Load classes here, after set_page_config ---
    class_names_loaded = get_dog_breed_classes(DATASET_BASE_DIR)

    # Display error messages if classes couldn't be loaded
    if not class_names_loaded:
        st.error(f"Error: Training directory '{os.path.join(DATASET_BASE_DIR, 'training')}' not found or empty.")
        st.error("Please run prepare_dataset.py first to create the dataset structure and ensure it contains breed folders.")
        model = None # Prevent model loading if classes are missing
    else:
        # Pass class_names_loaded to load_cached_model or directly use it
        @st.cache_resource
        def load_cached_model_with_classes():
            return load_custom_model()
        model = load_cached_model_with_classes()

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        btn = st.button("Classify Breed")

        if btn:
            if model is None:
                st.error("Model could not be loaded due to missing class information. Please check setup.")
            else:
                with st.spinner("Analyzing Breed..."):
                    # Open image and ensure it's RGB
                    image = Image.open(uploaded_file)
                    predictions = classify_image(model, image, class_names_loaded) # Pass class_names

                    if predictions:
                        st.subheader("Top Predictions")
                        num_predictions_to_show = min(5, len(predictions))
                        for label, score in predictions[:num_predictions_to_show]:
                            st.write(f"**{label}**: {score:.2%}")
                        st.info("Note: Breed names are WordNet IDs (e.g., n02085936).")
                        st.markdown("You can find a mapping for these IDs [here](https://image-net.org/challenges/LSVRC/2012/supercategories.php) (search for 'dog' and then click on specific breeds to see their WordNet IDs).")


if __name__ == "__main__":
    main()