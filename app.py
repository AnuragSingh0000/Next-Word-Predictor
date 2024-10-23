import torch
import streamlit as st
import tensorflow as tf  # Import TensorFlow for Keras models
from Task_1 import stoi, itos, generate_next_words, device

# Streamlit app title
st.title("Next Word Predictor")

# Select the model variant
model_variant = st.selectbox(
    "Choose a Pre-trained Model Variant:",
    ("Model Variant 1")  # Add more variants as needed in the tuple
)

# Input for content and k (number of words to predict)
content = st.text_input("Enter some content:")
k = st.number_input("Number of words to predict:", min_value=1, max_value=20, value=5)


def load_pretrained_model(variant):
    """Load a pre-trained model based on the user's selection."""
    model = None
    if variant == 'Model Variant 1':
        # block_size = 5; emb_dim = 64; hidden_dim = 1024; activation_fn = 'relu'; seed_value = 42
        
        # Load the model
        model = torch.load('model_variant_1.pth', map_location=device)

        
    # Add more variants as needed
    # if variant == 'Model Variant 2':
    #     Load Model Variant 2...

    model.eval()
    return model

# Button to trigger prediction
if st.button("Predict Next Words"):
    # Load the selected pre-trained model
    model = load_pretrained_model(model_variant)

    # Generate predictions
    para = generate_next_words(model, itos, stoi, content, 1, k)
    st.subheader("Content with Predicted Next Words")
    st.write(para)
