import torch
import streamlit as st
import base64
from model_utils import stoi, itos, generate_next_words, Next_Word_Predictor, load_pretrained_model

# Streamlit app title
st.title("Next Word Predictor")

# Function to get SVG as base64
def get_svg_as_base64(svg_file_path):
    with open(svg_file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Load the help icon SVG
help_icon_base64 = get_svg_as_base64("assets\help_icon.svg")

# Hyperparameter options
context_length_options = [5, 15]  # Example values
embedding_dim_options = [64, 128]  # Example values
activation_fn_options = ['relu', 'sigmoid']  # Example values
random_seed_options = [42, 99]  # Example values

# Create two columns for hyperparameter selection
col1, col2 = st.columns(2)

# Add dropdowns for hyperparameters with help icons next to the labels
with col1:
    st.markdown(f"**Choose Context Length:** <img src='data:image/svg+xml;base64,{help_icon_base64}' title='Number of previous tokens to consider.' width='15' height='15' style='vertical-align: middle;'>", unsafe_allow_html=True)
    context_length = st.selectbox(" ", context_length_options)

    st.markdown(f"**Choose Embedding Dimension:** <img src='data:image/svg+xml;base64,{help_icon_base64}' title='Size of the embedding vector for tokens.' width='15' height='15' style='vertical-align: middle;'>", unsafe_allow_html=True)
    embedding_dim = st.selectbox(" ", embedding_dim_options)

with col2:
    st.markdown(f"**Choose Activation Function:** <img src='data:image/svg+xml;base64,{help_icon_base64}' title='Function to introduce non-linearity in the model.' width='15' height='15' style='vertical-align: middle;'>", unsafe_allow_html=True)
    activation_fn = st.selectbox(" ", activation_fn_options)

    st.markdown(f"**Choose Random Seed:** <img src='data:image/svg+xml;base64,{help_icon_base64}' title='Seed value for random state of the model.' width='15' height='15' style='vertical-align: middle;'>", unsafe_allow_html=True)
    random_seed = st.selectbox(" ", random_seed_options)

# Input for content and k (number of words to predict)
content = st.text_input("**Enter some content:**")
k = st.number_input("**Number of words to predict:**", min_value=1, max_value=100, value=5)

model_mapping = {
    (5, 64, 'relu', 42): 'model_variants\model_variant_1.pth',
    (5, 64, 'relu', 99): 'model_variants\model_variant_2.pth',
    (5, 64, 'sigmoid', 42): 'model_variants\model_variant_3.pth',
    (5, 64, 'sigmoid', 99): 'model_variants\model_variant_4.pth',
    (5, 128, 'relu', 42): 'model_variants\model_variant_5.pth',
    (5, 128, 'relu', 99): 'model_variants\model_variant_6.pth',
    (5, 128, 'sigmoid', 42): 'model_variants\model_variant_7.pth',
    (5, 128, 'sigmoid', 99): 'model_variants\model_variant_8.pth',
    (15, 64, 'relu', 42): 'model_variants\model_variant_9.pth',
    (15, 64, 'relu', 99): 'model_variants\model_variant_10.pth',
    (15, 64, 'sigmoid', 42): 'model_variants\model_variant_11.pth',
    (15, 64, 'sigmoid', 99): 'model_variants\model_variant_12.pth',
    (15, 128, 'relu', 42): 'model_variants\model_variant_13.pth',
    (15, 128, 'relu', 99): 'model_variants\model_variant_14.pth',
    (15, 128, 'sigmoid', 42): 'model_variants\model_variant_15.pth',
    (15, 128, 'sigmoid', 99): 'model_variants\model_variant_16.pth',
}


# Button to trigger prediction
if st.button("Predict Next Words"):
    # Create the key based on selected hyperparameters
    selected_key = (context_length, embedding_dim, activation_fn, random_seed)

    # Retrieve the model path using the dictionary
    model_path = model_mapping.get(selected_key, None)

    # If model exists for the selected hyperparameters, load and predict
    if model_path:
        model = load_pretrained_model(model_path)
        model.eval()
        if model:  # Proceed only if the model was successfully loaded
            para = generate_next_words(model, itos, stoi, content, 42, k)
            st.subheader("Content with Predicted Next Words")
            st.write(para)
    else:
        st.error("No model found for the selected hyperparameter combination.")