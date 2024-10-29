import time
import streamlit as st
import base64
from PIL import Image
from model_utils import stoi, itos, generate_next_words, Next_Word_Predictor, load_pretrained_model
import warnings

# Suppressing all warnings
warnings.filterwarnings("ignore")

# Loading Image using PIL
im = Image.open('assets\App-Icon.png')
st.set_page_config(page_title="Next Word Predictor App", page_icon = im)

# Streamlit app title
st.title("Next Word Predictor")

# Function to get SVG as base64
def get_svg_as_base64(svg_file_path):
    with open(svg_file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Loading the help icon SVG
help_icon_base64 = get_svg_as_base64("assets\help_icon.svg")

# Hyperparameter options
context_length_options = [5, 15]  
embedding_dim_options = [64, 128]  
activation_fn_options = ['relu', 'sigmoid']  
random_seed_options = [42, 99]  

# Two columns for hyperparameter selection
col1, col2 = st.columns(2)

# Adding dropdowns for hyperparameters
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

# Asking for the temperature of predictions
st.markdown(f"**Choose Temperature:** <img src='data:image/svg+xml;base64,{help_icon_base64}' title='Controls the randomness of predictions: lower values make the output more deterministic, while higher values increase diversity.' width='15' height='15' style='vertical-align: middle;'>", unsafe_allow_html=True)
temperature = st.slider(" ", min_value=0.1, max_value=2.0, value=1.0, step=0.1)

# Input for content and k (number of words to predict)
content = st.text_input("**Enter some content:**")
k = st.number_input("**Number of words to predict:**", min_value=1, max_value=100, value=10)

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
    selected_key = (context_length, embedding_dim, activation_fn, random_seed)
    model_path = model_mapping.get(selected_key, None)

    if model_path:
        model = load_pretrained_model(model_path)
        model.eval()
        if model:  # Proceed only if the model was successfully loaded
            para = generate_next_words(model, itos, stoi, content, 42, k, temperature)
            st.subheader("Content with Predicted Next Words")
            # Placeholder for typewriter effect
            placeholder = st.empty()

            original_length = len(content)  # Length of original content
            typed_text = ""
            highlighted_text = ""

            for char in para:
                typed_text += char
                # Highlight predicted text after the original content
                if len(typed_text) > original_length:
                    highlighted_text = f"<span style='color: #ff4b4b; font-weight: semi-bold;'>{typed_text[original_length:]}</span>"
                    display_text = content + highlighted_text
                else:
                    display_text = content[:len(typed_text)]
                
                placeholder.markdown(display_text, unsafe_allow_html=True)  
                time.sleep(0.02)  # Adding a small delay for typewriter effect
    else:
        st.error("No model found for the selected hyperparameter combination.")