import torch
import torch.nn as nn
import re
import pandas as pd
import streamlit as st
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
file_path = os.path.join('assets', 'leo-tolstoy-war-and-peace.txt')


with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()

filtered_text = re.sub(r'-', ' ', text)
filtered_text = re.sub('[^a-zA-Z0-9 \.\n]', '', filtered_text)
filtered_text = filtered_text.lower()

lines=filtered_text.split(".")
words=['.']
for l in lines:
    for w in l.split():
        if (len(w)>0):
            words.append(w)

words=list(pd.Series(words).unique())


stoi={s:i for i,s in enumerate(words)}
itos={i:s for s,i in stoi.items()}


# Generate names from trained model

def generate_next_words(model, itos, stoi, content, seed_value, k, temperature=1.0, max_len=10):
    # torch.manual_seed(seed_value)
    
    block_size = model.block_size
    context = content.lower()
    context = re.sub('[^a-zA-Z0-9 \.]', '', context)
    context = re.sub('\.', ' . ', context)
    word_c = context.split()
    context = []
    for i in range(len(word_c)):
        try:
            if stoi[word_c[i]]:
                context.append(word_c[i])
        except:
            context = [stoi[w] for w in context]
            if len(context) <= block_size:
                context = [0] * (block_size - len(context)) + context
            elif len(context) > block_size:
                context = context[-block_size:]
            x = torch.tensor(context).view(1, -1).to(device)
            y_pred = model(x)
            logits = y_pred
            logits = logits/temperature

            ix = torch.distributions.categorical.Categorical(logits=logits).sample().item()
            word = itos[ix]
            content += " " + word
            context = context [1:] + [ix]
            context = [itos[w] for w in context]
            
    context = [stoi[w] for w in context]
               
    if len(context) <= block_size:
        context = [0] * (block_size - len(context)) + context
    elif len(context) > block_size:
        context = context[-block_size:]

    for i in range(k):
        x = torch.tensor(context).view(1, -1).to(device)
        y_pred = model(x)
        logits = y_pred
        logits = logits/temperature
        ix = torch.distributions.categorical.Categorical(logits=logits).sample().item()
        word = itos[ix]
        content += " " + word
        context = context [1:] + [ix]
        
    return content

class Next_Word_Predictor(nn.Module):
    def __init__(self, block_size, vocab_size, emb_dim, hidden_dim, activation_fn, seed_value):
        super().__init__()
        self.block_size = block_size
        self.hyperparams = {'block_size':self.block_size, 'emb_dim':emb_dim, 'hidden_dim':hidden_dim, 'activation_fn':activation_fn, 'seed_value':seed_value}
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.linear1 = nn.Linear(block_size * emb_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, vocab_size)
         
        if activation_fn == 'sigmoid':
            self.activation = torch.sigmoid  
        else:
            self.activation = torch.relu 

    def forward(self, x):
        # Embedding layer
        x = self.emb(x)
        x = x.view(x.shape[0], -1)  
        
        # Hidden layer
        x = self.linear1(x)
        x = self.activation(x)
        
        # Output layer
        x = self.linear2(x)
        
        return x


def load_pretrained_model(model_path):
    """Load a pre-trained model from the specified path."""
    try:
        model = torch.load(model_path, map_location=device)  # Load the entire model
        model.eval()  # Set model to evaluation mode
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please check the model path.")
    except Exception as e:
        st.error(f"An error occurred while loading the model: {str(e)}")