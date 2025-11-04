import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import pygments
from pygments import lexers, formatters, highlight

st.set_page_config(page_title="Code Generator", page_icon="🧩", layout="wide")
data = torch.load("data/kernel_dataset.pt")
token_to_id = data["token_to_id"]
id_to_token = data["id_to_token"]
vocab_size = 35833

data = torch.load("data/kernel_dataset.pt")

# ---------- Model ----------
class MLPcodegen(nn.Module):
    def __init__(self,vocab_size, context_window=5, embedding_dim=64, hidden_size=512, activation="ReLU"):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        self.fc1 = nn.Linear(context_window*embedding_dim,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.act = nn.ReLU() if activation.lower() == "ReLU" else nn.Tanh()
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(hidden_size,vocab_size)
    
    def forward(self,x):
        embeds = self.embedding(x)
        embeds = embeds.view(embeds.size(0),-1)
        hidden = self.drop(self.act(self.fc1(embeds)))
        hidden = self.drop(self.act(hidden))
        logits = self.out(hidden)
        return logits


def generate_code(model, start_text, token_to_id, id_to_token, max_new_tokens=30, context_window=5, device="cuda", strategy="Greedy"):
    model.eval()
    # Convert start text to tokens
    tokens = start_text.strip().split()
    context = [token_to_id.get(t, 0) for t in tokens]  # unknown tokens → 0
    
    generated = tokens.copy()
    
    for _ in range(max_new_tokens):
        # Take the last `context_window` tokens
        x = context[-context_window:]
        if len(x) < context_window:
            x = [0] * (context_window - len(x)) + x  # pad left with zeros
        
        x = torch.tensor([x], dtype=torch.long).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=-1)
            if strategy == "Greedy":
                next_id = torch.argmax(probs[0]).item()
            else:
                next_id = torch.multinomial(probs[0], num_samples=1).item()
        
        next_token = id_to_token[next_id]
        generated.append(next_token)
        context.append(next_id)
    
    return " ".join(generated)


# ---------- Streamlit UI ----------
st.title("Linux Kernel Source Code Generator (MLP-based)")
st.markdown("Generate kernel-like C code snippets using a trained MLP language model.")

# Sidebar parameters
st.sidebar.header("Model Parameters")
context_window = st.sidebar.selectbox("Context Size", [5, 10])
embedding_dim = st.sidebar.selectbox("Embedding Dimension", [64])
activation = st.sidebar.selectbox("Activation Function", ["ReLU", "Tanh"])

# NEW: Add selection for generation strategy
strategy = st.sidebar.radio("Token Selection Strategy", ["Sampling", "Greedy"],
                            help="Sampling: more randomness, creative outputs.\nGreedy: deterministic, focused outputs.")

# Input text
prompt = st.text_area("Enter your starting code prompt:", "int main (")

# Custom CSS
st.markdown("""
    <style>
    .generated-box {
        width: 100%;
        margin: auto;
        background-color: #1e1e1e;
        border: 1px solid #3c3c3c;
        border-radius: 10px;
        padding: 1rem;
        font-family: 'Fira Code', monospace;
        font-size: 15px;
        line-height: 1.5;
        white-space: pre-wrap;
        word-wrap: break-word;
        overflow-wrap: break-word;
    }
    </style>
""", unsafe_allow_html=True)

# Generate button
if st.button("Generate Code"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLPcodegen(vocab_size, context_window, embedding_dim,512, activation).to(device)
    try:
        model.load_state_dict(torch.load(
            f"models/model_category_2{context_window}{embedding_dim}_{activation}_state.pt",
            map_location=device
        ), strict=False)
        st.success("Model loaded successfully!")
    except Exception as e:
        st.warning(f"Model not loaded from file: {e}")

    generated = generate_code(model, prompt, token_to_id, id_to_token,
                              max_new_tokens=40, context_window=context_window,
                              device=device, strategy=strategy)

    # Syntax highlighting
    formatter = formatters.HtmlFormatter(style="monokai", noclasses=True)
    highlighted_code = highlight(generated, lexers.CLexer(), formatter)

    st.subheader("Generated Code:")
    st.markdown(f"<div class='generated-box'>{highlighted_code}</div>", unsafe_allow_html=True)
