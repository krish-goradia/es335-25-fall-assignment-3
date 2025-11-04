import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

# ---------- Config ----------
st.set_page_config(page_title="Text Generator", page_icon="✍️", layout="wide")

# ---------- Load Vocabulary ----------
with open("data/stoi_text.pkl", "rb") as f:
    stoi = pickle.load(f)
with open("data/itos_text.pkl", "rb") as f:
    itos = pickle.load(f)

vocab_size = len(stoi)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- Model ----------
class MLPTextGenerator(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_size=1024, block_size=10, dropout_value=0.2, activation="ReLU"):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim * block_size, hidden_size)
        self.act = nn.ReLU() if activation.lower() == "relu" else nn.Tanh()
        self.dropout = nn.Dropout(dropout_value)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        emb = self.dropout(self.embedding(x))
        emb = emb.view(emb.size(0), -1)
        h = self.act(self.fc1(emb))
        h = self.dropout(h)
        h = self.act(self.fc2(h))
        h = self.dropout(h)
        logits = self.out(h)
        return logits

# ---------- Text Generation ----------
def generate_text(model, start_context, length=20, block_size=10, temperature=1.0, device="cpu"):
    model.eval()
    generated = start_context.lower().split()

    for _ in range(length):
        context_words = generated[-block_size:]
        if len(context_words) < block_size:
            context_words = ["<unk>"] * (block_size - len(context_words)) + context_words

        input_idx = [stoi.get(w, 0) for w in context_words]
        input_tensor = torch.tensor(input_idx, dtype=torch.long).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(input_tensor)
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1).item()

        generated.append(itos.get(next_idx, "<unk>"))

    return " ".join(generated)

# ---------- Streamlit UI ----------
st.title("Natural Language Text Generator (MLP-based)")
st.markdown("Generate natural language text using trained MLP models.")

st.sidebar.header("Model Parameters")
block_size = st.sidebar.selectbox("Context Size (Block Size)", [10, 20])
embed_dim = st.sidebar.selectbox("Embedding Dimension", [64])
activation = st.sidebar.selectbox("Activation Function", ["ReLU", "Tanh"])
temperature = st.sidebar.slider("Temperature", 0.5, 2.0, 1.0)

prompt = st.text_area("Enter your starting text:", "Once upon a time")

if st.button("Generate Text"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLPTextGenerator(vocab_size, embed_dim, 1024, block_size, 0.2, activation).to(device)

    try:
        model.load_state_dict(torch.load(
            f"models/model_category_1_{block_size}_{embed_dim}_{activation}_state.pt",
            map_location=device
        ), strict=False)
        st.success("Model loaded successfully!")
    except Exception as e:
        st.warning(f"Could not load model file: {e}")

    generated_text = generate_text(model, prompt, length=40, block_size=block_size, temperature=temperature, device=device)
    st.subheader("Generated Text:")
    st.write(generated_text)
