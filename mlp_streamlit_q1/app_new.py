import torch.nn as nn
import pickle
import torch
import streamlit as st
from model_define_ import MLPTextGenerator_Natural, MLPTextGenerator_Code
import torch.nn.functional as F

# ---------------------- MODEL LOADING ----------------------
@st.cache_resource
def load_model(dataset, block_size, emb_dim, activation):
    with open(f"vocab/category_{dataset}/itos.pkl", "rb") as f:
        itos = pickle.load(f)
    with open(f"vocab/category_{dataset}/stoi.pkl", "rb") as f:
        stoi = pickle.load(f)

    if dataset == 1:
        vocab_size = len(stoi)
        model = MLPTextGenerator_Natural(
            vocab_size=vocab_size,
            activation_fn=activation,
            embed_dim=emb_dim,
            block_size=block_size
        )
        model_path = f"models/category_{dataset}/model_category_{dataset}_{block_size}_{emb_dim}_{activation}_state.pt"
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        return stoi, itos, model

    elif dataset == 2:
        vocab_size = len(stoi)
        model = MLPTextGenerator_Code(
            vocab_size=vocab_size,
            activation=activation,
            embedding_dim=emb_dim,
            context_window=block_size
        )
        model_path = f"models/category_{dataset}/model_category_{dataset}_{block_size}_{emb_dim}_{activation}_state.pt"
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        if "<unk>" not in stoi:
            stoi["<unk>"] = 0
        if 0 not in itos:
            itos[0] = "<unk>"
        return stoi, itos, model


# ---------------------- TEXT GENERATION FUNCTIONS ----------------------
def generate_text_sampling(model, start_context, length, block_size, temperature, stoi, itos):
    model.eval()
    generated = start_context.lower().split()
    for _ in range(length):
        context_words = generated[-block_size:]
        if len(context_words) < block_size:
            context_words = ["<unk>"] * (block_size - len(context_words)) + context_words
        input_idx = [stoi.get(w, 0) for w in context_words]
        input_tensor = torch.tensor(input_idx, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            logits = model(input_tensor)
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1).item()
        generated.append(itos.get(next_idx, "<unk>"))
    return " ".join(generated)


def generate_text_greedy(model, start_context, length, block_size, temperature, stoi, itos):
    model.eval()
    generated = start_context.lower().split()
    for _ in range(length):
        context_words = generated[-block_size:]
        if len(context_words) < block_size:
            context_words = ["<unk>"] * (block_size - len(context_words)) + context_words
        input_idx = [stoi.get(w, 0) for w in context_words]
        input_tensor = torch.tensor(input_idx, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            logits = model(input_tensor)
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.argmax(probs, dim=-1).item()
        generated.append(itos.get(next_idx, "<unk>"))
    return " ".join(generated)


# ✅ YOUR UPDATED FUNCTION (NO DIVISION, DEVICE = "cuda" DEFAULT)
def generate_code(model, start_text, token_to_id, id_to_token, max_new_tokens=30, context_window=5, device="cpu", strategy="Greedy"):
    model.eval()
    tokens = start_text.strip().split()
    context = [token_to_id.get(t, 0) for t in tokens]
    generated = tokens.copy()

    for _ in range(max_new_tokens):
        x = context[-context_window:]
        if len(x) < context_window:
            x = [0] * (context_window - len(x)) + x
        x = torch.tensor([x], dtype=torch.long).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=-1)
            if strategy == "Greedy":
                next_id = torch.argmax(probs[0]).item()
            else:
                next_id = torch.multinomial(probs[0], num_samples=1).item()

        # Safe lookup to prevent KeyError
        next_token = id_to_token.get(next_id, "<unk>")
        generated.append(next_token)
        context.append(next_id)

    return " ".join(generated)


# ---------------------- LOSS PLOT ----------------------
def plot_loss(dataset, block_size, embed_dim, activation):
    image_path = f"loss_images/model_category_{dataset}_{block_size}_{embed_dim}_{activation}.png"
    st.image(
        image_path,
        caption=f"Graph for Dataset {dataset}, Context Size {block_size}, Embedding Size {embed_dim}, Activation Function {activation}",
        use_container_width=True
    )


# ---------------------- STREAMLIT UI ----------------------
st.title("MLP Next Token Generator Playground")

st.markdown("----")
st.markdown("""
### About this Project
This app explores next-word generation using MLPs trained on **Paul Graham Essays** and **Linux Source Code**.
Adjust the controls in each tab to load different trained models.
The temperature parameter affects diversity of generated tokens.
Each model was trained with varying hyperparameters to analyze
generalization and convergence behavior.

There were **2 Fully-Connected Hidden Layers** used in the MLP Architecture of **size 1024 neurons** each.
""")
st.markdown("----")

# ---------------------- TAB LAYOUT ----------------------
tab1, tab2 = st.tabs(["📝 Natural Language", "💻 Code"])

# ---------------------- NATURAL LANGUAGE TAB ----------------------
with tab1:
    st.header("Natural Language (Paul Graham Essays)")

    output_approach = st.selectbox("Generation Approach", ["Greedy", "Sampling"], key="nat_approach")
    block_size = st.selectbox("Block Size", [10, 20], key="nat_block")
    embed_dim = 64
    activation = st.selectbox("Activation", ["ReLU", "Tanh"], key="nat_act")
    temperature = st.slider("Temperature", 0.5, 2.0, 1.0, key="nat_temp")

    with st.spinner("Loading Selected Model..."):
        stoi, itos, model = load_model(1, block_size, embed_dim, activation)
    st.write(f"Vocabulary Size used in **Natural Language Dataset**: {len(stoi)} tokens")

    output_size = st.slider("Output Size", 10, 60, 30, key="nat_out")
    prompt = st.text_input("**Enter Prompt**:", key="nat_prompt")

    if st.button("**Generate**", key="nat_gen"):
        if output_approach == "Greedy":
            output = generate_text_greedy(model, prompt, output_size, block_size, temperature, stoi, itos)
        else:
            output = generate_text_sampling(model, prompt, output_size, block_size, temperature, stoi, itos)
        st.write('**Output:**')
        st.write(output)

    plot_loss(1, block_size, embed_dim, activation)

    st.markdown("""
    ### Training Summary
    The loss curve below shows the **training and validation losses** over many epochs 
    for visualization purposes.  
    However, the **final model deployed here** was trained for a limited number of epochs 
    to avoid overfitting and ensure better generalization.
    """)

# ---------------------- CODE TAB ----------------------
with tab2:
    st.header("Code (Linux Source)")

    output_approach = st.selectbox("Generation Approach", ["Greedy", "Sampling"], key="code_approach")
    block_size = st.selectbox("Block Size", [5, 10], key="code_block")
    embed_dim = 64
    activation = st.selectbox("Activation", ["ReLU", "Tanh"], key="code_act")
    temperature = st.slider("Temperature", 0.5, 2.0, 1.0, key="code_temp")

    with st.spinner("Loading Selected Model..."):
        stoi, itos, model = load_model(2, block_size, embed_dim, activation)
    st.write(f"Vocabulary Size used in **Code Dataset**: {len(stoi)} tokens")

    output_size = st.slider("Output Size", 10, 60, 30, key="code_out")
    prompt = st.text_input("**Enter Prompt**:", key="code_prompt")

    if st.button("**Generate**", key="code_gen"):
        output = generate_code(
            model,
            start_text=prompt,
            token_to_id=stoi,
            id_to_token=itos,
            max_new_tokens=output_size,
            context_window=block_size,
            device="cpu",
            strategy=output_approach
        )
        st.write('**Output:**')
        st.write(output)

st.markdown("----")
with st.expander("**Credits**"):
    st.markdown("""
    The natural language dataset for this project includes text from essays by **[Paul Graham](http://paulgraham.com/)**, 
    used solely for educational and research purposes.
    """)
