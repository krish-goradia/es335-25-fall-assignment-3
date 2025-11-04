import torch.nn as nn
import pickle
import torch
import streamlit as st
from model_define_ import MLPTextGenerator_Natural
from model_define_ import MLPTextGenerator_Code

import torch.nn.functional as F
@st.cache_resource
def load_model(dataset,block_size,emb_dim,activation):

    with open(f"vocab/category_{dataset}/itos.pkl","rb") as f:
        itos = pickle.load(f)

    with open(f"vocab/category_{dataset}/stoi.pkl","rb") as f:
        stoi = pickle.load(f)

   
    if dataset == 1:
        
        vocab_size = len(stoi)
        model = MLPTextGenerator_Natural(vocab_size=vocab_size,activation_fn=activation,embed_dim=emb_dim,block_size=block_size)
        model_path = f"models/category_{dataset}/model_category_{dataset}_{block_size}_{emb_dim}_{activation}_state.pt"
        model.load_state_dict(torch.load(model_path,map_location="cpu"))
        model.eval()
        return stoi,itos,model
    if dataset == 2:
        
        block_size = block_size//2
        vocab_size = len(stoi)
        model_path = f"models/category_{dataset}/model_category_{dataset}_{block_size}_{emb_dim}_{activation}_state.pt"
        model = MLPTextGenerator_Code(vocab_size=vocab_size,activation=activation,embedding_dim=emb_dim,context_window=block_size)
        model.load_state_dict(torch.load(model_path,map_location="cpu"))
        model.eval()
        if "<unk>" not in stoi:
            stoi["<unk>"] = 0  # reuse an existing embedding (like padding or first token)
        if 0 not in itos:
            itos[0] = "<unk>"
        return stoi,itos,model
        
    

    # elif dataset == 2:
    #     model = MLPTextGenerator_Code(vocab_size=vocab_size,embed_dim=emb_dim,block_size=block_size)
    #     model.load_state_dict(torch.load(model_path,map_location="cpu"))
    #     model.eval()
    #     return stoi,itos,model
    
def generate_text_sampling(model, start_context, length, block_size, temperature,stoi,itos):
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
def generate_text_greedy(model, start_context, length, block_size, temperature,stoi,itos):
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
            logits = logits
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.argmax(probs,dim=-1).item()

        generated.append(itos.get(next_idx, "<unk>"))

    return " ".join(generated)
def generate_code(model, start_text, stoi, itos, max_new_tokens=30, context_window=5, device="cpu", strategy="Greedy"):
    model.eval()
    # Convert start text to tokens
    tokens = start_text.strip().split()
    context = [stoi.get(t, 0) for t in tokens]  # unknown tokens → 0
    
    generated = tokens.copy()
    
    for _ in range(max_new_tokens):
        # Take the last context_window tokens
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
        
        next_token = itos[next_id]
        generated.append(next_token)
        context.append(next_id)
    
    return " ".join(generated)



def plot_loss(dataset,block_size,embed_dim,activation):
    image_path = f"loss_images/model_category_{dataset}_{block_size}_{embed_dim}_{activation}.png"
    st.image(image_path, caption=f"Graph for Dataset {dataset_category}, Context Size {block_size}, Embeddings Size {embed_dim}, Activation Function {activation}", use_container_width=True)

st.title("MLP Next Token Generator Playground")
#dataset_category = st.sidebar.radio("Select Category",["Natural Language","Code"])
dataset_category = "Natural Language"
output_approach= st.sidebar.selectbox("Generation Approach",["Greedy","Sampling"])
block_size = st.sidebar.selectbox("Block Size",[10,20])
embed_dim = st.sidebar.selectbox("Embedding Dimension",[32,64])
#embed_dim = 64
dataset = 1
activation = st.sidebar.selectbox("Activation",["ReLU","Tanh"])
temperature = st.sidebar.slider("Temperature",0.5,2.0,1.0)

st.markdown("----")


st.markdown(""" 
### About this Project
This app explores next-word generation using MLPs trained on **Paul Graham Essays**.
Adjust the controls in the sidebar to load different trained models.
The temperature parameter affects diversity of generated tokens.
Each model was trained with varying hyperparameters to analyze
generalization and convergence behavior.

There were **2 Fully-Connected Hidden Layers** used in the MLP Architecture of **size 1024 neurons** each.
""")
st.markdown("----")


if dataset_category == "Natural Language":
    dataset = 1
elif dataset_category == "Code":
    dataset = 2

with st.spinner("Loading Selected Model"):
    stoi,itos,model = load_model(dataset,block_size,embed_dim,activation)
st.write(f"Vocabulary Size used in **Dataset {dataset_category}**: {len(stoi)} tokens")

output_size = st.sidebar.slider("Output Size",10,60,30)
prompt = st.text_input("**Enter Prompt**: ")



if(st.button("**Generate**")):
    if dataset == 1:
        if(output_approach == "Greedy"):
            output = generate_text_greedy(model,prompt,output_size,block_size,temperature,stoi,itos)
            st.write('**Output:**')
            st.write(output)
        elif(output_approach == "Sampling"):
            output = generate_text_sampling(model,prompt,output_size,block_size,temperature,stoi,itos)
            st.write('**Output:**')
            st.write(output)
    elif dataset == 2:
        block_size = block_size//2
        output = generate_code(model, start_text = prompt, stoi = stoi, itos = itos, max_new_tokens=output_size, context_window=block_size, device="cpu", strategy=output_approach)
        st.write('**Output:**')
        st.write(output)
if(dataset==1):
    plot_loss(dataset,block_size,embed_dim,activation)

st.markdown("""
### Training Summary
The loss curve below shows the **training and validation losses** over a large number of epochs 
for visualization purposes.  
However, the **final model deployed here was trained for a certain number of epochs** 
to avoid overfitting and ensure better generalization and better validation loss.
""")


st.markdown("----")
with st.expander("**Credits**"):
    st.markdown("""
    The natural language dataset for this project includes text from essays by **[Paul Graham](http://paulgraham.com/)**, 
    used solely for educational and research purposes.
    """)
