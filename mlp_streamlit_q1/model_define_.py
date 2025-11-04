import torch.nn as nn
import torch
class MLPTextGenerator_Natural(nn.Module):
    def __init__(self, vocab_size,activation_fn, embed_dim=64, hidden_size=1024,block_size=10,dropout_value = 0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim * block_size, hidden_size)
        self.act = nn.ReLU() if activation_fn == "ReLU" else nn.Tanh()
        self.dropout = nn.Dropout(dropout_value)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
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

class MLPTextGenerator_Code(nn.Module):
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
