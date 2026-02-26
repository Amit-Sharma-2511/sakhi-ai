from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import uuid
import time
from transformers import AutoTokenizer

app = Flask(__name__)
CORS(app)

# Hyperparameters (MUST MATCH retrain_sakhi.py)
block_size = 128
n_embd = 128
n_head = 4
n_layer = 2
dropout = 0.3
device = 'cpu'
model_path = 'tiny_gpt.model'

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
special_tokens = {"additional_special_tokens": ["<|user|>", "<|assistant|>"]}
tokenizer.add_special_tokens(special_tokens)
vocab_size = len(tokenizer)

def encode(s): return tokenizer.encode(s, add_special_tokens=False)
def decode(l): return tokenizer.decode(l, skip_special_tokens=True)

# Model Components
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key, self.query, self.value = nn.Linear(n_embd, head_size, bias=False), nn.Linear(n_embd, head_size, bias=False), nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1]**-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        return F.softmax(wei, dim=-1) @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa, self.ffwd = MultiHeadAttention(n_head, n_embd // n_head), FeedForward(n_embd)
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table, self.position_embedding_table = nn.Embedding(vocab_size, n_embd), nn.Embedding(block_size, n_embd)
        self.blocks, self.ln_f, self.lm_head = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)]), nn.LayerNorm(n_embd), nn.Linear(n_embd, vocab_size)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb, pos_emb = self.token_embedding_table(idx), self.position_embedding_table(torch.arange(T, device=device))
        logits = self.lm_head(self.ln_f(self.blocks(tok_emb + pos_emb)))
        return logits, None
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -block_size:])
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, num_samples=1)), dim=1)
        return idx

# Initialize model
print("Loading Sakhi's brain for API...")
model = GPTLanguageModel().to(device)

def load_brain():
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            print("Brain loaded successfully.")
        except Exception as e:
            print(f"Warning: Failed to load brain. Error: {e}")

# Removed server-side TTS for deployment compatibility

@app.route('/chat', methods=['POST'])
def chat():
    load_brain() # Reload to get latest weights from training
    data = request.json
    user_msg = data.get('message', '')
    prompt = f"<|user|> {user_msg}\n<|assistant|>"
    tokens = encode(prompt)
    
    input_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
    # Generate response
    generated = model.generate(input_tensor, max_new_tokens=50)[0].tolist()
    
    response_full = decode(generated[len(tokens):]).strip()
    # Cleaning: take only the first sentence or first line, and stop at next participant
    response = response_full.split('\n')[0].split('<|user|>')[0].split('<|assistant|>')[0].split('User:')[0].split('Sakhi:')[0].strip()
    
    if not response: response = "I'm listening. Tell me more!"
    
    return jsonify({
        'response': response,
        'tokens': [], 
        'audio_url': None
    })

# /audio route removed

if __name__ == '__main__':
    load_brain()
    app.run(port=5000)
