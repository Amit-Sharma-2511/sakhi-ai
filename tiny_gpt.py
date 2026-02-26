import torch
import torch.nn as nn
from torch.nn import functional as F
import requests
import os
from transformers import AutoTokenizer
import pyttsx3
import speech_recognition as sr
import threading
import queue

# Hyperparameters
batch_size = 32 
block_size = 128 
max_iters = 3000 
eval_interval = 250 
learning_rate = 1e-3 
device = 'cpu' 
eval_iters = 50 
n_embd = 128 
n_head = 4 
n_layer = 2 
dropout = 0.3
# -----------------------------------------------------------------------------

torch.manual_seed(1337)

# 1. Dataset Preparation - Indian Assistant persona
data_path = 'indian_chat.txt'
if not os.path.exists(data_path):
    print("Dataset not found. Please ensure 'indian_chat.txt' exists.")
    text = "User: Hello Sakhi!\nSakhi: Namaste! I am Sakhi, your Indian voice assistant.\n" * 100
else:
    with open(data_path, 'r', encoding='utf-8') as f:
        base_text = f.read()
    # Augment the small dataset by repeating it to help the model learn fast
    text = (base_text + "\n") * 50 

# 2. Tokenizer (BPE - GPT-2)
print("Loading GPT-2 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
special_tokens = {"additional_special_tokens": ["<|user|>", "<|assistant|>"]}
tokenizer.add_special_tokens(special_tokens)
vocab_size = len(tokenizer)

# Encoder/Decoder functions
def encode(s):
    return tokenizer.encode(s, add_special_tokens=False)

def decode(l):
    return tokenizer.decode(l, skip_special_tokens=True)

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# Data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# 3. Transformer Components

class Head(nn.Module):
    """ One head of self-attention """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        # Dot product of queries and keys, then scale by sqrt(head_size)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1]**-0.5) 
        # Causal mask: don't look into the future
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B, T, head_size)
        out = wei @ v # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ A simple linear layer followed by a non-linearity """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # x + ... implements residual connections
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# 4. GPT Model Implementation

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

# 5. Initialization and Parameter Counting
model = GPTLanguageModel()
m = model.to(device)

# Total number of parameters calculation
total_params = sum(p.numel() for p in m.parameters())
print(f"Total parameters: {total_params / 1e6:.2f}M")

# 5.5 Voice Assistant Logic - Defensive Implementation
def speak(text_to_speak):
    if not text_to_speak.strip(): return
    print(f"Sakhi is speaking...", end="\r")
    try:
        # Move import and init inside to isolate driver issues
        import pyttsx3
        temp_engine = pyttsx3.init()
        voices = temp_engine.getProperty('voices')
        if len(voices) > 1:
            temp_engine.setProperty('voice', voices[1].id)
        temp_engine.setProperty('rate', 160)
        temp_engine.say(text_to_speak)
        temp_engine.runAndWait()
        # Do not call stop() or __del__ explicitly, let GC handle it
    except Exception as e:
        print(f"\nVoice Output Notice: {e}")

def listen():
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            print("\nListening...", end="\r")
            r.adjust_for_ambient_noise(source, duration=0.5)
            audio = r.listen(source, timeout=8, phrase_time_limit=10)
        print("Recognizing...", end="\r")
        query = r.recognize_google(audio, language='en-in')
        print(f"You said: {query}")
        return query
    except Exception:
        print("I didn't catch that. Please speak again.")
        return "None"

# 6. Training and Response Logic - Setup
model_path = 'tiny_gpt.model'

def get_sakhi_response(user_input, chat_history_tokens, model_instance):
    """Generates a response from Sakhi given user input and rolling token history."""
    # Add user input to history and encode
    chat_history_tokens.extend(encode(user_input + "\n"))
    
    # Trim context to fit block_size
    if len(chat_history_tokens) > block_size:
        chat_history_tokens = chat_history_tokens[-block_size:]
        
    input_tensor = torch.tensor([chat_history_tokens], dtype=torch.long, device=device)
    
    # Generate response (reduced tokens for speed)
    generated_idx = model_instance.generate(input_tensor, max_new_tokens=30)[0].tolist()
    
    new_tokens = generated_idx[len(chat_history_tokens):]
    full_response = decode(new_tokens).strip()
    
    # Clean up response: take first sentence/newline
    response = full_response.split('\n')[0].split('.')[0] + "."
    if len(response) < 5: response = full_response.split('\n')[0][:50]
    
    # Update history for next turn
    chat_history_tokens.extend(new_tokens)
    if len(chat_history_tokens) > block_size:
        chat_history_tokens = chat_history_tokens[-block_size:]
        
    return response, chat_history_tokens

if __name__ == "__main__":
    # Training or Loading Model
    train_choice = 'n'

    if os.path.exists(model_path):
        print(f"\nFound existing model at '{model_path}'.")
        train_choice = input("Do you want to retrain from scratch? (y/n): ").lower()

    if train_choice == 'y':
        print("Resetting model to Indian Persona 'Sakhi'...")
        model = GPTLanguageModel().to(device)
        m = model 
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        print(f"Training for {max_iters} steps. This will make Sakhi much smarter!")
        for iter in range(max_iters):
            if iter % eval_interval == 0:
                losses = estimate_loss(model)
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            xb, yb = get_batch('train')
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        
        torch.save(model.state_dict(), model_path)
        print(f"Training complete. Model saved to {model_path}")
    else:
        print(f"Loading model from '{model_path}'...")
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}. Please consider retraining.")

    # 7. Initial Text Generation
    print("\nGenerating a quick sample to check performance...")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print("-" * 30)
    print(decode(m.generate(context, max_new_tokens=200)[0].tolist()))
    print("-" * 30)

    # 8. Interactive Voice Chat Mode - Sakhi
    print("\n" + "="*50)
    print(" SAKHI INDIAN VOICE ASSISTANT ".center(50, "="))
    print("="*50)
    print("Namaste! I am Sakhi, your voice assistant.")
    speak("Namaste! I am Sakhi, your voice assistant. How can I help you today?")

    chat_context = []

    while True:
        print("\n1. Type your message")
        print("2. Speak your message")
        print("Type 'exit' to quit")
        choice = input("Choice (1/2): ")

        if choice == '1':
            user_input = input("You: ")
        elif choice == '2':
            user_input = listen()
            if user_input == "None": continue
        elif choice.lower() == 'exit':
            speak("Alvida! See you soon.")
            break
        else:
            continue

        if user_input.lower() in ['exit', 'quit']:
            speak("Alvida! See you soon.")
            break
        
        response, chat_context = get_sakhi_response(user_input, chat_context, m)
        print(f"Sakhi: {response}")
        speak(response)

# 9. Educational Commentary: How to Scale
"""
HOW TO SCALE THE MODEL:
1. Increase n_layer (Depth): Adding more Transformer blocks allows the model to learn more abstract concepts.
2. Increase n_embd (Width): Increasing the vector dimension allows the model to capture more features per token.
3. Increase n_head: More attention heads allow the model to focus on different parts of the context simultaneously.
4. Increase block_size: Longer sequence lengths allow the model to remember farther back in time.
5. Larger Dataset: Instead of characters, use Byte Pair Encoding (BPE) (like the 'tiktoken' library) and train on massive datasets like OpenWebText.
6. GPU Training: Switch 'device' to 'cuda' to train significantly faster.
"""
