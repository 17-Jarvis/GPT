import tiktoken
import torch
import math
print(torch.cuda.is_available())
import torch.nn as nn
from torch.nn import functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 64 
block_size = 256 
num_iter = 10000
eval_interval = 500
eval_iters = 200 
d_model = 512
d_k = 16 
Nx = 6 
dropout_rate = 0.2
lr_rate = 1e-3 
h = 6 

torch.manual_seed(47)
with open('./input.txt', 'r', encoding = 'utf-8') as f:
    text = f.read()

unique_chars = set(text)
list_unique_chars = list(unique_chars)
chars = sorted(list_unique_chars)
vocab_size = len(chars)

chars_to_int = {c:i for i, c in enumerate(chars)}
int_to_chars = {i:c for i, c in enumerate(chars)} 

def encode(s):
    encoding = [chars_to_int[c] for c in s]
    return encoding

def decode(l):
    decoding = ''.join([int_to_chars[i] for i in l])
    return decoding
data = torch.tensor(encode(text), dtype=torch.long)


split_90perc = int(0.9*len(data))
train_data = data[:split_90perc]
valid_data = data[split_90perc:]

torch.manual_seed(47)
def get_batch(split):
    if split == "train":
        data = train_data
    else:
        data = valid_data
    ix = torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix]) 
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) 
    x, y = x.to(device), y.to(device)
    return x,y

class SelfAttention(nn.Module):
    """Self Attention (One Head)"""
    """ d_k = C """
    def __init__(self, d_k):
        super().__init__() 
        d_k = d_model // h
        self.keys = nn.Linear(d_model, d_k, bias = False)
        self.queries = nn.Linear(d_model, d_k, bias = False)
        self.values = nn.Linear(d_model, d_k, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, X):
        """Computing Attention Matrix"""
        B, T, C = X.shape
        K = self.keys(X) 
        Q = self.queries(X) 
        scaled_dot_product = Q @ K.transpose(-2,-1) * 1/math.sqrt(C)
        scaled_dot_product_masked = scaled_dot_product.masked_fill(self.tril[:T, :T]==0, float('-inf'))
        attention_matrix = F.softmax(scaled_dot_product_masked, dim=-1) 
        attention_matrix = self.dropout(attention_matrix)
        V = self.values(X) 
        output =  attention_matrix @ V 
        return output
    

class MultiHeadAttention(nn.Module):
    """Multi Head Self Attention"""
    """h: #heads"""
    def __init__(self, h, d_k):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttention(d_k) for _ in range(h)])
        self.projections = nn.Linear(h*d_k, d_model)
        self.droupout = nn.Dropout(dropout_rate)
    
    def forward(self, X):
        combined_attentions = torch.cat([h(X) for h in self.heads], dim = -1)
        combined_attentions = self.projections(combined_attentions)
        combined_attentions = self.droupout(combined_attentions)
        return combined_attentions
    

class FeedForward(nn.Module):
    """FeedForward Layer with ReLU activation function"""

    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.ReLU(),
            nn.Linear(4*d_model, d_model),
            nn.Dropout(dropout_rate)
        )
    def forward(self, X):
        
        return self.net(X)
class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.eps = eps
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

    def __call__(self, x):
        xmean = x.mean(1, keepdim=True)  
        xvar = x.var(1, keepdim=True)  
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)  
        self.out = self.gamma * xhat + self.beta      
        return self.out
    
    def parameters(self):
        return [self.gamma, self.beta]


class Block(nn.Module):
    """Multiple Blocks of Transformer"""
    def __init__(self, d_model, h):
        super().__init__()
        d_k = d_model // h
        self.attention_head = MultiHeadAttention(h, d_k) 
       
        self.feedforward = FeedForward(d_model)
        
        self.ln1 = nn.LayerNorm(d_model)
        
        self.ln2 = nn.LayerNorm(d_model)
    
    def forward(self,X):
        X = X + self.attention_head(self.ln1(X))
        X = X + self.feedforward(self.ln2(X))
        return X
class BigramLM(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.token_embedding_table = nn.Embedding(vocab_size, d_model)
        self.positional_encodings_table = nn.Embedding(block_size, d_model)
        self.blocks = nn.Sequential(
            Block(d_model, h = 4),
            Block(d_model, h = 4),
            Block(d_model, h = 4),
            #adding one to the blocks
            nn.LayerNorm(d_model),
        ) 
        
        self.lin_layer = nn.Linear(d_model, vocab_size) 


    def forward(self, idx, targets=None):    
        B, T = idx.shape
        tok_embeddings = self.token_embedding_table(idx) 
        pos_encodings = self.positional_encodings_table(torch.arange(T, device=device)) 
        X = tok_embeddings + pos_encodings
        logits = self.lin_layer(X) # (B, T, vocab_size) 

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C )
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    # function that generates updated tokens as the new tokens are added per time step, per batch
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1) 
            idx_next = torch.multinomial(probs, num_samples=1) 
            idx = torch.cat((idx, idx_next), dim=1) 
        return idx


@torch.no_grad()
def estimate_loss():
    result = {}
    model.eval()
    for split in ['train', 'valid_date']:
        losses = torch.zeros(eval_iters)
        for e in range(eval_iters):
            X,Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[e] = loss.item()
        result[split] = losses.mean()
    model.train()
    return result

model = BigramLM()
# moving model paramters to device
model_device = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr = lr_rate)
for iter in range(num_iter):
    if iter % eval_interval == 0:
       losses = estimate_loss()
       print(f"step {iter}: train loss is {losses['train']:.5f} and validation loss is {losses['valid_date']:.5f}")
    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


custom_context = "I am ARQ friend of ABISH"
encoded_context = encode(custom_context)
context_tensor = torch.tensor(encoded_context,dtype=torch.long,device=device).unsqueeze(0)

#context = torch.zeros((1, 1), dtype=torch.long, device = device)
print(decode(model.generate(context_tensor, max_new_tokens=500)[0].tolist()))

    