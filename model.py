import torch
import torch.nn as nn
import math


class InputEmbeddings(nn.Module):
    
    def __init__(self, d_model:int, vocab_size:int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, seq_len: int, dropout:float) ->None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        
        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) #(seq_len,1)
        
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000.0)/d_model)) # (d_model/2)
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position* div_term) # sin(position* (10000 **(2i/d_model)))
        
        # Apply cosine to odd indices
        pe[:,1::2] = torch.cos(position*div_term)
        
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1,seq_len , d_model)
        
        # Register the positional encoding as a buffer
        self.register_buffer('pe',pe)
        
    def forward(self,x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad(False) # (batch , seq_len, d_model)
        return self.dropout(x)
    
class LayerNormalization(nn.Module):
    def __init__(self, features:int , eps: float= 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) # Alpha is learnable parameter
        self.bias = nn.Parameter(torch.zeros(features)) # bias is also learnable parameter
        
    def forward(self, x):
        # x: ( bathc, seq_len,hidden_size)
        
        # Keep the dimensions for broadcasting
        mean =  x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1)
        
        std = x.std(dim= -1 , keepdim = True)
        
        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x-mean)/( std+ self.eps) + self.bias
    
    
class FeedForwardBlock(nn.Module):
    
    def __init__(self, d_model:int, d_ff:int, dropout:float) ->None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2
        
    def forward(self,x):
        # (batch , seq_len, d_model) --> (batch, seq_len, d_ff) --> ( batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout:float) -> None:
        super().__init__()
        self.d_model = d_model # Embedding vector size
        self.h = h # number of  heads
        
        # We have to insure here d_model is divisibleby h
        assert d_model % h == 0 , "d_model is not divisible by h"
        
        self.d_k = d_model // h #dimension of vector seen by each head
        self.w_q = nn.Linear(d_model , d_model , bias = False) # Wq
        self.w_k = nn.Linear(d_model, d_model, bias = False) # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv
        self.w_o = nn.Linear(d_model, d_model, bias = False) # Wo
        self.dropout = nn.Dropout(dropout)
        
        