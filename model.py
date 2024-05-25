import torch
import torch.nn as nn
import math

# Input embedding does what? Converts words (ID tokens) into 512 bit vectors
# These vectors contain parameters which later help determine the context of each word.


class InputEmbeddings(nn.Module):
# Constructor for class --> initializes values when instantiating the class!
# Takes in size of vector we want (d_model) and sequence length (vocab_size)
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

#This Vector is then added with the vector produced from the positional embedding process,
# which will then fully represent each word before being encoded, with both the context and position of the word.

# nn.Module is a class from Pytorch which is used as a subclass to build the layers of NN
# Including the input, output, and hidden --> Explains why in the code you can see how certain output is hidden/inputted
# Without classifying it as hidden, input, etc. 
class PositionalEcoding(nn.Module):
    # Returns the input embedded matrix with the positional embedding added on to it.

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # Dropouts

        #Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)

        #Create a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sin for the even positions
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply sin for the even positions
        pe[:,1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # Tensor becomes of dimension (1, seq_len, d_model)

        # Here we save the buffer; You keep the tensor not as a learned parameter but just save it, you register it as a buffer.
        # So you can compare the state of the original tensor as well as the current state of the model.
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        # Requires grad -> False tells model not to learn from this; Positional embedding is fixed so model should not pay attention to this.
        return self.dropout(x)

# Next is the encoding process --> Multi head attention, forward feeding, add and normalization
# We will start with adding and normalization first because it is easiest.

class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # This is multiplied
        self.bias = nn.Parameter(torch.zeros(1)) # This is added

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1, keepdim = True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias









