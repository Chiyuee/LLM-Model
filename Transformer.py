import torch
from torch import nn
from torch.utils.data import DataLoader
from torchtext.data import get_tokenizer


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        
        # Embed_size is the amount of dimensions for the word mapping
        # We can split these embedding sizes by the amount of heads
        # For this example, we are doing 256 embeddings / 8 heads
        # Each N instance will have 32 embeddings
        self.head_dim = embed_size // heads
        
        # This is to make sure embedding size is divisible by heads
        assert(self.head_dim * heads == embed_size), 'Embedding size is not divisble by heads'
        
        # print(f'Embed_size: {self.embed_size} Heads: {self.heads} Head_dim: {self.head_dim}')
        
        # V, K, Q will have input 32 head_dims, 32 output head_dims and no bias 
        self.values = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias = False)
        
        # 256 input, 256 output
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
        
    def forward(self, values, keys, query, mask):
        
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        # print(f'Value length: {value_len} Key length: {key_len} Query length: {query_len}')

        # Splitting embeddings to each head
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)
        
        # Torch.einsum does multiple matrix multplications in a small format and returns everything into one variable
        energy = torch.einsum('nqhd,nkhd->nhqk', [queries, keys])
    
        # This will check if the index is == to 0, it will switch it off and will not be calculated
        # This makes the matrix into a triangular matrix for our self attention
        if mask is not None: energy = energy.masked_fill(mask == 0, float('-1e20'))
        
        # ATTENTION MECHANISM
        # Einsum shortens the code to calculate multiple matrix multiplications
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim = 3)
        out = torch.einsum('nhql,nlhd->nqhd', [attention, values]).reshape(N, query_len, self.heads * self.head_dim)
        out = self.fc_out(out)
        
        return out




# This class contains the multi-head attention and feedforward network 
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        
        self.feedforward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        
        x = self.dropout(self.norm1(attention + query))
        forward = self.feedforward(x)
        out = self.dropout(self.norm2(forward + x))
        return out





class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        
        # nn.Embedding is the word embedding of each word's relationship with other
        # words within the vocab. src_vocab_size is the amount of words you have
        # Embedding size is how many dimensions you have for your vocab space.
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        
        # This is an alternative to implement positions for each embedding
        self.position_embedding = nn.Embedding(max_length, embed_size)
        
        self.layers = nn.ModuleList([TransformerBlock(embed_size, heads, dropout, forward_expansion) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        N, seq_length = x.shape
        # print(f'X Shape: {x.shape}')
        
        # This is to mark the word's position within the sentence
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        # print(positions)
        
        # Injecting positions to each word
        out = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))
        # print(out)
        for layer in self.layers:
            out = layer(out, out, out, mask)
        return out
    
    
    
    
    
class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.norm = nn.LayerNorm(embed_size)
        self.attention = SelfAttention(embed_size, heads = heads)
        self.transformer_block = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        
        return out





class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length):
        super(Decoder, self).__init__()
        self.device = device
        
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        
        self.layers = nn.ModuleList([DecoderBlock(embed_size, 
                                                  heads, 
                                                  forward_expansion, 
                                                  dropout, 
                                                  device) for _ in range(num_layers)])
        
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))
        
        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)
            
        out = self.fc_out(x)
        
        return out





class Transformer(nn.Module):
    def __init__(self, 
                 src_vocab_size, 
                 trg_vocab_size, 
                 src_pad_idx, 
                 trg_pad_idx, 
                 embed_size = 512, 
                 num_layers = 6, 
                 forward_expansion = 4, 
                 heads = 8, 
                 dropout = 0, 
                 device = "cuda" if torch.cuda.is_available() else 'cpu', 
                 max_length = 100):
        
        super(Transformer, self).__init__()
        
        self.encoder = Encoder(src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length)
        
        self.decoder = Decoder(trg_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length)
        
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
    
    def make_src_mask(self, src):
        
        # Unsqueeze adds an extra dimension
        # What this is doing is making each index a 3 dim tensor
        # unsqueeze(1) makes it 2 dim then unsqueeze(1).unsqueeze(2) makes it 3 dim
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        
        return src_mask.to(self.device)
    
    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        
        # torch.tril returns the lower triangular part of a matrix
        # [1 0 0 0 0]
        # [1 1 0 0 0]
        # [1 1 1 0 0]
        # [1 1 1 1 0]
        # [1 1 1 1 1]
        # This is to prevent words from calculating words after its own
        trg_mask = torch.tril(torch.ones(trg_len, trg_len)).expand(N, 1, trg_len, trg_len)
        
        return trg_mask.to(self.device)
    
    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        
        return out