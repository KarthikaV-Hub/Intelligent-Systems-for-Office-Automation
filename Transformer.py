# Work with transformer architecture using various attention mechanism
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttn(nn.Module):
    def __init__(self, embed, heads, attn_type="standard", window_size=4):
        super().__init__()
        self.embed = embed
        self.heads = heads
        self.head_dim = embed // heads
        self.attn_type = attn_type
        self.window = window_size
        self.lin_q = nn.Linear(embed, embed)
        self.lin_k = nn.Linear(embed, embed)
        self.lin_v = nn.Linear(embed, embed)
        self.out = nn.Linear(embed, embed)

    def forward(self, v, k, q, mask=None):
        N, seq = q.shape[0], q.shape[1]
        Q = self.lin_q(q)
        K = self.lin_k(k)
        V = self.lin_v(v)
        Q = Q.view(N, seq, self.heads, self.head_dim).transpose(1,2)
        K = K.view(N, seq, self.heads, self.head_dim).transpose(1,2)
        V = V.view(N, seq, self.heads, self.head_dim).transpose(1,2)
        if self.attn_type == "standard":
            energy = torch.matmul(Q, K.transpose(-1,-2)) / math.sqrt(self.head_dim)
            if mask is not None:
                energy = energy.masked_fill(mask==0, float('-inf'))
            attn = torch.softmax(energy, dim=-1)
        elif self.attn_type == "relative":
            rel_pos = torch.arange(-seq+1, seq).to(q.device)
            rel_pos = rel_pos.unsqueeze(0).unsqueeze(0)
            energy = torch.matmul(Q, K.transpose(-1,-2)) / math.sqrt(self.head_dim)
            energy = energy + rel_pos[:, :, :seq, :seq].float()
            attn = torch.softmax(energy, dim=-1)
        elif self.attn_type == "sparse":
            energy = torch.zeros(N, self.heads, seq, seq).to(q.device)
            for i in range(seq):
                start = max(0, i-self.window)
                end = min(seq, i+self.window+1)
                # Explicitly add and remove a dimension for batched matrix-vector multiplication
                matmul_result = torch.matmul(Q[:,:,i,:].unsqueeze(-2), K[:,:,start:end,:].transpose(-1,-2)).squeeze(-2)
                energy[:,:,i,start:end] = matmul_result / math.sqrt(self.head_dim)
            attn = torch.softmax(energy, dim=-1)
        elif self.attn_type == "linear":
            Q_ = F.elu(Q) + 1
            K_ = F.elu(K) + 1
            KV = torch.matmul(K_.transpose(-1,-2), V)
            attn = torch.matmul(Q_, KV)
            attn = attn / seq
        out = torch.matmul(attn, V) if self.attn_type != "linear" else attn
        out = out.transpose(1,2).contiguous().view(N, seq, self.embed)
        return self.out(out)

class FF(nn.Module):
    def __init__(self, embed, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(embed, hidden), nn.ReLU(), nn.Linear(hidden, embed))
    def forward(self,x):
        return self.net(x)

class EncoderLayer(nn.Module):
    def __init__(self, embed, heads, hidden, drop, attn_type="standard"):
        super().__init__()
        self.attn = MultiHeadAttn(embed, heads, attn_type)
        self.ff = FF(embed, hidden)
        self.norm1 = nn.LayerNorm(embed)
        self.norm2 = nn.LayerNorm(embed)
        self.drop = nn.Dropout(drop)
    def forward(self,x,mask=None):
        x = self.norm1(x + self.drop(self.attn(x,x,x,mask)))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, embed, heads, hidden, drop, attn_type="standard"):
        super().__init__()
        self.self_attn = MultiHeadAttn(embed, heads, attn_type)
        self.cross_attn = MultiHeadAttn(embed, heads, "standard")
        self.ff = FF(embed, hidden)
        self.norm1 = nn.LayerNorm(embed)
        self.norm2 = nn.LayerNorm(embed)
        self.norm3 = nn.LayerNorm(embed)
        self.drop = nn.Dropout(drop)
    def forward(self,x,enc,src_mask=None,trg_mask=None):
        x = self.norm1(x + self.drop(self.self_attn(x,x,x,trg_mask)))
        x = self.norm2(x + self.drop(self.cross_attn(enc,enc,x,src_mask)))
        x = self.norm3(x + self.drop(self.ff(x)))
        return x

class Encoder(nn.Module):
    def __init__(self,vocab,embed,layers,heads,hidden,drop,max_len=100,attn_type="standard"):
        super().__init__()
        self.embed = nn.Embedding(vocab, embed)
        self.pos = nn.Embedding(max_len, embed)
        self.layers = nn.ModuleList([EncoderLayer(embed,heads,hidden,drop,attn_type) for _ in range(layers)])
        self.drop = nn.Dropout(drop)
    def forward(self,x,mask=None):
        N, seq = x.shape
        pos = torch.arange(0,seq).unsqueeze(0).expand(N,seq).to(x.device)
        x = self.drop(self.embed(x) + self.pos(pos))
        for layer in self.layers:
            x = layer(x,mask)
        return x

class Decoder(nn.Module):
    def __init__(self,vocab,embed,layers,heads,hidden,drop,max_len=100,attn_type="standard"):
        super().__init__()
        self.embed = nn.Embedding(vocab,embed)
        self.pos = nn.Embedding(max_len,embed)
        self.layers = nn.ModuleList([DecoderLayer(embed,heads,hidden,drop,attn_type) for _ in range(layers)])
        self.out = nn.Linear(embed,vocab)
        self.drop = nn.Dropout(drop)
    def forward(self,x,enc,src_mask=None,trg_mask=None):
        N, seq = x.shape
        pos = torch.arange(0,seq).unsqueeze(0).expand(N,seq).to(x.device)
        x = self.drop(self.embed(x)+self.pos(pos))
        for layer in self.layers:
            x = layer(x,enc,src_mask,trg_mask)
        return self.out(x)

class Transformer(nn.Module):
    def __init__(self,src_vocab,trg_vocab,embed=256,layers=2,heads=8,hidden=512,drop=0.1,max_len=100,attn_type="standard"):
        super().__init__()
        self.enc = Encoder(src_vocab,embed,layers,heads,hidden,drop,max_len,attn_type)
        self.dec = Decoder(trg_vocab,embed,layers,heads,hidden,drop,max_len,attn_type)
    def make_trg_mask(self,trg):
        N, seq = trg.shape
        mask = torch.tril(torch.ones(seq,seq)).expand(N,1,seq,seq)
        return mask.to(trg.device)
    def forward(self,src,trg):
        trg_mask = self.make_trg_mask(trg)
        enc_out = self.enc(src)
        out = self.dec(trg,enc_out,trg_mask=trg_mask)
        return out

src_vocab = 1000
trg_vocab = 1000
model = Transformer(src_vocab,trg_vocab,attn_type="sparse")

src = torch.randint(0,src_vocab,(2,10))
trg = torch.randint(0,trg_vocab,(2,10))

out = model(src,trg)
print(out.shape)
