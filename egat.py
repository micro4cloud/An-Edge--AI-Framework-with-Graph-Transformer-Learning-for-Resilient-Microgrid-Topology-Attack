import torch  
import torch.nn as nn  
import math  
 
 
class EGAT(nn.Module):  
 
    def __init__(  
        self, 
        input_dim,  
        hidden_dim,  
        num_heads,  
        num_classes 
    ): 
 
        super(EGAT, self).__init__()  
 
        self.cnn = nn.Conv1d(  
            input_dim,  
            hidden_dim,  
            kernel_size=3,  
            padding=1  
        ) 
 
        self.rnn = nn.GRU(  
            input_size=input_dim,  
            hidden_size=hidden_dim,  
            batch_first=True  
        ) 
 

        self.proj = nn.Linear(  
            hidden_dim * 2,  
            hidden_dim  
        ) 
 
        self.attn = nn.MultiheadAttention(  
            embed_dim= hidden_dim,  
            num_heads=num_heads,  
            batch_first=True  
        ) 
 
        self.ffn = nn.Sequential(  
            nn.Linear(hidden_dim, hidden_dim),  
            nn.ReLU(),  
            nn.Linear(hidden_dim, hidden_dim)  
        ) 
 
        self.norm = nn.LayerNorm(hidden_dim)  
 
        self.classifier = nn.Linear(  
            hidden_dim,  
            num_classes  
        ) 
 
    def forward(self, x):  
 
        cnn_in = x.permute(0, 2, 1)  
 
        h_cnn = self.cnn(cnn_in)  
 
        h_cnn = h_cnn.pe rmute(0, 2, 1)  
 
        h_rnn, _ = self.rnn(x)  
 
        h_fuse = torch.cat([h_cnn, h_rnn], dim= -1) 
 
        z0 = self.proj(h_fuse)  
 
        attn_out, _ = self.attn(z0, z0, z0)  
 
        out = self.norm(  
            z0 + attn_out + self.ffn(attn_out)  
        ) 
 
        out = out.mean(dim=1)  
 
        logits = self.classifier(out)  
 
        return torch.softmax(logits, dim=1)
