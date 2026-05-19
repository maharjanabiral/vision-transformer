import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbeddings(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, bias=True):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        self.bias = bias

        self.projection = nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, attn_p=0.0, proj_p=0.0, flash_attention=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_p = attn_p
        self.proj_p = proj_p
        self.flash_attention = flash_attention

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads

        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(attn_p)
        self.proj_drop = nn.Dropout(proj_p)

        self.projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape

        q = self.q(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if self.flash_attention:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_p if self.training else 0.0)

        att = (q @ k.transpose(-2,-1)) * (self.head_dim ** -0.5)
        att = att.softmax(dim=-1)
        att = self.attn_drop(att)
        x = att @ v
        x = x.transpose(1,2).reshape(batch_size, seq_len, self.num_heads * self.head_dim)

        out = self.projection(x)
        out = self.proj_drop(out)
        return out


class FeedForwardNetwork(nn.Module):
    def __init__(self, in_features=768, mlp_ratio=4, mlp_p=0.0):
        super().__init__()
        self.in_features = in_features
        self.mlp_ratio = mlp_ratio
        self.mlp_p = mlp_p

        self.ln1 = nn.Linear(in_features, mlp_ratio * in_features)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(mlp_p)

        self.ln2 = nn.Linear(mlp_ratio * in_features, in_features)
        self.dropout2 = nn.Dropout(mlp_p)

    def forward(self, x):
        x = self.gelu(self.ln1(x)) 
        x = self.dropout1(x)
        x = self.ln2(x)
        x = self.dropout2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=768):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention_block = MultiHeadedSelfAttention()

        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward_network = FeedForwardNetwork()

    def forward(self, x):
        x = x + self.attention_block(self.norm1(x))
        x = x + self.feed_forward_network(self.norm2(x)) 
        return x

class VisionTransformer(nn.Module):
    def __init__(self, embed_dim=768, pos_p=0.0, n_layers=12, head_p=0.0, num_classes=100):
        super().__init__()
        
        self.patch_embeddings = PatchEmbeddings()
        self.pos_dropout = nn.Dropout(pos_p)

        self.norm = nn.LayerNorm(embed_dim)
        self.head_drop = nn.Dropout(head_p)

        num_tokens = self.patch_embeddings.num_patches + 1
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embeddings = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(n_layers)])
        
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):

        x = self.patch_embeddings(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embeddings
        x = self.pos_dropout(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        x = x[:, 0]
        x = self.head_drop(x)
        x = self.head(x)

        return x


if __name__ == "__main__":
    rand = torch.randn(4, 3, 224, 224)
    model = VisionTransformer()
    out = model(rand)
    print(out.shape)



