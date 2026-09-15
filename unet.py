import math
import torch
import torch as nn
import torch.nn.functional as F
from torch.nn.functional import embedding
from transformers import kernelize
from transformers.models.reformer.modeling_reformer import PositionEmbeddings

# ==============================================================================
# 1. TIME AND CLASS EMBEDDINGS
# ==============================================================================

class SinusoidalPositionEmbeddings(nn.Module):

    """
    converts an integer time step t into embeddings using sinusoidal position encoding
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim


    def forward(self, time):
        device = time.device
        half_dim = self.dim//2
        embeddings = math.log(10000) / (half_dim-1)
        embeddings = torch.exp(torch.arange(half_dim, device=device)* -embeddings)
        embeddings = torch.cat((embeddings.sin(),embeddings.cos()), dim=-1)
        return embeddings

# ==============================================================================
# 2. BUILDING BLOCKS (RESNET & ATTENTION)
# ==============================================================================

class ResBlock(nn.Module):
    """
    standard residual block for time and class embeds as inputs embeds are added to feature maps for condition generation
    """

    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_channels, out_channels),
        )

        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

        self.block2 = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

        self.res_conv = nn.Conv2d(in_channels, out_channels,1) if in_channels != out_channels else nn.Identity()

    def forward(self, x,time_emb):
        h = self.block1(x)
        #add time class embed condition
        condition = self.mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + condition
        h = self.block2(h)
        return h + self.res_conv(h)


class AttentionBlock(nn.Module):
    def __init__(self, channels,num_heads = 4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels,channels*3,kernel_size = 1)
        self.proj = nn.Conv2d(channels,channels,kernel_size = 1)


    def forward(self, x):
        B, C, H, W = x.shape
        #compute q k v
        qkv = self.qkv(self.norm(x))
        q, k, v = torch.chunk(qkv, 3, dim = -1)

        #Reshape for multihead attention
        head_dim = C // self.num_heads
        q = q.view(B,self.num_heads,head_dim,H*W).transpose(-2,-1)
        k = k.view(B,self.num_heads,head_dim,H*W)
        v = v.view(B,self.num_heads,head_dim,H*W).transpose(-2,-1)

        scale = 1.0/math.sqrt(head_dim)
        attn = torch.softmax(torch.matmul(q, k), dim = -1)
        out = torch.matmul(attn, v)

        out = out.transpose(-2,-1).reshape(B,C,H,W)
        return x + self.proj(out)#skip connection



# ==============================================================================
# 3. MAIN UNET ARCHITECTURE
# ==============================================================================


class UNet(nn.Module):
    def __init__(
            self,
            in_channels=3,
            model_channels=128,
            out_channels=3,
            num_classes=10,  # CIFAR-10 has 10 classes
            channel_mult=(1, 2, 2, 2),  # Channels: 128 -> 256 -> 256 -> 256
            attention_res=(16, 8)  # Apply attention at these spatial resolutions
    ):
        super().__init__()

        # TIME & CLASS EMBEDDING
        time_emb_dim = model_channels*4
        self.time_emb = nn.Sequential(
            SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(model_channels,time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim,time_emb_dim)
        )
        # Class embedding (will be added to time embedding)

        self.class_emb = nn.Embedding(num_classes, time_emb_dim)

        #initial conv
        self.init_conv = nn.Conv2d(in_channels,model_channels, kernel_size=3, padding=1)

        #Down blocks
        self.downs = nn.ModuleList()
        current_res = 32 #cifar 32x32 as input
        in_channels = model_channels
        self.down_block_channels = [in_channels]

        for mult in channel_mult:
            out_channels = model_channels*mult

            # Add ResBlock and optionally Attention
            layer = nn.ModuleList([ResBlock(in_channels, out_channels, time_emb_dim)])
            if current_res in attention_res:
                layer.append(AttentionBlock(out_channels))

            # Downsample using Conv2d with stride 2 (except for the last stage)
            is_last = (mult == channel_mult[-1])
            if not is_last:
                layer.append(nn.Conv2d(out_channels,out_channels,kernel_size= 3 ,stride= 2,padding = 1))
                current_res //=2


            self.downs.append(layer)
            in_channels = out_channels
            self.down_block_channels.append(in_channels)

        # MIDDLE BLOCKS
        self.mid = nn.ModuleList([
            ResBlock(in_channels
                     , in_channels, time_emb_dim),
            AttentionBlock(in_channels),
            ResBlock(in_channels, in_channels, time_emb_dim)
        ])

        # UP BLOCKS
        self.ups = nn.ModuleList()
        # Reverse the channel multipliers for upsampling
        for i, mult in reversed(list(enumerate(channel_mult))):
            out_channels = model_channels * mult

            layer = nn.ModuleList([
                # + in_ch because of the skip connection from the down blocks
                ResBlock(in_channels + self.down_block_channels.pop(), out_channels, time_emb_dim)
            ])

            if current_res in attention_res:
                layer.append(AttentionBlock(out_channels))

            # Upsample using Transposed Conv (except for the first stage in reverse)
            if i != 0:
                layer.append(nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1))
                current_res *= 2

            self.ups.append(layer)
            in_ch = out_channels

        # FINAL OUTPUT
        self.final_conv = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, time , class_labels):
        # 1. Condition computation (Time + Class)
        time_emb = self.time_emb(x)
        class_emb = self.class_emb(class_labels)
        condition_emb = time_emb + class_emb

        # 2. Initial convolution
        x = self.init_conv(x)
        skip_connections = [x]

        # 3. Down path
        for block in self.downs:
            x = block[0](x, condition_emb)  # ResBlock
            if len(block) > 1 and isinstance(block[1], AttentionBlock):
                x = block[1](x)  # AttentionBlock
            if len(block) > 1 and isinstance(block[-1], nn.Conv2d):
                x = block[-1](x)  # Downsample Conv
            skip_connections.append(x)

            # 4. Middle path
        x = self.mid[0](x, condition_emb)
        x = self.mid[1](x)
        x = self.mid[2](x, condition_emb)

        # 5. Up path
        for block in self.ups:
            # Concatenate skip connection
            skip_x = skip_connections.pop()
            x = torch.cat([x, skip_x], dim=1)

            x = block[0](x, condition_emb)  # ResBlock
            if len(block) > 1 and isinstance(block[1], AttentionBlock):
                x = block[1](x)  # AttentionBlock
            if len(block) > 1 and isinstance(block[-1], nn.ConvTranspose2d):
                x = block[-1](x)  # Upsample Conv

        # 6. Final output
        return self.final_conv(x)




