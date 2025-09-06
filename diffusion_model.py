from typing import Tuple, Union
import mlx.core as mx
import mlx.nn as nn
import mlx.nn.init as init
import numpy as np
from MLXCrossAttention import CrossAttentionBlock

VAE_LATENT_SPACE_SIZE=128
UNET_TEXT_EMBEDDING_SIZE=384

# Other training parameters
PRED_CLAMPING=10.0

nl_activation = nn.SELU


def lecun_normal(tensor):
    """Function LeCun normal initialization for mx.array"""
    shape = tensor.shape
    if len(shape) < 2:
        raise ValueError("LeCun initialization requires at least 2 dimensions")

    # Calculate fan-in
    if len(shape) == 2:
        # Linear layer: (out_features, in_features)
        fan_in = shape[1]
    else:
        # Conv layer: (out_channels, in_channels, *kernel_size)
        fan_in = np.prod(shape[1:])

    std = 1.0 / np.sqrt(fan_in)
    values = mx.random.normal(shape=shape) * std
    return values
    
    
class AlphaDropout(nn.Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p
        self.alpha = 1.6732632423543772
        self.scale = 1.0507009873554805
        self.alpha_prime = -self.scale * self.alpha  # ≈ -1.7580993408473766

    def __call__(self, x):
        if self.p == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.p
        mask = mx.random.uniform(shape=x.shape) < keep_prob

        # Apply mask and correct mean/variance
        x_keep = x * mask / keep_prob
        x_drop = self.alpha_prime * (1.0 - mask)

        return x_keep + x_drop

def adaptive_average_pool2d(x: mx.array,  output_size: tuple) -> mx.array:
    B, H, W, C = x.shape
    x = x.reshape(B, H // output_size[0], output_size[0], W // output_size[1], output_size[1], C)
    x = mx.mean(x, axis=(1, 3))
    return x


class AdaptiveAveragePool2D(nn.Module):
    def __init__(self, output_size: Union[int, Tuple[int, int]] = 1):
        super().__init__()
        self.output_size = (
            output_size
            if isinstance(output_size, tuple)
            else (output_size, output_size)
        )

    def __call__(self, x: mx.array) -> mx.array:
        return adaptive_average_pool2d(x, self.output_size)


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(1, channels // reduction)  # prevent zero hidden units
        
#        self.avg_pool = AdaptiveAveragePool2D(1)  # (B, C, 1, 1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nl_activation(),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def __call__(self, x):
        B, H, W, C = x.shape
        pooling = nn.AvgPool2d(kernel_size=(H, W))
        avg = pooling(x)  # Shape: (B, 1, 1, C)
#        avg = self.avg_pool(x)     # Global context per channel
        attn = self.fc(avg)
        return x * attn             # Apply attention

    def init_weights(self):
        """PyTorch-style weight initialization."""
        for layer in self.fc:
            if isinstance(layer, nn.Conv2d):
                layer.weight = lecun_normal(layer.weight)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, group_size = 4, dropout=0.2):
        super().__init__()
        self.groupnorm1 = nn.GroupNorm(group_size, out_channels, pytorch_compatible=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.nl_activation1 = nl_activation()
        self.drop1 = nn.Dropout(p=dropout)
        #self.drop1 = AlphaDropout(p=dropout)
        self.groupnorm2 = nn.GroupNorm(group_size, out_channels, pytorch_compatible=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.nl_activation2 = nl_activation()
        
        if in_channels == out_channels:
            self.residual_layer = lambda x: x
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def __call__(self, x):
        residue = mx.array(x)
        x = self.conv1(x)
        x = self.groupnorm1(x)
        x = self.nl_activation1(x)
        x = self.drop1(x)
        x = self.conv2(x)
        x = self.groupnorm2(x)
        x = self.nl_activation2(x)
        return x + self.residual_layer(residue)

    def init_weights(self):
        """Initialise all Conv2d weights using Kaiming normal init, and GroupNorm weights/biases with zeros."""
        for attr_name in dir(self):
            layer = getattr(self, attr_name)
            if isinstance(layer, nn.Conv2d):
                layer.weight = lecun_normal(layer.weight)
                if layer.bias is not None:
                    layer.bias = mx.zeros(layer.bias.shape)
            elif isinstance(layer, nn.GroupNorm):
                layer.weight = mx.ones(layer.weight.shape)  # Optional: some use zeros
                layer.bias = mx.zeros(layer.bias.shape)


# Variational Autoencoder (VAE)
VAE_CH_LAYER1 = 4
VAE_CH_LAYER2 = 8
VAE_CH_LAYER3 = 16
VAE_CH_LAYER4 = 32
VAE_CH_LAYER5 = 64
VAE_CH_LAYER6 = 128
class VAE(nn.Module):
    def __init__(self, latent_size=VAE_LATENT_SPACE_SIZE, dropout=0.2):
        super().__init__()
        
        self.latent_size=latent_size

        self.encoder = nn.Sequential(
            nn.Conv2d(1, VAE_CH_LAYER1, kernel_size=3, padding=1), # 49x64
#            ChannelAttention(VAE_CH_LAYER1),
            ResidualBlock(VAE_CH_LAYER1, VAE_CH_LAYER2, group_size=2, dropout=dropout),
            nn.Conv2d(VAE_CH_LAYER2, VAE_CH_LAYER2, kernel_size=3, stride=2, padding=1), # 25x32
            ChannelAttention(VAE_CH_LAYER2),
            ResidualBlock(VAE_CH_LAYER2, VAE_CH_LAYER3, group_size=4, dropout=dropout),
            nn.Conv2d(VAE_CH_LAYER3, VAE_CH_LAYER3, kernel_size=3, stride=2, padding=1), # 13x16
            ChannelAttention(VAE_CH_LAYER3),
            ResidualBlock(VAE_CH_LAYER3, VAE_CH_LAYER4, group_size=8, dropout=dropout),
            nn.Conv2d(VAE_CH_LAYER4, VAE_CH_LAYER4, kernel_size=3, stride=2, padding=1), #7x8
            ChannelAttention(VAE_CH_LAYER4),
            ResidualBlock(VAE_CH_LAYER4, VAE_CH_LAYER5, group_size=8, dropout=dropout),
            nn.Conv2d(VAE_CH_LAYER5, VAE_CH_LAYER5, kernel_size=3, stride=2, padding=1), #4x4
            ChannelAttention(VAE_CH_LAYER5),
            ResidualBlock(VAE_CH_LAYER5, VAE_CH_LAYER6, group_size=8, dropout=dropout),
            nn.Conv2d(VAE_CH_LAYER6, VAE_CH_LAYER6, kernel_size=3, stride=2, padding=1), #2x2
            nl_activation(),
        )
        

        self.mean_layer = nn.Linear(VAE_CH_LAYER6*2*2, latent_size)
        
        self.logvar_layer = nn.Linear(VAE_CH_LAYER6*2*2, latent_size)
        
        self.decoder_input = nn.Linear(latent_size, VAE_CH_LAYER6*2*2)
        
        self.decoder = nn.Sequential(
            ChannelAttention(VAE_CH_LAYER6), # 2x2
            ResidualBlock(VAE_CH_LAYER6, VAE_CH_LAYER5, group_size=8, dropout=dropout),
            nn.ConvTranspose2d(VAE_CH_LAYER5, VAE_CH_LAYER5, kernel_size=3, stride=2, padding=1, output_padding=1),  # 4x4
            ChannelAttention(VAE_CH_LAYER5),
            ResidualBlock(VAE_CH_LAYER5, VAE_CH_LAYER4, group_size=8, dropout=dropout),
            nn.ConvTranspose2d(VAE_CH_LAYER4, VAE_CH_LAYER4, kernel_size=3, stride=2, padding=1, output_padding=(0, 1)),  # 7x8
            ChannelAttention(VAE_CH_LAYER4),
            ResidualBlock(VAE_CH_LAYER4, VAE_CH_LAYER3, group_size=8, dropout=dropout),
            nn.ConvTranspose2d(VAE_CH_LAYER3, VAE_CH_LAYER3, kernel_size=3, stride=2, padding=1, output_padding=(0, 1)),  # 13x16
            ChannelAttention(VAE_CH_LAYER3),
            ResidualBlock(VAE_CH_LAYER3, VAE_CH_LAYER2, group_size=4, dropout=dropout),
            nn.ConvTranspose2d(VAE_CH_LAYER2, VAE_CH_LAYER2, kernel_size=3, stride=2, padding=1, output_padding=(0, 1)),  # 25x32
            ChannelAttention(VAE_CH_LAYER2),
            ResidualBlock(VAE_CH_LAYER2, VAE_CH_LAYER1, group_size=2, dropout=dropout),
            nn.ConvTranspose2d(VAE_CH_LAYER1, 1, kernel_size=3, stride=2, padding=1, output_padding=(0, 1)),  # 49x64
        )

    def encode(self, x):  # x: (B, 49, 64, 1)
        hidden = self.encoder(x).transpose(0, 3, 1, 2).reshape(x.shape[0], 1, VAE_CH_LAYER6*2*2)
        mean = self.mean_layer(hidden)
        logvar = mx.clip(self.logvar_layer(hidden), -30, 20)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = mx.exp(0.5 * logvar)
        eps = mx.random.normal(shape=std.shape)
        return mean + eps * std

    def decode(self, z):  # z: (B, latent_channels, H, W)
        hidden = self.decoder_input(z).reshape(z.shape[0], VAE_CH_LAYER6, 2, 2).transpose(0, 2, 3, 1)
        return self.decoder(hidden)

    def __call__(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        recon = self.decode(z)
        return recon, mean, logvar, z

    def init_weights(self):
        # Traverse encoder and decoder to initialize inner modules
        for block in self.encoder:
            if hasattr(block, "init_weights"):
                block.init_weights()
            elif isinstance(block, nn.Conv2d):
                block.weight = lecun_normal(block.weight.shape)
                if block.bias is not None:
                    block.bias = mx.zeros(block.bias.shape)

        for block in self.decoder:
            if hasattr(block, "init_weights"):
                block.init_weights()
            elif isinstance(block, (nn.Conv2d, nn.ConvTranspose2d)):
                block.weight = lecun_normal(block.weight.shape)
                if block.bias is not None:
                    block.bias = mx.zeros(block.bias.shape)

        # Linear layers for latent space
        self.mean_layer.weight = lecun_normal(self.mean_layer.weight)
        self.mean_layer.bias = mx.zeros(self.mean_layer.bias.shape)

        self.logvar_layer.weight = lecun_normal(self.logvar_layer.weight)
        self.logvar_layer.bias = mx.zeros(self.logvar_layer.bias.shape)

        self.decoder_input.weight = lecun_normal(self.decoder_input.weight)
        self.decoder_input.bias = mx.zeros(self.decoder_input.bias.shape)











class LinearNoiseScheduler:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps
        self.betas = mx.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = mx.cumprod(self.alphas, axis=0)

#    def add_noise(self, x_start, noise, timesteps):
#        alphas_bar_t = mx.expand_dims(self.alpha_bars[timesteps.squeeze().long()], axis=1)
#        #print(f"x_start.shape={x_start.shape}, noise.shape={noise.shape}, timesteps.shape={timesteps.shape}, alphas_bar_t.shape={alphas_bar_t.shape}")
#        return mx.sqrt(alphas_bar_t).view(-1, 1, 1, 1) * x_start + mx.sqrt(1 - alphas_bar_t).view(-1, 1, 1, 1) * noise

    def add_noise(self, x_start, noise, timesteps):
        # Ensure timesteps is integer type for indexing
        timesteps = timesteps.squeeze().astype(dtype=mx.int32)
        
        # Select alphas_bar_t using advanced indexing
        alphas_bar_t = mx.take(self.alpha_bars, timesteps)

        # Reshape to broadcast over image dimensions (B, 1, 1)
        alphas_bar_t = mx.reshape(alphas_bar_t, (-1, 1, 1))
        
        sqrt_alpha = mx.sqrt(alphas_bar_t)
        sqrt_one_minus_alpha = mx.sqrt(1.0 - alphas_bar_t)
        
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise
    
#    def predict_start_from_noise(self, x_t, t, predicted_noise):
#        alpha_bar = self.alpha_bars[t.squeeze().long()].view(-1, 1)
#        return (x_t - mx.sqrt(1 - alpha_bar) * predicted_noise) / mx.sqrt(alpha_bar)
    def predict_start_from_noise(self, x_t, t, predicted_noise):
        # Ensure `t` is squeezed and converted to int32 for indexing
        t = t.squeeze().astype(dtype=mx.int32)

        # Gather alpha_bar values using indexing
        alpha_bar = mx.take(self.alpha_bars, t)

        # Reshape for broadcasting over x_t (assumes x_t shape is [B, ...])
        alpha_bar = mx.reshape(alpha_bar, (-1,) + (1,) * (len(x_t.shape) - 1))

        return (x_t - mx.sqrt(1.0 - alpha_bar) * predicted_noise) / mx.sqrt(alpha_bar)


class UNetBlockLayer(nn.Module):
    def __init__(self, input_dim, output_dim, time_dim, dropout):
        super().__init__()
        self.l = nn.Linear(input_dim, output_dim)
        self.t = nn.Linear(time_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.act = nl_activation()
        self.drop = nn.Dropout(dropout)
    
    def __call__(self, x, t_embed):
        h = self.l(x)
        h += self.t(t_embed)
        h = self.norm(h)
        h = self.act(h)
        return self.drop(h)
    
    def init_weights(self):
        self.l.weight = lecun_normal(self.l.weight)
        if self.l.bias is not None:
            self.l.bias = mx.zeros(self.l.bias.shape)
        self.t.weight = lecun_normal(self.t.weight)
        if self.t.bias is not None:
            self.t.bias = mx.zeros(self.t.bias.shape)


class UNetBlock(nn.Module):
    def __init__(self, latent_dim=VAE_LATENT_SPACE_SIZE, hidden_dim=512, time_dim=128, num_heads=4, dropout=0.2):
        super().__init__()
        self.latent_dim = latent_dim

        self.cross_attn = CrossAttentionBlock(latent_dim, num_heads, dropout=dropout)
        
        self.layer1 = UNetBlockLayer(latent_dim, hidden_dim, time_dim, dropout)
        self.layer2 = UNetBlockLayer(hidden_dim, hidden_dim, time_dim, dropout)
        self.layer3 = UNetBlockLayer(hidden_dim, hidden_dim, time_dim, dropout)
        self.outlyr = nn.Linear(hidden_dim, latent_dim, bias=False)

    def __call__(self, latent_z, t_emb, conditioning):
        batch_size = latent_z.shape[0]
        out = latent_z
        if conditioning is not None:
            conditioning = conditioning.reshape(batch_size, 1, self.latent_dim)
            out = self.cross_attn(out, conditioning)

        # Feedforward over each token (here just 1)
        out = self.layer1(out, t_emb)
        out = self.layer2(out, t_emb)
        out = self.layer3(out, t_emb)
        out = self.outlyr(out)

        # Return residual
        return out.reshape(latent_z.shape) + latent_z

    def init_weights(self):
        self.cross_attn.init_weights(init_fn=lecun_normal)
        self.layer1.init_weights()
        self.layer2.init_weights()
        self.layer3.init_weights()
        self.outlyr.weight = lecun_normal(self.outlyr.weight)
        


def get_timestep_embedding(timesteps, dim):
    """
    timesteps: mx.array of shape (batch_size,)
    dim: embedding dimension (e.g., 128 or 256)
    """
    half_dim = dim // 2
    emb = mx.log(10000) / (half_dim - 1)
    emb = mx.exp(mx.arange(half_dim) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=1)
    if dim % 2 == 1:
        emb = mx.pad(emb, [(0, 0), (0, 1)])
    return emb.reshape(-1, 1, dim)

# New UNet model for latent space diffusion
class LatentUNet(nn.Module):
    def __init__(self, latent_dim=VAE_LATENT_SPACE_SIZE, text_dim=UNET_TEXT_EMBEDDING_SIZE, hidden_dim=512, time_dim=128, num_heads=4, dropout=0.2):
        super().__init__()
        self.latent_dim = latent_dim
        self.time_dim = time_dim

        self.scheduler = LinearNoiseScheduler()
        
        self.conditioning = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nl_activation(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nl_activation(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim)
        )

        self.res1 = UNetBlock(latent_dim=latent_dim, hidden_dim=hidden_dim, time_dim=time_dim, num_heads=num_heads, dropout=dropout)
        self.res2 = UNetBlock(latent_dim=latent_dim, hidden_dim=hidden_dim, time_dim=time_dim, num_heads=num_heads, dropout=dropout)
        self.res3 = UNetBlock(latent_dim=latent_dim, hidden_dim=hidden_dim, time_dim=time_dim, num_heads=num_heads, dropout=dropout)
        self.res4 = UNetBlock(latent_dim=latent_dim, hidden_dim=hidden_dim, time_dim=time_dim, num_heads=num_heads, dropout=dropout)

    def __call__(self, latent_z, text_embedding, timestep):
        # Reshape latent to (B, L=1, D)
        original_shape = latent_z.shape
        
        t_emb = get_timestep_embedding(timestep, self.time_dim)

        if text_embedding is not None:
            conditioning = self.conditioning(text_embedding)
        else:
            conditioning = None

        out = self.res1(latent_z, t_emb, conditioning)
        out = self.res2(out, t_emb, conditioning)
        out = self.res3(out, t_emb, conditioning)
        out = self.res4(out, t_emb, conditioning)
        return out.reshape(original_shape)

    def init_weights(self):
        # Projectors
        #self.conditioning.weight = lecun_normal(self.conditioning.weight)
        #self.conditioning.bias = mx.zeros(self.conditioning.bias.shape)
        for layer in self.conditioning:
            if isinstance(layer, nn.Linear):
                layer.weight = lecun_normal(layer.weight)
                if layer.bias is not None:
                    layer.bias = mx.zeros(layer.bias.shape)

        self.res1.init_weights()
        self.res2.init_weights()
        self.res3.init_weights()
        self.res4.init_weights()

    def denoise_loop(self, text_embedding, key=None, z0=None, time_start=999, steps=15, cfg=1):
        
        # Build the noisy latent state
        if z0 is None:
            z0 = mx.zeros((1, 1, VAE_LATENT_SPACE_SIZE))
        noise = mx.random.normal(shape=(1, 1, VAE_LATENT_SPACE_SIZE), key=key)
        x = self.scheduler.add_noise(z0, noise, mx.array([time_start]))
        
        # The first step is the noise added to the latent
        yield x, mx.zeros(x.shape)
        
        # Build the denoising schedule
        times = mx.linspace(time_start, 0, steps + 1, dtype=mx.float32)

        for i in range(steps):
            # Get times
            t = times[i].reshape(1, 1)
            next_t = times[i + 1].reshape(1, 1)

            # Predict noise
            predicted_noise = self(x, text_embedding, t)
            
            # If we are doing classifier free guidance, do the prediction here
            if (cfg > 0):
                pred_uncond = self(x, mx.zeros(text_embedding.shape), t)
                predicted_noise = predicted_noise + cfg * (predicted_noise - pred_uncond)

            # Predict x_0
            x0 = self.scheduler.predict_start_from_noise(x, t, predicted_noise)

            # DDIM update
            alpha_t = self.scheduler.alpha_bars[t.astype(mx.int32)].reshape(-1, 1)
            alpha_next = self.scheduler.alpha_bars[next_t.astype(mx.int32)].reshape(-1, 1)
            eps = (x - mx.sqrt(alpha_t) * x0) / mx.sqrt(1 - alpha_t)
            x = mx.sqrt(alpha_next) * x0 + mx.sqrt(1 - alpha_next) * eps
            
            # This is a generator which yields the latent and predicted noise at every step
            # This simplifies debugging and enables other cool stuff like intercepting partially
            # denoised latents for musical things
            yield x, predicted_noise
