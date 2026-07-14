import torch
import torch.nn as nn

ACTIVATIONS = {
    "softplus":   nn.Softplus,
    "relu":       nn.ReLU,
    "gelu":       nn.GELU,
    "silu":       nn.SiLU,
    "leaky_relu": nn.LeakyReLU,
}

class BottleneckAttention(nn.Module):
    def __init__(self, dim, h, w, num_heads=4, depth=2):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, h * w, dim))
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim, nhead=num_heads, dim_feedforward=dim * 4,
                batch_first=True, activation="gelu",
            )
            for _ in range(depth)
        ])

    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2) + self.pos_embed
        for block in self.blocks:
            tokens = block(tokens)
        # MPS backward mishandles reshape-of-transpose here; force an explicit copy.
        return tokens.transpose(1, 2).contiguous().view(B, C, H, W)


class VAE(nn.Module):
    def __init__(self, input_hw=(48, 48), latent=4, channels=(32, 64, 128, 256), kernel=5, stride=2, padding=2,  activation="softplus", p_enc=0.0,
                 use_bottleneck_attn=False, attn_after_stage=None, attn_heads=4, attn_depth=2):
        super(VAE, self).__init__()

        in_h, in_w = input_hw
        layers = []
        in_ch = 2
        scale = 2 ** len(channels)

        act_cls = ACTIVATIONS[activation]

        for out_ch in channels:
            layers.append(nn.Conv2d(in_ch, out_ch, kernel, stride, padding))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(act_cls())
            layers.append(nn.Dropout2d(p_enc))
            in_ch = out_ch

        self.use_bottleneck_attn = use_bottleneck_attn
        if use_bottleneck_attn:
            stage = attn_after_stage if attn_after_stage is not None else len(channels) - 2
            split = (stage + 1) * 4  # 4 modules per stage: conv/bn/act/dropout
            self.encoder_stem = nn.Sequential(*layers[:split])
            self.encoder_tail = nn.Sequential(*layers[split:])
            attn_scale = 2 ** (stage + 1)
            self.bottleneck_attn = BottleneckAttention(
                channels[stage], in_h // attn_scale, in_w // attn_scale,
                num_heads=attn_heads, depth=attn_depth,
            )
        else:
            self.encoder = nn.Sequential(*layers)

        self.h_enc = in_h // scale
        self.w_enc = in_w // scale

        flat_size = channels[-1] * self.h_enc * self.w_enc

        self.fc_mu = nn.Linear(flat_size, latent)
        self.fc_logvar = nn.Linear(flat_size, latent)

        self.fc_dec = nn.Linear(latent, flat_size)

        dec_channels = list(reversed(channels))

        layers = []

        for in_ch, out_ch in zip(dec_channels, dec_channels[1:]):
            layers.append(nn.ConvTranspose2d(in_ch, out_ch, kernel, stride, padding, output_padding=1))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(act_cls())
            in_ch = out_ch

        layers.append(nn.ConvTranspose2d(in_ch, 2, kernel, stride, padding, output_padding=1))
        layers.append(act_cls())

        self.decoder = nn.Sequential(*layers)

        self.bottleneck_ch = channels[-1]
    
    def encode(self, x):
        if self.use_bottleneck_attn:
            h = self.encoder_tail(self.bottleneck_attn(self.encoder_stem(x)))
        else:
            h = self.encoder(x)
        h = h.flatten(1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.fc_dec(z)
        h = h.view(z.size(0), self.bottleneck_ch, self.h_enc, self.w_enc)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z