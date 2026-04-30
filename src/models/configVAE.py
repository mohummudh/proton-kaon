import torch
import torch.nn as nn

ACTIVATIONS = {
    "softplus":   nn.Softplus,
    "relu":       nn.ReLU,
    "gelu":       nn.GELU,
    "silu":       nn.SiLU,
    "leaky_relu": nn.LeakyReLU,
}

class VAE(nn.Module):
    def __init__(self, input_hw=(48, 48), latent=4, channels=(32, 64, 128, 256), kernel=5, stride=2, padding=2,  activation="softplus", p_enc=0.0):
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