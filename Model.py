import torch
from torch import nn
# from torchsummary import summary


class ImgPatches(nn.Module):
    def __init__(self, input_channel=3, dim=1024, patch_size=8):
        super().__init__()
        self.patch_embed = nn.Conv2d(input_channel, dim,
                                     kernel_size=patch_size, stride=patch_size)

    def forward(self, img):
        patches = self.patch_embed(img)
        patches = patches.flatten(2).transpose(1, 2)
        return patches


class Attention(nn.Module):
    def __init__(self, dim, heads=4, attention_dropout=0., proj_dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = 1. / dim ** 0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(proj_dropout)
        )

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, c // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        dot = (q @ k.transpose(-2, -1)) * self.scale
        attn = dot.softmax(dim=-1)
        attn = self.attention_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.out(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_feat, hid_feat=None, out_feat=None,
                 dropout=0.):
        super().__init__()
        if not hid_feat:
            hid_feat = in_feat
        if not out_feat:
            out_feat = in_feat
        self.fc1 = nn.Linear(in_feat, hid_feat)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hid_feat, out_feat)
        self.droprateout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return self.droprateout(x)


class Encoder_Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4, drop_rate=0.):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, drop_rate, drop_rate)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim * mlp_ratio, dropout=drop_rate)

    def forward(self, x):
        x1 = self.ln1(x)
        x = x + self.attn(x1)
        x2 = self.ln2(x)
        x = x + self.mlp(x2)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, depth, dim, heads, mlp_ratio=4, drop_rate=0.):
        super().__init__()
        self.Encoder_Blocks = nn.ModuleList([
            Encoder_Block(dim, heads, mlp_ratio, drop_rate) for _ in range(depth)
        ])

    def forward(self, x):
        for encoder_block in self.Encoder_Blocks:
            x = encoder_block(x)
        return x


class ConvolutionBlockG(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_activation=True, use_norm=True, **kwargs):
        super(ConvolutionBlockG, self).__init__()
        self.convolution = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs) if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels) if use_norm else nn.Identity(),
            nn.ReLU(inplace=True) if use_activation else nn.Identity(),
        )

    def forward(self, x):
        return self.convolution(x)


class Generator(nn.Module):
    def __init__(self, img_channels=3, width=256, height=256, patch_size=8, dim=1024, depth=1, heads=4,
                 mlp_ratio=4, drop_rate=0.):
        super(Generator, self).__init__()
        if width % patch_size != 0 or height % patch_size != 0:
            raise ValueError('Image size must be divisible by patch size.')

        self.num_patches_1d = height // patch_size
        self.num_patches_2d = width // patch_size
        self.patch_size = patch_size
        self.depth = depth
        self.patches = ImgPatches(img_channels, dim, self.patch_size)

        self.dim = dim
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.dropout_rate = drop_rate

        self.positional_embedding = nn.Parameter(torch.zeros(1, self.num_patches_1d * self.num_patches_2d, dim))

        self.TransformerEncoder = TransformerEncoder(depth=self.depth, dim=self.dim, heads=self.heads,
                                                     mlp_ratio=self.mlp_ratio, drop_rate=self.dropout_rate)

        self.up_blocks = nn.ModuleList(
            [
                ConvolutionBlockG(dim, dim // 2, down=False, kernel_size=3, stride=2, padding=1,
                                  output_padding=1),
                ConvolutionBlockG(dim // 2, dim // 4, down=False, kernel_size=3, stride=2, padding=1,
                                  output_padding=1),
                ConvolutionBlockG(dim // 4, dim // 8, down=False, kernel_size=3, stride=2, padding=1,
                                  output_padding=1),
            ]
        )

        self.last = nn.Conv2d(dim // 8, img_channels, kernel_size=7, stride=1, padding=3,
                              padding_mode="reflect")

    def forward(self, x):
        x = self.patches(x)
        x = x + self.positional_embedding

        x = self.TransformerEncoder(x).permute(0, 2, 1).view(-1, self.dim, self.num_patches_1d, self.num_patches_2d)

        for layer in self.up_blocks:
            x = layer(x)

        x = self.last(x)

        return torch.tanh(x)


class ConvolutionBlockD(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ConvolutionBlockD, self).__init__()
        self.convolution = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=True, padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.convolution(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=(64, 128, 256, 512), augmentations=None):
        super(Discriminator, self).__init__()
        self.augmentations = augmentations
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, features[0], 4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.down_blocks = nn.ModuleList(
            [
                ConvolutionBlockD(features[0], features[1], stride=1),
                ConvolutionBlockD(features[1], features[2], stride=2),
                ConvolutionBlockD(features[2], features[3], stride=2),
            ]
        )

        self.last = nn.Conv2d(features[3], 1, 4, stride=1, padding=1, padding_mode="reflect")

    def forward(self, x):
        if self.augmentations is not None:
            x = self.augmentations(x)

        x = self.initial(x)

        for block in self.down_blocks:
            x = block(x)

        x = self.last(x)

        return torch.sigmoid(x)


if __name__ == "__main__":
    gen = Generator(width=512, height=256).to(torch.device("cuda"))
    # print(gen)
    tensor = torch.randn((1, 3, 512, 256)).to(torch.device("cuda"))
    output = gen(tensor)
    print(output.shape)
