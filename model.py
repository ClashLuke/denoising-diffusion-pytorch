import math

import torch
from torch import nn


@torch.jit.script
def mish(inp):
    return inp.mul(torch.nn.functional.softplus(inp).tanh())


@torch.no_grad()
def variance_scaling_init_(tensor, scale=1, mode="fan_avg", distribution="uniform"):
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)

    if mode == "fan_in":
        scale /= fan_in

    elif mode == "fan_out":
        scale /= fan_out

    else:
        scale /= (fan_in + fan_out) / 2

    if distribution == "normal":
        std = math.sqrt(scale)

        return tensor.normal_(0, std)

    else:
        bound = math.sqrt(3 * scale)

        return tensor.uniform_(-bound, bound)


def conv2d(in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True, scale=1., mode="fan_avg",
           transpose=False):
    conv_class = getattr(nn, f'Conv{"Transpose" if transpose else ""}2d')
    conv = conv_class(in_channel, out_channel, kernel_size, stride=stride, padding=padding, bias=bias)

    variance_scaling_init_(conv.weight, scale, mode=mode)

    if bias:
        nn.init.zeros_(conv.bias)

    return conv


def linear(in_channel, out_channel, scale=1, mode="fan_avg"):
    lin = nn.Linear(in_channel, out_channel)

    variance_scaling_init_(lin.weight, scale, mode=mode)
    nn.init.zeros_(lin.bias)

    return lin


class Mish(torch.jit.ScriptModule):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return mish(input)


def upsample(channel):
    return conv2d(channel, channel, 4, stride=2, padding=1, transpose=True)


def downsample(channel):
    return conv2d(channel, channel, 3, stride=2, padding=1)


@torch.jit.script
def nothing(inp):
    return inp


class ResBlock(torch.jit.ScriptModule):
    def __init__(self, in_channel, out_channel, time_dim, dropout):
        super().__init__()

        self.norm1 = nn.GroupNorm(32, in_channel)
        self.conv1 = conv2d(in_channel, out_channel, 3, padding=1)

        self.time = linear(time_dim, out_channel)

        self.norm2 = nn.GroupNorm(32, out_channel)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = conv2d(out_channel, out_channel, 3, padding=1, scale=1e-10)
        self.skip = conv2d(in_channel, out_channel, 1) if in_channel != out_channel else nothing

    def forward(self, input, time):
        batch = input.shape[0]

        out = self.conv1(mish(self.norm1(input)))

        out = out + self.time(mish(time)).view(batch, -1, 1, 1)

        out = self.conv2(self.dropout(mish(self.norm2(out))))

        return out + self.skip(input)


class SelfAttention(torch.jit.ScriptModule):
    def __init__(self, in_channel, heads=16):
        super().__init__()

        self.norm = nn.GroupNorm(32, in_channel)
        self.weight = torch.nn.Parameter(torch.randn(2 * (in_channel * heads + heads) + in_channel, in_channel))
        self.gate = torch.nn.Parameter(torch.zeros(1))
        self.heads = heads
        self.in_channel = in_channel

    def forward(self, inp):
        batch, channel, height, width = inp.shape

        out = self.weight.unsqueeze(0).expand(batch, -1, -1).bmm(self.norm(inp).view(batch, channel, -1))
        lin = out[:, :self.in_channel]
        key = out[:, self.in_channel:self.in_channel * (1 + self.heads)].view(batch, channel, self.heads,
                                                                              height * width)
        query = out[:, self.in_channel * (1 + self.heads):
                       self.in_channel + 2 * self.in_channel * self.heads].view(batch, channel, self.heads,
                                                                                height * width)
        key_choice = out[:, self.in_channel + 2 * self.in_channel * self.heads:
                            self.in_channel + 2 * self.in_channel * self.heads + self.heads].softmax(1)
        query_choice = out[:, self.in_channel + 2 * self.in_channel * self.heads + self.heads:].softmax(1)

        key = key.mul(key_choice.unsqueeze(1)).sum(1)
        query = query.mul(query_choice.unsqueeze(1)).sum(1)

        key = key.softmax(2).bmm(query.transpose(1, 2))
        lin = lin.bmm(key)
        out = lin * self.gate + inp
        return out


class TimeEmbedding(torch.jit.ScriptModule):
    def __init__(self, dim):
        super().__init__()

        self.dim = dim

        inv_freq = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000) / dim))

        self.register_buffer("inv_freq", inv_freq)

    def forward(self, input):
        shape = input.shape
        sinusoid_in = torch.ger(input.view(-1).float(), self.inv_freq)
        pos_emb = torch.cat([sinusoid_in.sin(), sinusoid_in.cos()], dim=-1)
        pos_emb = pos_emb.view(*shape, self.dim)

        return pos_emb


class ResBlockWithAttention(torch.jit.ScriptModule):
    def __init__(self, in_channel, out_channel, time_dim, dropout, use_attention=False):
        super().__init__()

        self.resblocks = ResBlock(in_channel, out_channel, time_dim, dropout)

        self.attention = SelfAttention(out_channel) if use_attention else nothing

    def forward(self, input, time):
        out = self.resblocks(input, time)
        out = self.attention(out)
        return out


def spatial_fold(input, fold):
    if fold == 1:
        return input

    batch, channel, height, width = input.shape
    h_fold = height // fold
    w_fold = width // fold

    return input.view(batch, channel, h_fold, fold, w_fold,
                      fold).permute(0, 1, 3, 5, 2, 4).reshape(batch, -1, h_fold, w_fold)


def spatial_unfold(input, unfold):
    if unfold == 1:
        return input

    batch, channel, height, width = input.shape
    h_unfold = height * unfold
    w_unfold = width * unfold

    return input.view(batch, -1, unfold, unfold, height,
                      width).permute(0, 1, 4, 2, 5, 3).reshape(batch, -1, h_unfold, w_unfold)


class UNet(torch.jit.ScriptModule):
    def __init__(
            self,
            in_channel,
            channel,
            channel_multiplier,
            n_res_blocks,
            attn_strides,
            dropout=0,
            fold=1,
    ):
        super().__init__()

        self.fold = fold

        time_dim = channel * 4

        n_block = len(channel_multiplier)

        self.time = nn.Sequential(
            TimeEmbedding(channel),
            linear(channel, time_dim),
            Mish(),
            linear(time_dim, time_dim),
        )

        down_layers = [conv2d(in_channel * (fold ** 2), channel, 3, padding=1)]
        feat_channels = [channel]
        in_channel = channel
        for i in range(n_block):
            for _ in range(n_res_blocks):
                channel_mult = channel * channel_multiplier[i]

                down_layers.append(
                    ResBlockWithAttention(
                        in_channel,
                        channel_mult,
                        time_dim,
                        dropout,
                        use_attention=2 ** i in attn_strides,
                    )
                )

                feat_channels.append(channel_mult)
                in_channel = channel_mult

            if i != n_block - 1:
                down_layers.append(downsample(in_channel))
                feat_channels.append(in_channel)

        self.down = nn.ModuleList(down_layers)

        self.mid = nn.ModuleList(
            [
                ResBlockWithAttention(
                    in_channel,
                    in_channel,
                    time_dim,
                    dropout=dropout,
                    use_attention=True,
                ),
                ResBlockWithAttention(
                    in_channel, in_channel, time_dim, dropout=dropout
                ),
            ]
        )

        up_layers = []
        for i in reversed(range(n_block)):
            for _ in range(n_res_blocks + 1):
                channel_mult = channel * channel_multiplier[i]

                up_layers.append(
                    ResBlockWithAttention(
                        in_channel + feat_channels.pop(),
                        channel_mult,
                        time_dim,
                        dropout=dropout,
                        use_attention=2 ** i in attn_strides,
                    )
                )

                in_channel = channel_mult

            if i != 0:
                up_layers.append(upsample(in_channel))

        self.up = nn.ModuleList(up_layers)

        self.out_norm = nn.GroupNorm(32, in_channel)
        self.out_conv = conv2d(in_channel, 3 * (fold ** 2), 3, padding=1, scale=1e-10)

    def forward(self, input, time):
        time_embed = self.time(time)

        feats = []

        out = spatial_fold(input, self.fold)
        for layer in self.down:
            if isinstance(layer, ResBlockWithAttention):
                out = layer(out, time_embed)

            else:
                out = layer(out)

            feats.append(out)

        for layer in self.mid:
            out = layer(out, time_embed)

        for layer in self.up:
            if isinstance(layer, ResBlockWithAttention):
                out = layer(torch.cat((out, feats.pop()), 1), time_embed)

            else:
                out = layer(out)

        out = self.out_conv(mish(self.out_norm(out)))
        out = spatial_unfold(out, self.fold)

        return out
