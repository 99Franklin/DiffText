# import ipdb
# import torch
# import torch.nn as nn
# import numpy as np
# from torch.functional import F
#
#
# class WithFussionTEAdapter(nn.Module):
#     def __init__(self, channels=[320, 640, 1280, 1280], nums_rb=3, cin=64, ksize=3, sk=True, use_conv=True):
#         super(WithFussionTEAdapter, self).__init__()
#         self.unshuffle = nn.PixelUnshuffle(8)
#         self.channels = channels
#         self.nums_rb = nums_rb
#         self.body = []
#         for i in range(len(channels)):
#             for j in range(nums_rb):
#                 if (i != 0) and (j == 0):
#                     self.body.append(
#                         ResnetBlock(channels[i - 1], channels[i], down=True, ksize=ksize, sk=sk, use_conv=use_conv))
#                 else:
#                     self.body.append(
#                         ResnetBlock(channels[i], channels[i], down=False, ksize=ksize, sk=sk, use_conv=use_conv))
#         self.body = nn.ModuleList(self.body)
#         self.conv_in = nn.Conv2d(cin, channels[0], 3, 1, 1)
#
#         embed_dim = channels[0]
#
#         self.conv_1 = nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=1, bias=False)
#         self.conv_1.weight.data = torch.zeros_like(self.conv_1.weight.data)
#         self.conv_2 = nn.Conv2d(in_channels=embed_dim * 2, out_channels=embed_dim * 2, kernel_size=1, bias=False)
#         self.conv_2.weight.data = torch.zeros_like(self.conv_2.weight.data)
#         self.conv_3 = nn.Conv2d(in_channels=embed_dim * 4, out_channels=embed_dim * 4, kernel_size=1, bias=False)
#         self.conv_3.weight.data = torch.zeros_like(self.conv_3.weight.data)
#         self.conv_4 = nn.Conv2d(in_channels=embed_dim * 4, out_channels=embed_dim * 4, kernel_size=1, bias=False)
#         self.conv_4.weight.data = torch.zeros_like(self.conv_4.weight.data)
#
#         self.zero_conv_list = [self.conv_1, self.conv_2, self.conv_3, self.conv_4]
#
#         char_embed_dim = 512
#
#         # self.embedding = nn.Embedding(100, char_embed_dim)
#         self.embedding = nn.Embedding(100, char_embed_dim * 2)
#         # self.text_pos_embed = nn.Parameter(torch.randn(1, 25, char_embed_dim) * .02)
#         self.text_pos_embed = nn.Parameter(torch.randn(1, 25, char_embed_dim * 2) * .02)
#
#         # self.encoder_layer = nn.TransformerEncoderLayer(char_embed_dim, nhead=8)
#         # self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
#
#         # self.up_embedding = nn.Linear(char_embed_dim, char_embed_dim * 2)
#
#         # self.transformer_decoder_layer = nn.TransformerDecoderLayer(
#         #     d_model=char_embed_dim * 2,
#         #     nhead=8
#         # )
#
#         transformer_decoder_layer = nn.TransformerDecoderLayer(
#             d_model=char_embed_dim * 2,
#             nhead=8
#         )
#
#         self.transformer_decoder = nn.TransformerDecoder(
#             transformer_decoder_layer,
#             num_layers=1
#         )
#
#         # self.init_layers = [self.body, self.transformer_encoder, self.up_embedding]
#         self.init_layers = [self.body]
#         self.initialize_weights()
#
#         self.zero_init_layers = [self.transformer_decoder]
#         self.zero_initialize_weights()
#
#     def initialize_weights(self):
#         for module in self.init_layers:
#             if isinstance(module, torch.nn.Conv2d):
#                 # 初始化卷积层的权重
#                 fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(module.weight)
#                 bound = np.sqrt(2.0 / fan_in)
#                 torch.nn.init.normal_(module.weight, 0, bound)
#                 if module.bias is not None:
#                     torch.nn.init.constant_(module.bias, 0)
#             elif isinstance(module, torch.nn.BatchNorm2d):
#                 # 初始化批归一化层的权重
#                 torch.nn.init.constant_(module.weight, 1)
#                 torch.nn.init.constant_(module.bias, 0)
#             elif isinstance(module, torch.nn.Linear):
#                 # 初始化全连接层的权重
#                 fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(module.weight)
#                 bound = np.sqrt(6.0 / (fan_in + fan_out))
#                 torch.nn.init.uniform_(module.weight, -bound, bound)
#                 if module.bias is not None:
#                     torch.nn.init.constant_(module.bias, 0)
#
#     def zero_initialize_weights(self):
#         for module in self.zero_init_layers:
#             if isinstance(module, torch.nn.BatchNorm2d):
#                 torch.nn.init.constant_(module.weight, 1)
#                 torch.nn.init.constant_(module.bias, 0)
#             elif isinstance(module, torch.nn.Linear):
#                 nn.init.zeros_(module.weight)
#                 nn.init.zeros_(module.bias)
#
#     def forward(self, x, text_embedding, encoder_hidden_states):
#         # unshuffle
#         x = self.unshuffle(x)
#         # extract features
#         features = []
#         x = self.conv_in(x)
#         for i in range(len(self.channels)):
#             for j in range(self.nums_rb):
#                 idx = i * self.nums_rb + j
#                 x = self.body[idx](x)
#             features.append(self.zero_conv_list[i](x))
#
#         text_embedding = self.embedding(text_embedding.long()).to(encoder_hidden_states.dtype)
#         text_embedding = text_embedding + self.text_pos_embed
#
#         text_embedding = text_embedding.permute(1, 0, 2)
#
#         # # Apply Transformer Encoder
#         # text_embedding = self.transformer_encoder(text_embedding)
#
#         # seq_len, bs, dim = text_embedding.shape
#         # text_embedding = text_embedding.reshape(bs * seq_len, dim)
#         # text_embedding = self.up_embedding(text_embedding)
#         # text_embedding = text_embedding.reshape(seq_len, bs, -1)
#
#         text_embedding = self.transformer_decoder(
#             tgt=encoder_hidden_states.permute(1, 0, 2),
#             memory=text_embedding
#         )
#
#         encoder_hidden_states = encoder_hidden_states + text_embedding.permute(1, 0, 2)
#
#         return features, encoder_hidden_states
#
#
# class ResnetBlock(nn.Module):
#     def __init__(self, in_c, out_c, down, ksize=3, sk=False, use_conv=True):
#         super().__init__()
#         ps = ksize // 2
#         if in_c != out_c or sk == False:
#             self.in_conv = nn.Conv2d(in_c, out_c, ksize, 1, ps)
#         else:
#             # print('n_in')
#             self.in_conv = None
#         self.block1 = nn.Conv2d(out_c, out_c, 3, 1, 1)
#         self.act = nn.ReLU()
#         self.block2 = nn.Conv2d(out_c, out_c, ksize, 1, ps)
#         if sk == False:
#             self.skep = nn.Conv2d(in_c, out_c, ksize, 1, ps)
#         else:
#             self.skep = None
#
#         self.down = down
#         if self.down == True:
#             self.down_opt = Downsample(in_c, use_conv=use_conv)
#
#     def forward(self, x):
#         if self.down == True:
#             x = self.down_opt(x)
#         if self.in_conv is not None:  # edit
#             x = self.in_conv(x)
#
#         h = self.block1(x)
#         h = self.act(h)
#         h = self.block2(h)
#         if self.skep is not None:
#             return h + self.skep(x)
#         else:
#             return h + x
#
#
# class Downsample(nn.Module):
#     """
#     A downsampling layer with an optional convolution.
#     :param channels: channels in the inputs and outputs.
#     :param use_conv: a bool determining if a convolution is applied.
#     :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
#                  downsampling occurs in the inner-two dimensions.
#     """
#
#     def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
#         super().__init__()
#         self.channels = channels
#         self.out_channels = out_channels or channels
#         self.use_conv = use_conv
#         self.dims = dims
#         stride = 2 if dims != 3 else (1, 2, 2)
#         if use_conv:
#             self.op = conv_nd(
#                 dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
#             )
#         else:
#             assert self.channels == self.out_channels
#             self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)
#
#     def forward(self, x):
#         assert x.shape[1] == self.channels
#         return self.op(x)
#
#
# def conv_nd(dims, *args, **kwargs):
#     """
#     Create a 1D, 2D, or 3D convolution module.
#     """
#     if dims == 1:
#         return nn.Conv1d(*args, **kwargs)
#     elif dims == 2:
#         return nn.Conv2d(*args, **kwargs)
#     elif dims == 3:
#         return nn.Conv3d(*args, **kwargs)
#     raise ValueError(f"unsupported dimensions: {dims}")
#
#
# def avg_pool_nd(dims, *args, **kwargs):
#     """
#     Create a 1D, 2D, or 3D average pooling module.
#     """
#     if dims == 1:
#         return nn.AvgPool1d(*args, **kwargs)
#     elif dims == 2:
#         return nn.AvgPool2d(*args, **kwargs)
#     elif dims == 3:
#         return nn.AvgPool3d(*args, **kwargs)
#     raise ValueError(f"unsupported dimensions: {dims}")
#
#
# if __name__ == "__main__":
#     print('This is main of module "hello.py"')
#     adapter = Adapter(channels=[320, 640, 1280, 1280][:4], nums_rb=2, ksize=1, sk=True, use_conv=False)
#     x = torch.zeros((8, 1, 512, 512))
#     features = adapter(x)
#
#     print(__name__+'from hello.main')

import ipdb
import torch
import torch.nn as nn
import numpy as np
from torch.functional import F


class WithFussionTEAdapter(nn.Module):
    def __init__(self, channels=[320, 640, 1280, 1280], nums_rb=3, cin=64, ksize=3, sk=True, use_conv=True):
        super(WithFussionTEAdapter, self).__init__()
        self.unshuffle = nn.PixelUnshuffle(8)
        self.channels = channels
        self.nums_rb = nums_rb
        self.body = []
        for i in range(len(channels)):
            for j in range(nums_rb):
                if (i != 0) and (j == 0):
                    self.body.append(
                        ResnetBlock(channels[i - 1], channels[i], down=True, ksize=ksize, sk=sk, use_conv=use_conv))
                else:
                    self.body.append(
                        ResnetBlock(channels[i], channels[i], down=False, ksize=ksize, sk=sk, use_conv=use_conv))
        self.body = nn.ModuleList(self.body)
        self.conv_in = nn.Conv2d(cin, channels[0], 3, 1, 1)

        embed_dim = channels[0]

        self.conv_1 = nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=1, bias=False)
        self.conv_1.weight.data = torch.zeros_like(self.conv_1.weight.data)
        self.conv_2 = nn.Conv2d(in_channels=embed_dim * 2, out_channels=embed_dim * 2, kernel_size=1, bias=False)
        self.conv_2.weight.data = torch.zeros_like(self.conv_2.weight.data)
        self.conv_3 = nn.Conv2d(in_channels=embed_dim * 4, out_channels=embed_dim * 4, kernel_size=1, bias=False)
        self.conv_3.weight.data = torch.zeros_like(self.conv_3.weight.data)
        self.conv_4 = nn.Conv2d(in_channels=embed_dim * 4, out_channels=embed_dim * 4, kernel_size=1, bias=False)
        self.conv_4.weight.data = torch.zeros_like(self.conv_4.weight.data)

        self.zero_conv_list = [self.conv_1, self.conv_2, self.conv_3, self.conv_4]

        char_embed_dim = 512

        # self.embedding = nn.Embedding(100, char_embed_dim)
        self.embedding = nn.Embedding(100, char_embed_dim * 2)
        # self.text_pos_embed = nn.Parameter(torch.randn(1, 25, char_embed_dim) * .02)
        self.text_pos_embed = nn.Parameter(torch.randn(1, 25, char_embed_dim * 2) * .02)

        # self.encoder_layer = nn.TransformerEncoderLayer(char_embed_dim, nhead=8)
        # self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

        # self.up_embedding = nn.Linear(char_embed_dim, char_embed_dim * 2)

        # self.transformer_decoder_layer = nn.TransformerDecoderLayer(
        #     d_model=char_embed_dim * 2,
        #     nhead=8
        # )

        transformer_decoder_layer = nn.TransformerDecoderLayer(
            d_model=char_embed_dim * 2,
            nhead=8
        )

        self.transformer_decoder = nn.TransformerDecoder(
            transformer_decoder_layer,
            num_layers=4
        )

        # self.init_layers = [self.body, self.transformer_encoder, self.up_embedding]
        self.init_layers = [self.body]
        self.initialize_weights()

        self.zero_init_layers = [self.transformer_decoder]
        self.zero_initialize_weights()

    def initialize_weights(self):
        for module in self.init_layers:
            if isinstance(module, torch.nn.Conv2d):
                # 初始化卷积层的权重
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = np.sqrt(2.0 / fan_in)
                torch.nn.init.normal_(module.weight, 0, bound)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)
            elif isinstance(module, torch.nn.BatchNorm2d):
                # 初始化批归一化层的权重
                torch.nn.init.constant_(module.weight, 1)
                torch.nn.init.constant_(module.bias, 0)
            elif isinstance(module, torch.nn.Linear):
                # 初始化全连接层的权重
                fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = np.sqrt(6.0 / (fan_in + fan_out))
                torch.nn.init.uniform_(module.weight, -bound, bound)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)

    def zero_initialize_weights(self):
        for module in self.zero_init_layers:
            if isinstance(module, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(module.weight, 1)
                torch.nn.init.constant_(module.bias, 0)
            elif isinstance(module, torch.nn.Linear):
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x, text_embedding, encoder_hidden_states):
        # unshuffle
        x = self.unshuffle(x)
        # extract features
        features = []
        x = self.conv_in(x)
        for i in range(len(self.channels)):
            for j in range(self.nums_rb):
                idx = i * self.nums_rb + j
                x = self.body[idx](x)
            features.append(self.zero_conv_list[i](x))

        # ipdb.set_trace()
        text_embedding = self.embedding(text_embedding.long()).to(encoder_hidden_states.dtype)
        text_embedding = text_embedding + self.text_pos_embed

        text_embedding = text_embedding.permute(1, 0, 2)

        # # Apply Transformer Encoder
        # text_embedding = self.transformer_encoder(text_embedding)

        # seq_len, bs, dim = text_embedding.shape
        # text_embedding = text_embedding.reshape(bs * seq_len, dim)
        # text_embedding = self.up_embedding(text_embedding)
        # text_embedding = text_embedding.reshape(seq_len, bs, -1)

        ####################################################################
        text_embedding = self.transformer_decoder(
            tgt=encoder_hidden_states.permute(1, 0, 2),
            memory=text_embedding
        )

        encoder_hidden_states = encoder_hidden_states + text_embedding.permute(1, 0, 2)

        return features, encoder_hidden_states
        ####################################################################

        ####################################################################
        # second fussion
        # encoder_hidden_states = self.transformer_decoder(
        #     tgt=text_embedding,
        #     memory=encoder_hidden_states.permute(1, 0, 2)
        # )
        #
        # encoder_hidden_states = encoder_hidden_states.permute(1, 0, 2)
        #
        # return features, encoder_hidden_states
        ####################################################################


class ResnetBlock(nn.Module):
    def __init__(self, in_c, out_c, down, ksize=3, sk=False, use_conv=True):
        super().__init__()
        ps = ksize // 2
        if in_c != out_c or sk == False:
            self.in_conv = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            # print('n_in')
            self.in_conv = None
        self.block1 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.act = nn.ReLU()
        self.block2 = nn.Conv2d(out_c, out_c, ksize, 1, ps)
        if sk == False:
            self.skep = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            self.skep = None

        self.down = down
        if self.down == True:
            self.down_opt = Downsample(in_c, use_conv=use_conv)

    def forward(self, x):
        if self.down == True:
            x = self.down_opt(x)
        if self.in_conv is not None:  # edit
            x = self.in_conv(x)

        h = self.block1(x)
        h = self.act(h)
        h = self.block2(h)
        if self.skep is not None:
            return h + self.skep(x)
        else:
            return h + x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


if __name__ == "__main__":
    print('This is main of module "hello.py"')
    adapter = Adapter(channels=[320, 640, 1280, 1280][:4], nums_rb=2, ksize=1, sk=True, use_conv=False)
    x = torch.zeros((8, 1, 512, 512))
    features = adapter(x)

    print(__name__+'from hello.main')
