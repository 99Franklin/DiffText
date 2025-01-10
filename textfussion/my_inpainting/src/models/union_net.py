import ipdb
import math
import torch
import torch.nn as nn


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=100):
        super(PositionalEncoder, self).__init__()
        self.d_model = d_model

        # 创建位置编码矩阵
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # 输入x的shape为(batch_size, seq_len, d_model)
        seq_len = x.size(1)

        # 将位置编码添加到输入的字符序列中
        x = x + self.pe[:, :seq_len]
        return x


class CharEncoder(nn.Module):
    def __init__(self):
        super(CharEncoder, self).__init__()

        char_embed_dim = 1024

        self.embedding = nn.Embedding(100, char_embed_dim)
        # self.text_pos_embed = nn.Parameter(torch.randn(1, 25, char_embed_dim) * .02)
        self.text_pos_embed = PositionalEncoder(char_embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(char_embed_dim, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

        # self.up_embedding = nn.Linear(char_embed_dim, char_embed_dim * 2)

        self.init_layers = [self.transformer_encoder]
        self.initialize_weights()

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

    def forward(self, text_tokens, dtype):
        text_embedding = self.embedding(text_tokens.long()).to(dtype)
        text_embedding = self.text_pos_embed(text_embedding)
        # text_embedding = text_embedding.permute(1, 0, 2)
        text_embedding = self.transformer_encoder(text_embedding)

        text_embedding = self.text_pos_embed(text_embedding)
        # text_embedding = text_embedding.permute(1, 0, 2)

        return text_embedding


class UnionNet(nn.Module):
    def __init__(self, unet):
        super(UnionNet, self).__init__()
        self.unet = unet
        self.char_encoder = CharEncoder()

        # for name, param in self.unet.named_parameters():
        #     if 'down_blocks' in name:
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False
        #
        # for name, param in self.controlnet.named_parameters():
        #     param.requires_grad = True

    def forward(self, noise_sample, timesteps, encoder_hidden_states, text_token=None):
        # ipdb.set_trace()
        text_embedding = self.char_encoder(text_token, noise_sample.dtype)
        out = self.unet(noise_sample, timesteps, encoder_hidden_states, char_embedding=text_embedding).sample

        return out

