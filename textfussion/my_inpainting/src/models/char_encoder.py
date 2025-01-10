import ipdb
import torch
import torch.nn as nn


class CharEncoder(nn.Module):
    def __init__(self):
        super(CharEncoder, self).__init__()

        char_embed_dim = 1024

        self.embedding = nn.Embedding(100, char_embed_dim)
        # self.text_pos_embed = nn.Parameter(torch.randn(1, 25, char_embed_dim) * .02)
        self.text_pos_embed = nn.Parameter(torch.randn(1, 25, char_embed_dim) * .02)

        encoder_layer = nn.TransformerEncoderLayer(char_embed_dim, nhead=8, batch_first=False)
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
        ipdb.set_trace()
        text_embedding = self.embedding(text_tokens.long()).to(dtype)
        text_embedding = text_embedding + self.text_pos_embed

        # text_embedding = text_embedding.permute(1, 0, 2)
        text_embedding = self.transformer_encoder(text_embedding)

        # text_embedding = text_embedding.permute(1, 0, 2)

        return text_embedding
