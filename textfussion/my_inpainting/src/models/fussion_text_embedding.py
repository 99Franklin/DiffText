import ipdb
import numpy as np
import torch
import torch.nn as nn
from torch.functional import F


class FussionTextEmbedding(nn.Module):
    def __init__(self):
        super(FussionTextEmbedding, self).__init__()

        embed_dim = 512
        self.embedding = nn.Embedding(100, embed_dim)
        self.text_pos_embed = nn.Parameter(torch.randn(1, 25, embed_dim) * .02)

        self.encoder_layer = nn.TransformerEncoderLayer(embed_dim, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

        self.up_embedding = nn.Linear(embed_dim, embed_dim*2)

        self.transformer_decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim * 2,
            nhead=8
        )
        self.transformer_decoder = nn.TransformerDecoder(
            self.transformer_decoder_layer,
            num_layers=1
        )

        self.init_layers = [self.transformer_encoder, self.up_embedding]
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

    def forward(self, encoder_hidden_states, text_embedding):
        text_embedding = self.embedding(text_embedding.long()).to(encoder_hidden_states.dtype)
        text_embedding = text_embedding + self.text_pos_embed

        text_embedding = text_embedding.permute(1, 0, 2)

        # Apply Transformer Encoder
        text_embedding = self.transformer_encoder(text_embedding)

        seq_len, bs, dim = text_embedding.shape
        text_embedding = text_embedding.reshape(bs * seq_len, dim)
        text_embedding = self.up_embedding(text_embedding)
        text_embedding = text_embedding.reshape(seq_len, bs, -1)

        text_embedding = self.transformer_decoder(
            tgt=encoder_hidden_states.permute(1, 0, 2),
            memory=text_embedding
        )

        encoder_hidden_states = encoder_hidden_states + text_embedding.permute(1, 0, 2)

        return encoder_hidden_states


if __name__ == "__main__":
    print('This is main of module "hello.py"')
    adapter = Adapter(channels=[320, 640, 1280, 1280][:4], nums_rb=2, ksize=1, sk=True, use_conv=False)
    x = torch.zeros((8, 1, 512, 512))
    features = adapter(x)

    print(__name__+'from hello.main')
