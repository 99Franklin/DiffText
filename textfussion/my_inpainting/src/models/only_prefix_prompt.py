import ipdb
import torch
import torch.nn as nn
import numpy as np
from torch.functional import F


# class OnlyPrefixPromptAdapter(nn.Module):
#     def __init__(self):
#         super(OnlyPrefixPromptAdapter, self).__init__()
#         char_embed_dim = 1024
#
#         ctx_vectors = torch.empty(16, char_embed_dim, dtype=torch.float16)
#         nn.init.normal_(ctx_vectors, std=0.02)
#         self.ctx = nn.Parameter(ctx_vectors)
#
#         self.text_pos_embed = nn.Parameter(torch.randn(1, 16, char_embed_dim) * .02)
#
#         encoder_layer = nn.TransformerEncoderLayer(char_embed_dim, nhead=8)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
#
#         self.init_layers = [self.transformer_encoder]
#         self.initialize_weights()
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
#     def forward(self, encoder_hidden_states):
#         ctx = self.ctx
#         # ipdb.set_trace()
#         if ctx.dim() == 2:
#             ctx = ctx.unsqueeze(0).expand(encoder_hidden_states.shape[0], -1, -1)
#
#         ctx = ctx + self.text_pos_embed
#
#         ctx = ctx.permute(1, 0, 2)
#         ctx = self.transformer_encoder(ctx)
#         ctx = ctx.permute(1, 0, 2)
#
#         encoder_hidden_states = torch.cat([ctx, encoder_hidden_states], dim=1)
#
#         return encoder_hidden_states


# class OnlyPrefixPromptAdapter(nn.Module):
#     def __init__(self):
#         super(OnlyPrefixPromptAdapter, self).__init__()
#         char_embed_dim = 1024
#
#         ctx_vectors = torch.empty(16, char_embed_dim, dtype=torch.float16)
#         nn.init.normal_(ctx_vectors, std=0.02)
#         self.ctx = nn.Parameter(ctx_vectors)
#
#         self.ctx_pos_embed = nn.Parameter(torch.randn(1, 16, char_embed_dim) * .02)
#
#         self.total_text_pos_embed = nn.Parameter(torch.randn(1, 77, char_embed_dim) * .02)
#
#         # encoder_layer = nn.TransformerEncoderLayer(char_embed_dim, nhead=8)
#         # self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
#         #
#         # self.init_layers = [self.transformer_encoder]
#         # self.initialize_weights()
#
#         self.embedding = nn.Embedding(100, char_embed_dim)
#         # self.text_pos_embed = nn.Parameter(torch.randn(1, 25, char_embed_dim) * .02)
#         self.char_embedding_pos_embed = nn.Parameter(torch.randn(1, 25, char_embed_dim) * .02)
#
#         transformer_decoder_layer = nn.TransformerDecoderLayer(
#             d_model=char_embed_dim,
#             nhead=8
#         )
#
#         self.transformer_decoder = nn.TransformerDecoder(
#             transformer_decoder_layer,
#             num_layers=6
#         )
#
#         self.init_layers = [self.transformer_decoder]
#         self.initialize_weights()
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
#     def forward(self, encoder_hidden_states, text_embedding):
#         ctx = self.ctx
#         # ipdb.set_trace()
#         if ctx.dim() == 2:
#             ctx = ctx.unsqueeze(0).expand(encoder_hidden_states.shape[0], -1, -1)
#
#         ctx = ctx + self.ctx_pos_embed
#
#         text_embedding = self.embedding(text_embedding.long()).to(encoder_hidden_states.dtype)
#         text_embedding = text_embedding + self.char_embedding_pos_embed
#
#         text_embedding = text_embedding.permute(1, 0, 2)
#
#         ctx = ctx.permute(1, 0, 2)
#         # ctx = self.transformer_encoder(ctx)
#         ctx = self.transformer_decoder(
#             tgt=ctx,
#             memory=text_embedding
#         )
#         ctx = ctx.permute(1, 0, 2)
#
#         encoder_hidden_states = torch.cat([ctx, encoder_hidden_states], dim=1)
#         encoder_hidden_states = encoder_hidden_states + self.total_text_pos_embed
#
#         return encoder_hidden_states


class OnlyPrefixPromptAdapter(nn.Module):
    def __init__(self):
        super(OnlyPrefixPromptAdapter, self).__init__()
        char_embed_dim = 1024

        ctx_vectors = torch.empty(25, char_embed_dim, dtype=torch.float16)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)

        self.ctx_pos_embed = nn.Parameter(torch.randn(1, 25, char_embed_dim) * .02)

        self.embedding = nn.Embedding(100, char_embed_dim)
        self.char_embedding_pos_embed = nn.Parameter(torch.randn(1, 25, char_embed_dim) * .02)

        transformer_decoder_layer = nn.TransformerDecoderLayer(
            d_model=char_embed_dim,
            nhead=8
        )

        self.te_transformer_decoder = nn.TransformerDecoder(
            transformer_decoder_layer,
            num_layers=6
        )

        self.clip_transformer_decoder = nn.TransformerDecoder(
            transformer_decoder_layer,
            num_layers=6
        )

        self.init_layers = [self.te_transformer_decoder, self.clip_transformer_decoder]
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

    def zero_initialize_weights(self):
        for module in self.zero_init_layers:
            if isinstance(module, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(module.weight, 1)
                torch.nn.init.constant_(module.bias, 0)
            elif isinstance(module, torch.nn.Linear):
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, encoder_hidden_states, text_embedding):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(encoder_hidden_states.shape[0], -1, -1)

        ctx = ctx + self.ctx_pos_embed

        text_embedding = self.embedding(text_embedding.long()).to(encoder_hidden_states.dtype)
        text_embedding = text_embedding + self.char_embedding_pos_embed

        ctx = ctx.permute(1, 0, 2)
        # ctx = self.transformer_encoder(ctx)
        ctx = self.te_transformer_decoder(
            tgt=ctx,
            memory=text_embedding.permute(1, 0, 2)
        )

        ctx = self.clip_transformer_decoder(
            tgt=ctx,
            memory=encoder_hidden_states.permute(1, 0, 2)
        )

        ctx = ctx.permute(1, 0, 2)

        return ctx
