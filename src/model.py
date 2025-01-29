import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class CSEBert(nn.Module):
    def __init__(self, path_to_model_weight, gradient_checkpointing=True):
        super().__init__()
        self.path_to_model_weight = path_to_model_weight

        self.transformer = AutoModel.from_pretrained(
            path_to_model_weight, use_cache=False
        )

        if gradient_checkpointing:
            self.transformer.gradient_checkpointing_enable()

    def forward(self, input_ids, attention_mask):
        y = self.transformer(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True
        )

        token_embeddings = y[0]

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()
        ).to(token_embeddings.dtype)
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_tokenizer(self, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(
            self.path_to_model_weight, **kwargs
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token
        return tokenizer
