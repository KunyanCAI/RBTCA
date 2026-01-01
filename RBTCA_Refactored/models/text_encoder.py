import torch
import torch.nn as nn
from functools import partial
try:
    from .xtransformer import Encoder, TransformerWrapper
except ImportError:
    from xtransformer import Encoder, TransformerWrapper
import torch.optim as optim
import os

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError
class BERTTokenizer(AbstractEncoder):
    """ Uses a pretrained BERT tokenizer by huggingface."""
    def __init__(self, device="cuda", vq_interface=True, max_length=77):
        super().__init__()
        from transformers import BertTokenizerFast  # TODO: add to reuquirements
        bert_path = os.path.join(os.path.dirname(__file__), 'bert-base-uncased')
        self.tokenizer = BertTokenizerFast.from_pretrained(bert_path)
        self.device = device
        self.vq_interface = vq_interface
        self.max_length = max_length

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        return tokens

    @torch.no_grad()
    def encode(self, text):
        tokens = self(text)
        if not self.vq_interface:
            return tokens
        return None, None, [None, None, tokens]

    def decode(self, text):
        return text


class BERTEmbedder(AbstractEncoder):
    """Uses the BERT tokenizr model and add some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size=30522, max_seq_len=32,
                use_tokenizer=True, embedding_dropout=0.0,use_optimize=False,custom_tokens=None):      
        super().__init__()
        self.use_tknz_fn = use_tokenizer
        if self.use_tknz_fn:
            self.tknz_fn = BERTTokenizer(vq_interface=False, max_length=max_seq_len)
        if custom_tokens is not None:
            vocab_size += len(custom_tokens)
            self.add_custom_tokens(custom_tokens)
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer),
                                              emb_dropout=embedding_dropout)
        if use_optimize:
            self.optimizer = optim.Adam(self.transformer.parameters(), lr=1e-4)
        else:
            self.optimizer = None

    def forward(self, text,device="cuda"):
        if self.use_tknz_fn:
            tokens = self.tknz_fn(text).to(device)
        else:
            tokens = text
        z = self.transformer(tokens, return_embeddings=True)
        return z
    def add_custom_tokens(self, custom_tokens):
        for token in custom_tokens:
            self.tknz_fn.tokenizer.add_tokens(token)

    def encode(self, text,device="cuda"):
        # output of length 32
        return self(text,device)
    def optimize(self, loss):
        # Perform an optimization step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()