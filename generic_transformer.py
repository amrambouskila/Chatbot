import re
import time
import nltk
import math
import torch
import pynvml
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn as nn
from pathlib import Path
from typing import Callable
import torch.optim as optim
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from datasets import load_dataset
from tokenizers.models import WordLevel
from torch.cuda.amp import GradScaler, autocast
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import Trainer, TrainingArguments
from torch.utils.data.dataloader import default_collate
from tokenizers import Tokenizer, pre_tokenizers, models, trainers
from torch.utils.data import Dataset, DataLoader, random_split, dataset
from torch.optim.lr_scheduler import StepLR, CyclicLR, CosineAnnealingLR
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup

matplotlib.use('Qt5Agg', force=True)
sns.set()


def relu(x: torch.Tensor):
    return torch.max(x, torch.tensor(0.0))


def softmax(x: torch.Tensor, dim: int = -1):
    max_val, _ = torch.max(x, dim=dim, keepdim=True)
    x_exp = torch.exp(x - max_val)
    sum_x_exp = torch.sum(x_exp, dim=dim, keepdim=True)
    return x_exp / sum_x_exp


class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -math.log(10000) / d_model)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * ((x - mean) / torch.sqrt(std + self.eps)) + self.bias


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, 'd_model is not divisible by h'
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self._attention_scores = None

    @property
    def attention_scores(self):
        return self._attention_scores

    @attention_scores.setter
    def attention_scores(self, attention_scores: torch.Tensor):
        self._attention_scores = attention_scores

    def attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor):
        d_k = query.shape[-1]
        attention_scores = ((query @ key.transpose(-2, -1)) / (math.sqrt(d_k)))

        if mask is not None:
            mask = mask.to(torch.bool)  # Ensure mask is boolean
            mask = mask.expand_as(attention_scores)  # Expand mask dimensions
            attention_scores = attention_scores.float()  # Ensure attention scores are float32
            attention_scores.masked_fill_(mask == 0, -1e9)

        attention_scores = torch.softmax(attention_scores, dim=-1)

        if self.dropout is not None:
            attention_scores = self.dropout(attention_scores)

        return attention_scores @ value, attention_scores

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
        x, self.attention_scores = self.attention(query, key, value, mask)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        return self.w_o(x)


class ResidualConnection(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x: torch.Tensor, sublayer: Callable):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor):
        x = self.residual_connections[0](x, lambda x_i: self.self_attention_block(x_i, x_i, x_i, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor):
        x = self.residual_connections[0](x, lambda x_i: self.self_attention_block(x_i, x_i, x_i, tgt_mask))
        x = self.residual_connections[1](x, lambda x_j: self.cross_attention_block(x_j, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        return self.norm(x)


class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor):
        epsilon = 1e-8
        return torch.log(torch.softmax(self.proj(x), dim=-1) + epsilon)


class Transformer(nn.Module):
    def __init__(
            self,
            encoder: Encoder,
            decoder: Decoder,
            src_embed: InputEmbeddings,
            tgt_embed: InputEmbeddings,
            src_pos: PositionalEncoding,
            tgt_pos: PositionalEncoding,
            projection_layer: ProjectionLayer
    ):

        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: torch.Tensor = None, tgt_mask: torch.Tensor = None):
        # Encode the source sequence
        encoder_output = self.encode(src, src_mask)

        # Decode the target sequence
        decoder_output = self.decode(encoder_output, src_mask, tgt, tgt_mask)

        # Project the decoder output to the vocabulary space
        output = self.projection_layer(decoder_output)

        return output


class ChatbotDataset(Dataset):
    special_tokens = []

    def __init__(self, dataset: dataset.Subset, tokenizer: Tokenizer, seq_len: int):
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.sos_token = torch.tensor([tokenizer.token_to_id('[SOS]')], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer.token_to_id('[PAD]')], dtype=torch.int64)
        self.unk_token = torch.tensor([tokenizer.token_to_id('[UNK]')], dtype=torch.int64)
        self.__class__.special_tokens = list({*self.__class__.special_tokens, *[self.sos_token, self.eos_token, self.pad_token, self.unk_token]})

    def __len__(self):
        return len(self.dataset.dataset.keys())

    def __getitem__(self, index: int):
        context = self.dataset.dataset[index]['context']
        response = self.dataset.dataset[index]['response']
        context_tokens = self.tokenizer.encode(context).ids
        response_tokens = self.tokenizer.encode(response).ids

        num_enc_padding_token = self.seq_len - (len(context_tokens) + 2)
        num_dec_padding_token = self.seq_len - (len(response_tokens) + 2)

        if num_enc_padding_token < 0 or num_dec_padding_token < 0:
            raise ValueError('Sentence is too long')

        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(context_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * num_enc_padding_token, dtype=torch.int64)
            ]
        )

        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(response_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * num_dec_padding_token, dtype=torch.int64)
            ]
        )

        label = torch.cat(
            [
                torch.tensor(response_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * (num_dec_padding_token + 1), dtype=torch.int64),
            ]
        )

        encoder_input_size = encoder_input.size(0)
        decoder_input_size = decoder_input.size(0)
        label_size = label.size(0)
        assert encoder_input_size == self.seq_len, f'sequence length must be {self.seq_len}, encoder input length is {encoder_input_size}'
        assert decoder_input_size == self.seq_len, f'sequence length must be {self.seq_len}, decoder input length is {decoder_input_size}'
        assert label_size == self.seq_len, f'sequence length must be {self.seq_len}, label length is {label_size}'

        # Encoder mask: [batch_size, 1, seq_len]
        encoder_mask = (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(1).type(torch.bool)

        # Decoder mask: [1, seq_len, seq_len]
        seq_len = self.seq_len
        subsequent_mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8), diagonal=1).to(torch.bool)
        decoder_mask = (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(1) & ~subsequent_mask

        return {
            'encoder_input': encoder_input,
            'decoder_input': decoder_input,
            'encoder_mask': encoder_mask,
            'decoder_mask': decoder_mask,
            'label': label,
            'context': context,
            'response': response
        }


def remove_stutter(text: str) -> str:
    stutter_pattern = re.compile(r'\b(\w)-\1(\w+)\b', re.IGNORECASE)
    corrected_text = stutter_pattern.sub(r'\1\2', text)
    return corrected_text


def preprocess_text_data(dataset: list) -> dict:
    text = [remove_stutter(' '.join(t.replace('\\N', ' ').split())).lower() for t in dataset]
    text = [re.sub(r'[^A-Za-z\s]', '', t) for t in text]
    text = [t for t in text if t != '']

    text_dict = {}
    max_input_length = -1
    for i, t in enumerate(text[:-1]):
        n_words = len(t.split(' '))
        if n_words > max_input_length:
            max_input_length = n_words

        text_dict[i] = {'context': t, 'response': text[i + 1]}

    return text_dict


def get_or_build_tokenizer(config: dict, dataset: dict) -> Tokenizer:
    tokenizer_path = Path(config['tokenizer_path'])
    if not tokenizer_path.exists():
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))

        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=['[UNK]', '[PAD]', '[SOS]', '[EOS]'],
            min_frequency=2
        )

        lexicon = [dataset[i]['context'] for i in dataset.keys()] + [dataset[len(dataset) - 1]['response']]
        tokenizer.train_from_iterator(lexicon, trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer


def split_dataset(dataset: dict, tokenizer: Tokenizer, dataset_class: type, config: dict):

    # Split the dataset into training and validation sets
    train_dataset_size = int(0.9 * len(dataset))
    val_dataset_size = len(dataset) - train_dataset_size
    train_dataset_raw, val_dataset_raw = random_split(dataset, (train_dataset_size, val_dataset_size))

    # Create the dataset
    train_dataset = dataset_class(train_dataset_raw, tokenizer, config['seq_len'])
    val_dataset = dataset_class(val_dataset_raw, tokenizer, config['seq_len'])

    return train_dataset, val_dataset


def custom_collate_fn(batch: list):
    filtered_batch = []
    for idx, item in enumerate(batch):
        try:
            # Ensure item has all required keys
            if not all(key in item for key in ['encoder_input', 'decoder_input', 'encoder_mask', 'decoder_mask', 'context', 'response', 'label']):
                print(f"Missing keys at index {idx} - {item}")
                continue

            encoder_input = item['encoder_input']
            decoder_input = item['decoder_input']

            if (encoder_input is not None and decoder_input is not None and
                    not torch.isnan(encoder_input).any() and not torch.isnan(decoder_input).any() and
                    not torch.isinf(encoder_input).any() and not torch.isinf(decoder_input).any()):

                filtered_batch.append(item)
            else:
                print(f"Invalid data (None, NaN, Inf) found at index {idx} - encoder_input: {encoder_input}, decoder_input: {decoder_input}")
        except BaseException as e:
            print(f"Error at index {idx} - {item}: {e}")
            continue

    # Use default_collate to collate the filtered batch
    if len(filtered_batch) == 0:
        return {
            'encoder_input': torch.tensor([]),
            'decoder_input': torch.tensor([]),
            'encoder_mask': torch.tensor([]),
            'decoder_mask': torch.tensor([]),
            'label': torch.tensor([]),
            'context': [],
            'response': [],
        }

    else:
        batch = default_collate(filtered_batch)

        # Ensure no NaN or Inf values in the batch
        for key in ['encoder_input', 'decoder_input']:
            batch[key] = torch.where(
                torch.isfinite(batch[key]), batch[key], torch.zeros_like(batch[key])
            )

        return batch


def build_transformer(vocab_size: int, seq_len: int, d_model: int, N: int, h: int, dropout: float, d_ff: int) -> Transformer:
    src_embed = InputEmbeddings(d_model, vocab_size)
    tgt_embed = InputEmbeddings(d_model, vocab_size)
    src_pos = PositionalEncoding(d_model, seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, seq_len, dropout)

    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block,
                                     dropout)
        decoder_blocks.append(decoder_block)

    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    projection_layer = ProjectionLayer(d_model, vocab_size)
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer


def find_embedding_layer_names(model: Transformer):
    embedding_layer_names = []

    def _find_embeddings(model, prefix: str = ''):
        for name, child in model.named_children():
            # Check if the child is an embedding layer
            if isinstance(child, nn.Embedding):
                # If prefix exists, append the current name to the prefix
                # Otherwise, use the current name as is
                full_name = f'{prefix}.{name}' if prefix else name
                embedding_layer_names.append(full_name)
            else:
                # Recursively call _find_embeddings on the child module
                # Append the current name to the prefix
                new_prefix = f'{prefix}.{name}' if prefix else name
                _find_embeddings(child, new_prefix)

    # Start the recursive search from the top-level model
    _find_embeddings(model)
    return embedding_layer_names


def zero_special_token_grads(model: Transformer, special_token_indices: list, embedding_layer_names: list):
    for name, param in model.named_parameters():
        if any(embedding_layer_name in name for embedding_layer_name in embedding_layer_names) and param.grad is not None:
            for idx in special_token_indices:
                if idx < param.grad.size(0):
                    param.grad[idx].zero_()


def print_gpu_utilization():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used // 1024 ** 2:.2f} MB.")


def autoregressive_decode(model: Transformer, input_tokens: torch.Tensor, seq_len: int, tokenizer: Tokenizer, device: torch.device):
    model.eval()
    sos_token = torch.tensor([tokenizer.token_to_id('[SOS]')], dtype=torch.long, device=device).unsqueeze(0)
    decoded_tokens = sos_token

    for _ in range(seq_len):
        with torch.no_grad():
            output = model.forward(input_ids=input_tokens, decoder_input_ids=decoded_tokens)
            next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            decoded_tokens = torch.cat((decoded_tokens, next_token), dim=1)
            if next_token.item() == tokenizer.token_to_id('[EOS]'):
                break

    return decoded_tokens


def beam_search_decode(model: Transformer, input_tokens: torch.Tensor, seq_len: int, tokenizer: Tokenizer, device: torch.device, num_beams: int = 3):
    model.eval()
    # Initialize the decoding with the start-of-sequence (SOS) token
    sos_token = torch.tensor([tokenizer.token_to_id('[SOS]')], dtype=torch.long, device=device).unsqueeze(0)
    # Beam initialization: each beam is a tuple of (decoded tokens, log-probability)
    beams = [(sos_token, 0.0)]

    for _ in range(seq_len):
        new_beams = []
        for decoded_tokens, log_prob in beams:
            with torch.no_grad():
                # Forward pass through the model
                output = model.forward(input_tokens, decoded_tokens)
                # Get the log probabilities of the next tokens
                next_token_log_probs = torch.log_softmax(output.logits[:, -1, :], dim=-1)
                # Get the top `num_beams` tokens and their log probabilities
                top_next_tokens = next_token_log_probs.topk(num_beams, dim=-1)
                for i in range(num_beams):
                    next_token = top_next_tokens.indices[:, i].unsqueeze(0)
                    next_log_prob = top_next_tokens.values[:, i].item()
                    # Append the predicted token to the decoded sequence
                    new_decoded_tokens = torch.cat((decoded_tokens, next_token), dim=1)
                    new_log_prob = log_prob + next_log_prob
                    new_beams.append((new_decoded_tokens, new_log_prob))
        # Keep only the top `num_beams` sequences
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:num_beams]

        # Check if any beam has generated the EOS token
        if any(tokenizer.token_to_id('[EOS]') in beam[0].squeeze().tolist() for beam in beams):
            break

    # Return the best sequence
    best_sequence = beams[0][0]
    return best_sequence


def nucleus_sampling_decode(model: Transformer, input_tokens: torch.Tensor, seq_len: int, tokenizer: Tokenizer, device: torch.device, top_p: float = 0.9):
    model.eval()
    # Initialize the decoding with the start-of-sequence (SOS) token
    sos_token = torch.tensor([tokenizer.token_to_id('[SOS]')], dtype=torch.long, device=device).unsqueeze(0)
    decoded_tokens = sos_token

    for _ in range(seq_len):
        with torch.no_grad():
            # Forward pass through the model
            output = model(input_ids=input_tokens, decoder_input_ids=decoded_tokens)
            # Get the probabilities of the next tokens
            next_token_probs = torch.softmax(output.logits[:, -1, :], dim=-1)
            # Sort the probabilities to get cumulative probabilities
            sorted_probs, sorted_indices = torch.sort(next_token_probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            # Find the smallest set of tokens with cumulative probability > top_p
            nucleus_mask = cumulative_probs <= top_p
            nucleus_mask[..., 1:] = nucleus_mask[..., :-1].clone()
            nucleus_mask[..., 0] = True
            # Filter out tokens outside the nucleus
            next_token_probs = next_token_probs * nucleus_mask
            next_token_probs /= next_token_probs.sum(dim=-1, keepdim=True)
            # Sample the next token from the nucleus
            next_token = torch.multinomial(next_token_probs, num_samples=1)
            # Append the sampled token to the decoded sequence
            decoded_tokens = torch.cat((decoded_tokens, next_token), dim=1)
            # Stop decoding if the end-of-sequence token is produced
            if next_token.item() == tokenizer.token_to_id('[EOS]'):
                break

    return decoded_tokens


def train_chatbot(raw_dataset: list, config: dict):
    model_path = Path(config['model_path'])
    if not model_path.exists():
        train_info = {}
        dataset = preprocess_text_data(raw_dataset)
        train_info['dataset'] = dataset
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'The neural network will be running on {device}')
        tokenizer = get_or_build_tokenizer(config=config, dataset=dataset)
        train_dataset, val_dataset = split_dataset(dataset=dataset, tokenizer=tokenizer, dataset_class=ChatbotDataset, config=config)
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=custom_collate_fn, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=custom_collate_fn, pin_memory=True)
        vocab_size = len(tokenizer.get_vocab())
        total_batches = len(train_loader)

        model = build_transformer(
            vocab_size=vocab_size,
            seq_len=config['seq_len'],
            d_model=config['d_model'],
            N=config['N'],
            h=config['h'],
            dropout=config['dropout'],
            d_ff=config['d_ff']
        )

        embedding_layer_names = find_embedding_layer_names(model)
        model.to(device)
        criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.pad_token)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

        # scheduler = CosineAnnealingLR(optimizer, T_max=total_batches * config['num_epochs'])
        scheduler = CyclicLR(
            optimizer,
            base_lr=config['learning_rate'],
            max_lr=1e-3,
            step_size_up=int(total_batches / 2),
            mode='triangular2'
        )

        scaler = GradScaler()
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        train_info['train_losses'] = []
        train_info['val_losses'] = []
        epoch_iterator = tqdm(range(config['num_epochs']), desc=f'Epoch counter')
        for epoch in epoch_iterator:
            epoch_time = time.time()
            model.train()
            train_loss = 0
            last_printed_progress = 0
            for idx, batch in enumerate(train_loader):
                train_batch_start_time = time.time()
                progress = (idx + 1) / total_batches * 100

                if int(progress) // 10 > last_printed_progress:
                    last_printed_progress = int(progress) // 10
                    print(f'Epoch {epoch + 1}/{config["num_epochs"]} Training Loop is {progress:.2f}% completed')

                encoder_input = batch['encoder_input'].to(device, non_blocking=True)
                decoder_input = batch['decoder_input'].to(device, non_blocking=True)
                encoder_mask = batch['encoder_mask'].to(device, non_blocking=True)
                decoder_mask = batch['decoder_mask'].to(device, non_blocking=True)
                labels = batch['label'].to(device, non_blocking=True)

                # print(f"Encoder input stats - Min: {encoder_input.float().min()}, Max: {encoder_input.float().max()}, Mean: {encoder_input.float().mean()}")
                # print(f"Decoder input stats - Min: {decoder_input.float().min()}, Max: {decoder_input.float().max()}, Mean: {decoder_input.float().mean()}")

                optimizer.zero_grad()

                with autocast():
                    output = model.forward(encoder_input, decoder_input, encoder_mask, decoder_mask)
                    loss = criterion(output.view(-1, vocab_size), labels.view(-1))

                    # Mask the special tokens in the loss calculation
                    # mask = torch.ones_like(labels, dtype=torch.bool)
                    # for token in [train_dataset.sos_token, train_dataset.eos_token, train_dataset.pad_token, train_dataset.unk_token]:
                    #     mask &= (labels != token.to(device))

                    # loss = (criterion(output.view(-1, vocab_size), labels.view(-1)) * mask.view(-1).float()).sum()

                if torch.any(torch.isnan(loss)) or torch.any(torch.isinf(loss)):
                    print(f"NaN or Inf detected in loss at batch {idx + 1} - {batch['context']}, {batch['response']}")
                    continue

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                # zero_special_token_grads(model=model, special_token_indices=ChatbotDataset.special_tokens, embedding_layer_names=embedding_layer_names)
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()
                scheduler.step()

                train_batch_end_time = time.time()
                if train_batch_end_time - train_batch_start_time > 3:
                    print(f'batch {idx + 1} took {train_batch_end_time - train_batch_start_time} seconds')


            train_loss /= len(train_loader)
            train_info['train_losses'].append(train_loss)
            print(f'Epoch {epoch + 1}/{config["num_epochs"]}, Loss: {train_loss}, Learning rate: {optimizer.param_groups[0]["lr"]}')
            print_gpu_utilization()

            # epoch_iterator.set_postfix({f'train_loss': f'{train_loss:6.3f}'})

            # model.eval()
            # val_loss = 0.0
            # with torch.no_grad():
            #     for idx, batch in enumerate(val_loader):
            #         val_batch_start_time = time.time()
            #         progress = (idx + 1) / total_batches * 100
            #
            #         if int(progress) // 10 > last_printed_progress:
            #             last_printed_progress = int(progress) // 10
            #             print(f'Epoch {epoch + 1}/{config["num_epochs"]} Training Loop is {progress:.2f}% completed')
            #
            #         encoder_input = batch['encoder_input'].to(device, non_blocking=True)
            #         encoder_mask = batch['encoder_mask'].to(device, non_blocking=True)
            #
            #         # Autoregressive decoding
            #         output_tokens = autoregressive_decode(model, encoder_input, config['seq_len'], tokenizer, device)
            #
            #         # Calculate loss
            #         output = model.forward(encoder_input, output_tokens[:, :-1], encoder_mask, None)
            #         loss = criterion(output.view(-1, vocab_size), batch['label'].view(-1).to(device))
            #
            #         if torch.isnan(loss) or torch.isinf(loss):
            #             print(f"NaN or Inf detected in loss at batch {idx + 1} - {batch['context']}, {batch['response']}")
            #             continue
            #
            #         val_loss += loss.item()
            #         val_batch_end_time = time.time()
            #         if val_batch_end_time - val_batch_start_time > 1:
            #             print(f'batch {idx + 1} took {val_batch_end_time - val_batch_start_time} seconds')
            #
            # val_loss /= len(val_loader)
            # val_losses.append(val_loss)
            # print(f'Epoch {epoch + 1}/{config["num_epochs"]}, Validation Loss: {val_loss}')

            # if val_loss < best_val_loss:
            #     best_val_loss = val_loss
            #     patience_counter = 0
            # else:
            #     patience_counter += 1
            #
            # if patience_counter >= patience:
            #     print("Early stopping triggered.")
            #     break

            print(f'Epoch {epoch + 1} took {time.time() - epoch_time:.2f} seconds')

        torch.save(model.state_dict(), config['model_path'])
        return train_info


def run_analytics(train_info: dict, config: dict):
    stop_words = set(stopwords.words('english'))
    n_strings = len(train_info['dataset'].keys())
    dataset = [train_info['dataset'][i]['context'] for i in range(n_strings)] + [train_info['dataset'][n_strings - 1]['response']]
    text = [word for sentence in dataset for word in sentence if word not in stop_words]

    # Wordcloud
    tokens = nltk.word_tokenize(' '.join(text))
    processed_text = ' '.join(tokens)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(processed_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

    # Word Distribution
    sequence_lengths = [len(sequence) for sequence in text]
    fig, ax = plt.subplots()
    ax.hist(sequence_lengths, bins=50)
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Frequency')
    plt.show()

    # Loss vs Training Time
    train_losses = train_info['train_losses']
    fig, ax = plt.subplots()
    ax.plot(list(range(1, config['num_epochs'] + 1)), train_losses, label='Training Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    plt.legend()
    plt.show()


def load_chatbot(device: torch.device, config: dict, vocab_size: int):
    model = build_transformer(
        vocab_size=vocab_size,
        seq_len=config['seq_len'],
        d_model=config['d_model'],
        N=config['N'],
        h=config['h'],
        dropout=config['dropout'],
        d_ff=config['d_ff']
    )

    model.load_state_dict(torch.load(config['model_path']))
    model.to(device)
    model.eval()
    return model


def preprocess_sentence(sentence: str, tokenizer: Tokenizer, seq_len: int, device: torch.device):
    tokens = tokenizer.encode(sentence).ids
    tokens = [tokenizer.token_to_id('[SOS]')] + tokens + [tokenizer.token_to_id('[EOS]')]
    tokens = tokens[:seq_len] + [tokenizer.token_to_id('[PAD]')] * (seq_len - len(tokens))
    return torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)


def chatbot_predict(model: Transformer, sentence: str, tokenizer: Tokenizer, seq_len: int, device: torch.device, output_length: int = 32):
    model.eval()
    input_tokens = preprocess_sentence(sentence, tokenizer, seq_len, device)
    special_tokens = [tokenizer.token_to_id('[SOS]'), tokenizer.token_to_id('[EOS]'), tokenizer.token_to_id('[PAD]'), tokenizer.token_to_id('[UNK]')]

    # Start decoding with the start-of-sequence token
    sos_token = torch.tensor([tokenizer.token_to_id('[SOS]')], dtype=torch.long, device=device).unsqueeze(0)
    decoded_tokens = sos_token

    while len(decoded_tokens[0]) < output_length:
        with torch.no_grad():

            output_tokens = model.forward(input_tokens, decoded_tokens)
            next_token_logits = output_tokens[:, -1, :]

            # Sample from the top k tokens
            top_k = 10
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
            probabilities = torch.nn.functional.softmax(top_k_logits, dim=-1)
            next_token_idx = torch.multinomial(probabilities, 1).item()

            # Get the actual token index
            next_token = top_k_indices[:, next_token_idx].unsqueeze(0)

            # Ensure next_token is properly shaped before concatenation
            next_token = next_token.view(1, -1)

            while next_token.item() in special_tokens:
                next_token_idx = torch.multinomial(probabilities, 1).item()
                next_token = top_k_indices[:, next_token_idx].unsqueeze(0)
                next_token = next_token.view(1, -1)

            if next_token.item() >= 4:
                decoded_tokens = torch.cat((decoded_tokens, next_token), dim=1)

            if next_token.item() == tokenizer.token_to_id('[EOS]'):
                if len(decoded_tokens[0]) >= output_length:
                    break

    output_ids = decoded_tokens.squeeze().tolist()

    if isinstance(output_ids, int):
        output_ids = [output_ids]

    output_sentence = tokenizer.decode(output_ids, skip_special_tokens=True)
    return output_sentence


def chat(config: dict):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = Tokenizer.from_file(str(config['tokenizer_path']))
    vocab_size = len(tokenizer.get_vocab())
    chatbot = load_chatbot(device=device, config=config, vocab_size=vocab_size)
    name = 'Goku'
    print('Type `quit` to end chatting')
    while True:
        sentence = input('You: ')
        if 'quit' == sentence:
            break

        output = chatbot_predict(chatbot, sentence, tokenizer, config['seq_len'], device)
        print(f'{name}: {output}')


def main():
    load_data_time = time.time()
    raw_dataset = load_dataset("Fishball02/anime-subtitle-dragon-ball")['train']['text']
    config = {
        'model_path': './LLM.pt',
        'tokenizer_path': './tokenizer.json',
        'dropout': 0.1,
        'seq_len': 32,
        'batch_size': 128,
        'd_model': 512,
        'h': 8,
        'N': 6,
        'd_ff': 2048,
        'num_epochs': 100,
        'learning_rate': 1e-5
    }
    print(f'Loading data took {time.time() - load_data_time:.2f} seconds')
    train_info = train_chatbot(raw_dataset=raw_dataset, config=config)

    if train_info is not None:
        run_analytics(train_info=train_info, config=config)

    chat(config=config)


if __name__ == '__main__':
    main()
