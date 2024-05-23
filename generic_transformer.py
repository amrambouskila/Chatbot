import re
import nltk
import time
import math
import torch
import pynvml
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn as nn
from pathlib import Path
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
from torch.optim.lr_scheduler import StepLR, CyclicLR
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset, DataLoader, random_split
from tokenizers import Tokenizer, pre_tokenizers, models, trainers
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup

matplotlib.use('Qt5Agg', force=True)
sns.set()


def relu(x):
    return torch.max(x, torch.tensor(0.0))


def softmax(x, dim: int = -1):
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

    def forward(self, x):
        # print(f"InputEmbeddings stats - Min: {x.float().min()}, Max: {x.float().max()}, Mean: {x.float().mean()}")
        if x is None:
            raise ValueError("Input to embedding is None")

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

    def forward(self, x):
        # print(f"PositionalEncoding stats - Min: {x.float().min()}, Max: {x.float().max()}, Mean: {x.float().mean()}")
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # print(f"LayerNormalization stats - Min: {x.float().min()}, Max: {x.float().max()}, Mean: {x.float().mean()}")
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * ((x - mean) / torch.sqrt(std + self.eps)) + self.bias


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # print(f"FeedForwardBlock stats - Min: {x.float().min()}, Max: {x.float().max()}, Mean: {x.float().mean()}")
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
    def attention_scores(self, attention_scores):
        self._attention_scores = attention_scores

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        attention_scores = ((query @ key.transpose(-2, -1)) / (math.sqrt(d_k)))

        if mask is not None:
            mask = mask.to(torch.bool)  # Ensure mask is boolean
            mask = mask.expand_as(attention_scores)  # Expand mask dimensions
            attention_scores = attention_scores.float()  # Ensure attention scores are float32
            attention_scores.masked_fill_(mask == 0, -1e9)

        attention_scores = torch.softmax(attention_scores, dim=-1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return attention_scores @ value, attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        # print(f"MultiHeadAttentionBlock stats - Min: {x.float().min()}, Max: {x.float().max()}, Mean: {x.float().mean()}")
        return self.w_o(x)


class ResidualConnection(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        # print(f"ResidualConnection stats - Min: {x.float().min()}, Max: {x.float().max()}, Mean: {x.float().mean()}")
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        # print(f"EncoderBlock stats - Min: {x.float().min()}, Max: {x.float().max()}, Mean: {x.float().mean()}")
        x = self.residual_connections[0](x, lambda x_i: self.self_attention_block(x_i, x_i, x_i, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        # print(f"Encoder stats - Min: {x.float().min()}, Max: {x.float().max()}, Mean: {x.float().mean()}")
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

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # print(f"DecoderBlock stats - Min: {x.float().min()}, Max: {x.float().max()}, Mean: {x.float().mean()}")
        x = self.residual_connections[0](x, lambda x_i: self.self_attention_block(x_i, x_i, x_i, tgt_mask))
        x = self.residual_connections[1](x, lambda x_j: self.cross_attention_block(x_j, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # print(f"Decoder stats - Min: {x.float().min()}, Max: {x.float().max()}, Mean: {x.float().mean()}")
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        return self.norm(x)


class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # print(f"ProjectionLayer stats - Min: {x.float().min()}, Max: {x.float().max()}, Mean: {x.float().mean()}")
        epsilon = 1e-8
        return torch.log(torch.softmax(self.proj(x), dim=-1) + epsilon)


class Transformer(nn.Module):
    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 src_embed: InputEmbeddings,
                 tgt_embed: InputEmbeddings,
                 src_pos: PositionalEncoding,
                 tgt_pos: PositionalEncoding,
                 projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # print(f"Transformer stats - Min: {src.float().min()}, Max: {src.float().max()}, Mean: {src.float().mean()}")
        # Encode the source sequence
        encoder_output = self.encode(src, src_mask)

        # Decode the target sequence
        decoder_output = self.decode(encoder_output, src_mask, tgt, tgt_mask)

        # Project the decoder output to the vocabulary space
        output = self.projection_layer(decoder_output)

        return output


class ChatbotDataset(Dataset):
    def __init__(self, dataset, tokenizer: Tokenizer, seq_len):
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.sos_token = torch.tensor([tokenizer.token_to_id('[SOS]')], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer.token_to_id('[PAD]')], dtype=torch.int64)
        self.unk_token = torch.tensor([tokenizer.token_to_id('[UNK]')], dtype=torch.int64)

    def __len__(self):
        return len(self.dataset.dataset.keys())

    def __getitem__(self, index):
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
    text = [example for example in dataset]
    text = [' '.join(t.replace('\\N', ' ').replace('.', ' ').replace('\'', '').replace('?', ' ').split()) for t in text]
    text = [remove_stutter(t) for t in text]
    text = [' '.join(t.replace('-', ' ').split()) for t in text]
    text_dict = {}
    for i, t in enumerate(text[:-1]):
        text_dict[i] = {'context': t, 'response': text[i + 1]}

    return text_dict


def get_or_build_tokenizer(config: dict, dataset: dict = None) -> Tokenizer:
    tokenizer_path = Path(config['tokenizer_path'])
    if not tokenizer_path.exists():
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))

        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = trainers.WordLevelTrainer(
            special_tokens=['[UNK]', '[PAD]', '[SOS]', '[EOS]'],
            min_frequency=2
        )

        lexicon = [dataset[i]['context'] for i in dataset.keys()] + [dataset[len(dataset) - 1]['response']]
        tokenizer.train_from_iterator(lexicon, trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer


def split_dataset(dataset: dict, tokenizer: Tokenizer, dataset_class, config: dict):
    # Split the dataset into training and validation sets
    train_dataset_size = int(0.9 * len(dataset))
    val_dataset_size = len(dataset) - train_dataset_size
    train_dataset_raw, val_dataset_raw = random_split(dataset, (train_dataset_size, val_dataset_size))

    # Create the dataset
    train_dataset = dataset_class(train_dataset_raw, tokenizer, config['seq_len'])
    val_dataset = dataset_class(val_dataset_raw, tokenizer, config['seq_len'])

    return train_dataset, val_dataset


def custom_collate_fn(batch):
    filtered_batch = []
    for idx, item in enumerate(batch):
        try:
            # Ensure item has all required keys
            if not all(key in item for key in ['encoder_input', 'decoder_input', 'encoder_mask', 'decoder_mask', 'context', 'response']):
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
            'context': [],
            'response': []
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


def print_gpu_utilization():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used // 1024 ** 2:.2f} MB.")


def autoregressive_decode(model, input_tokens, seq_len, tokenizer, device):
    # Start decoding with the start-of-sequence token
    sos_token = torch.tensor([tokenizer.token_to_id('[SOS]')], dtype=torch.long, device=device).unsqueeze(0)
    sos_token = sos_token.expand(input_tokens.size(0), -1)  # Ensure sos_token matches the batch size of input_tokens
    decoded_tokens = sos_token

    for _ in range(seq_len):
        with torch.no_grad():
            # Use model.forward to get the output tokens
            output_tokens = model.forward(input_tokens, decoded_tokens)

        # Get the last token from the output (the most recently predicted token)
        next_token = output_tokens[:, -1, :].argmax(dim=-1, keepdim=True)

        # Append the predicted token to the decoded sequence
        decoded_tokens = torch.cat((decoded_tokens, next_token), dim=1)

        # Stop decoding if the end-of-sequence token is produced
        if (next_token == tokenizer.token_to_id('[EOS]')).all():
            break

    return decoded_tokens


def train_chatbot(raw_dataset: list, config: dict):
    model_path = Path(config['model_path'])
    if not model_path.exists():
        dataset = preprocess_text_data(raw_dataset)
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

        model.to(device)
        criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.pad_token)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

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
        train_losses = []
        val_losses = []
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
                    for token in [train_dataset.sos_token, train_dataset.eos_token, train_dataset.pad_token, train_dataset.unk_token]:
                        loss = torch.where(labels == token.to(device), torch.tensor(0.0).to(device), loss)

                    # Sum the loss to get a scalar value
                    loss = loss.sum()

                if torch.any(torch.isnan(loss)) or torch.any(torch.isinf(loss)):
                    print(f"NaN or Inf detected in loss at batch {idx + 1} - {batch['context']}, {batch['response']}")
                    continue

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()
                scheduler.step()

                train_batch_end_time = time.time()
                if train_batch_end_time - train_batch_start_time > 3:
                    print(f'batch {idx + 1} took {train_batch_end_time - train_batch_start_time} seconds')


            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            print_gpu_utilization()
            print(f'Epoch {epoch + 1}/{config["num_epochs"]}, Loss: {train_loss}')
            epoch_iterator.set_postfix({f'train_loss': f'{train_loss:6.3f}'})

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
            print(f'Learning rate after epoch {epoch + 1}: {optimizer.param_groups[0]["lr"]}')

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

        plt.plot(range(1, config['num_epochs'] + 1), train_losses, label='Training Loss')
        # plt.plot(range(1, config['num_epochs'] + 1), val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        torch.save(model.state_dict(), config['model_path'])
    else:
        pass


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


def preprocess_sentence(sentence, tokenizer: Tokenizer, seq_len, device: torch.device):
    tokens = tokenizer.encode(sentence).ids
    tokens = [tokenizer.token_to_id('[SOS]')] + tokens + [tokenizer.token_to_id('[EOS]')]
    tokens = tokens[:seq_len] + [tokenizer.token_to_id('[PAD]')] * (seq_len - len(tokens))
    return torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)


def chatbot_predict(model, sentence, tokenizer: Tokenizer, seq_len: int, device: torch.device, output_length: int = 7):
    model.eval()
    input_tokens = preprocess_sentence(sentence, tokenizer, seq_len, device)

    # Start decoding with the start-of-sequence token
    sos_token = torch.tensor([tokenizer.token_to_id('[SOS]')], dtype=torch.long, device=device).unsqueeze(0)
    decoded_tokens = sos_token

    while len(decoded_tokens) < output_length:
        with torch.no_grad():
            output_tokens = model.forward(input_tokens, decoded_tokens)
            next_token_logits = output_tokens[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)

            print("Debug: next_token logits = ", next_token_logits)
            print("Debug: next_token = ", next_token)
            print("Debug: next_token item = ", next_token.item())

            if next_token.item() >= 4:
                decoded_tokens = torch.cat((decoded_tokens, next_token), dim=1)

    output_ids = decoded_tokens.squeeze().tolist()

    print("Debug: output_ids before checking =", output_ids)
    if isinstance(output_ids, int):
        output_ids = [output_ids]

    print("Debug: output_ids after checking =", output_ids)

    output_sentence = tokenizer.decode(output_ids, skip_special_tokens=True)
    return output_sentence


def chat(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = Tokenizer.from_file(str(config['tokenizer_path']))
    vocab_size = len(tokenizer.get_vocab())
    chatbot = load_chatbot(device=device, config=config, vocab_size=vocab_size)
    name = 'Tim'
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
        'seq_len': 128,
        'batch_size': 64,
        'd_model': 512,
        'h': 8,
        'N': 6,
        'd_ff': 2048,
        'num_epochs': 100,
        'learning_rate': 1e-5
    }
    print(f'Loading data took {time.time() - load_data_time:.2f} seconds')
    train_chatbot(raw_dataset=raw_dataset, config=config)
    chat(config=config)


if __name__ == '__main__':
    main()
