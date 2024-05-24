import re
import time
import torch
import pynvml
from tqdm import tqdm
import torch.nn as nn
from pathlib import Path
from typeguard import typechecked
from tokenizers.models import WordLevel
from torch.cuda.amp import GradScaler, autocast
from tokenizers.trainers import WordLevelTrainer
from tokenizers import Tokenizer, pre_tokenizers
from torch.utils.data.dataloader import default_collate
from generic_transformer import build_transformer, Transformer
from torch.utils.data import Dataset, DataLoader, random_split, dataset
from torch.optim.lr_scheduler import StepLR, CyclicLR, CosineAnnealingLR


@typechecked
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


@typechecked
def remove_stutter(text: str) -> str:
    stutter_pattern = re.compile(r'\b(\w)-\1(\w+)\b', re.IGNORECASE)
    corrected_text = stutter_pattern.sub(r'\1\2', text)
    return corrected_text


@typechecked
def preprocess_text_data(dataset: list) -> dict:
    text = [remove_stutter(' '.join(t.replace('\\N', ' ').split())).lower() for t in dataset]
    text = [re.sub(r'[^A-Za-z\s]', '', t) for t in text]
    text = [t for t in text if t != '' and len(t.split(' ')) > 1]

    text_dict = {}
    max_input_length = -1
    for i, t in enumerate(text[:-1]):
        n_words = len(t.split(' '))
        if n_words > max_input_length:
            max_input_length = n_words

        text_dict[i] = {'context': t, 'response': text[i + 1]}

    return text_dict


@typechecked
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


@typechecked
def split_dataset(dataset: dict, tokenizer: Tokenizer, config: dict):

    # Split the dataset into training and validation sets
    train_dataset_size = int(0.9 * len(dataset))
    val_dataset_size = len(dataset) - train_dataset_size
    train_dataset_raw, val_dataset_raw = random_split(dataset, (train_dataset_size, val_dataset_size))

    # Create the dataset
    train_dataset = ChatbotDataset(train_dataset_raw, tokenizer, config['seq_len'])
    val_dataset = ChatbotDataset(val_dataset_raw, tokenizer, config['seq_len'])

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


@typechecked
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


@typechecked
def zero_special_token_grads(model: Transformer, special_token_indices: list, embedding_layer_names: list):
    for name, param in model.named_parameters():
        if any(embedding_layer_name in name for embedding_layer_name in embedding_layer_names) and param.grad is not None:
            for idx in special_token_indices:
                if idx < param.grad.size(0):
                    param.grad[idx].zero_()


@typechecked
def print_gpu_utilization():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used // 1024 ** 2:.2f} MB.")


@typechecked
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


@typechecked
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


@typechecked
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


@typechecked
def train_chatbot(raw_dataset: list, config: dict):
    model_path = Path(config['model_path'])
    if not model_path.exists():
        train_info = {}
        dataset = preprocess_text_data(raw_dataset)
        train_info['dataset'] = dataset
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'The neural network will be running on {device}')
        tokenizer = get_or_build_tokenizer(config=config, dataset=dataset)
        train_dataset, val_dataset = split_dataset(dataset=dataset, tokenizer=tokenizer, config=config)
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

        # embedding_layer_names = find_embedding_layer_names(model)
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



