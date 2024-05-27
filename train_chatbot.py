import re
import time
import torch
import pynvml
import string
import logging
from tqdm import tqdm
import torch.nn as nn
from pathlib import Path
from collections import Counter
from typeguard import typechecked
from nltk.corpus import stopwords
from typing import Union, Optional
from tokenizers.models import WordLevel
from torch.cuda.amp import GradScaler, autocast
from tokenizers.trainers import WordLevelTrainer
from tokenizers import Tokenizer, pre_tokenizers
from torch.utils.data.dataloader import default_collate
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, logging as hf_logging, get_linear_schedule_with_warmup
from generic_transformer import build_transformer, Transformer
from torch.utils.data import Dataset, DataLoader, random_split, dataset
from torch.optim.lr_scheduler import StepLR, CyclicLR, CosineAnnealingLR

hf_logging.set_verbosity_error()


@typechecked
class ChatbotDataset(Dataset):
    special_tokens = []

    def __init__(self, dataset: dataset.Subset, tokenizer: Union[Tokenizer, GPT2Tokenizer], seq_len: int, logger: logging.Logger, transfer_learning: bool):
        # Call the constructor for the parent torch Dataset class
        super().__init__()

        # Store variables that will be called in the __getitem__ method
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.logger = logger
        self.transfer_learning = transfer_learning

        # Depending on the tokenizer, different sos, eos, pad, and unk tokens are assigned
        if not transfer_learning:
            self.sos_token = torch.tensor([tokenizer.token_to_id('[SOS]')], dtype=torch.int64)
            self.eos_token = torch.tensor([tokenizer.token_to_id('[EOS]')], dtype=torch.int64)
            self.pad_token = torch.tensor([tokenizer.token_to_id('[PAD]')], dtype=torch.int64)
            self.unk_token = torch.tensor([tokenizer.token_to_id('[UNK]')], dtype=torch.int64)
        else:
            self.sos_token = torch.tensor([tokenizer.convert_tokens_to_ids('<|sos|>')])
            self.eos_token = torch.tensor([tokenizer.eos_token_id])
            self.pad_token = torch.tensor([tokenizer.convert_tokens_to_ids('<|pad|>')])
            self.unk_token = torch.tensor([tokenizer.convert_tokens_to_ids('<|unk|>')])

        # Store special tokens as class objects in order to access them freely later
        self.__class__.special_tokens = list({*self.__class__.special_tokens, *[self.sos_token, self.eos_token, self.pad_token, self.unk_token]})

    def __len__(self):
        return len(self.dataset.dataset.keys())

    def __getitem__(self, index: int):
        # Extract the context and response using the DataLoader during training and validation
        context = self.dataset.dataset[index]['context']
        response = self.dataset.dataset[index]['response']

        # Tokenize the context and response using the tokenizer
        if self.transfer_learning:
            context_tokens = self.tokenizer.encode(context)
            response_tokens = self.tokenizer.encode(response)
        else:
            context_tokens = self.tokenizer.encode(context).ids
            response_tokens = self.tokenizer.encode(response).ids

        # If the length of the context or response is greater than the sequence length, truncate the context or response
        full_context_length = len(context_tokens) + 2
        full_response_length = len(response_tokens) + 2

        if full_context_length > self.seq_len:
            context_tokens = context_tokens[:self.seq_len - 2]
            full_context_length = len(context_tokens) + 2

        if full_response_length > self.seq_len:
            response_tokens = response_tokens[:self.seq_len - 2]
            full_response_length = len(response_tokens) + 2

        # Add pad tokens after sos tokens if the context or response is smaller than the sequence length
        num_enc_padding_token = self.seq_len - full_context_length
        num_dec_padding_token = self.seq_len - full_response_length

        if num_enc_padding_token < 0 or num_dec_padding_token < 0:
            self.logger.error(f'Sentence is too long - Context Token Length: {len(context_tokens)}, Response Token Length: {len(response_tokens)}')
            raise ValueError(f'Sentence is too long - Context Token Length: {len(context_tokens)}, Response Token Length: {len(response_tokens)}')

        # Create the encoder input, decoder input and the label to input into the transformer model
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

        # Ensure that the data sizes are correct
        encoder_input_size = encoder_input.size(0)
        decoder_input_size = decoder_input.size(0)
        label_size = label.size(0)
        assert encoder_input_size == self.seq_len, f'sequence length must be {self.seq_len}, encoder input length is {encoder_input_size}'
        assert decoder_input_size == self.seq_len, f'sequence length must be {self.seq_len}, decoder input length is {decoder_input_size}'
        assert label_size == self.seq_len, f'sequence length must be {self.seq_len}, label length is {label_size}'

        # Encoder mask: [batch_size, 1, seq_len]
        encoder_mask = (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(1).type(torch.bool)

        # Decoder mask: [1, seq_len, seq_len]
        subsequent_mask = torch.triu(torch.ones((self.seq_len, self.seq_len), dtype=torch.uint8), diagonal=1).to(torch.bool)
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
    # Remove any existing stutters defined as repeating letters separated by hyphens
    stutter_pattern = re.compile(r'\b(\w)-\1(\w+)\b', re.IGNORECASE)
    corrected_text = stutter_pattern.sub(r'\1\2', text)
    return corrected_text


@typechecked
def remove_most_common_words(sentences: list, top_n: int):
    stop_words = set(stopwords.words('english'))
    processed_sentences = []
    for sentence in sentences:
        words = sentence.split(' ')
        filtered_words = [word for word in words if word not in stop_words]
        processed_sentence = ' '.join(filtered_words)
        processed_sentences.append(processed_sentence)

    combined_text = ' '.join(processed_sentences)

    # Remove punctuation and convert to lowercase
    translator = str.maketrans('', '', string.punctuation)
    normalized_text = combined_text.translate(translator).lower()

    # Tokenize the text
    words = normalized_text.split(' ')

    # Count the frequency of each word
    word_counts = Counter(words)

    # Get the most common words
    most_common_words = word_counts.most_common(top_n)

    print(f'Top {str(top_n) + " words" if top_n > 1 else "word"} being removed from the text data:\n{most_common_words}')

    # Remove the top n common words from the sentences
    final_sentences = []
    for preprocessed_sentence in processed_sentences:
        preprocessed_words = preprocessed_sentence.split(' ')
        filtered_preprocessed_words = [preprocessed_word for preprocessed_word in preprocessed_words if preprocessed_word not in most_common_words]
        final_sentence = ' '.join(filtered_preprocessed_words)
        final_sentences.append(final_sentence)

    return final_sentences


def preprocess_text_data(dataset: list, seq_len: int, logger: logging.Logger, top_n: int = 1) -> dict:
    # The max sentence length should be at most 2 words less than the sequence length to adjust for sos and eos tokens.
    max_sentence_length = seq_len - 2

    # The text might have stutters, weird symbols or characters, or some extra white spaces so that will all be removed.
    normalized_text = []
    for t in dataset:
        sentence = remove_stutter(' '.join(t.replace('\\N', ' ').split())).lower()
        sentence = re.sub(r'[^0-9A-Za-z\s]', '', sentence)
        sentence = sentence.strip()
        sentence = re.sub(r'\s+', ' ', sentence)
        if len(sentence.split(' ')) > 1:
            normalized_text.append(sentence)

    # texts that are longer than max_sentence_length will be split into valid sentences
    text = []
    for t in normalized_text:
        sentences = []
        words = t.split(' ')
        while len(words) > max_sentence_length:
            sentences.append(' '.join(words[:max_sentence_length]))
            words = words[max_sentence_length:]

        if len(words) > 0:
            sentences.append(' '.join(words))

        text.extend(sentences)

    # Remove the top n most common words from the text data to avoid overfitting
    text = remove_most_common_words(sentences=text, top_n=top_n)

    # Prepare the text data into a dictionary that can be used to build a torch Dataset object
    text_dict = {}
    max_input_length = -1
    for i, t in enumerate(text[:-1]):
        if len(t.split(' ')) > max_sentence_length:
            logger.error(len(t))

        n_words = len(t.split(' '))
        if n_words > max_input_length:
            max_input_length = n_words

        text_dict[i] = {'context': t, 'response': text[i + 1]}

    return text_dict


@typechecked
def get_or_build_tokenizer(config: dict, dataset: Optional[dict] = None) -> Union[GPT2Tokenizer, Tokenizer]:
    # If tranfer learning is enabled, the GPT2 tokenizer will be loaded into memory and configured
    if config['transfer_learning']:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        # Define the special tokens
        sos_token = "<|sos|>"
        pad_token = "<|pad|>"
        unk_token = "<|unk|>"

        # Add the special tokens to the tokenizer
        tokenizer.add_special_tokens({
            'pad_token': pad_token,
            'unk_token': unk_token,
            'additional_special_tokens': [sos_token]
        })

        # Resize the model's token embeddings to accommodate the new tokens
        return tokenizer

    # Otherwise, if the tokenizer doesn't exist in storage, it will be trained on the text dataset
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

    # If the tokenizer exists in storage, it will be loaded into memory
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer


@typechecked
def split_dataset(
        dataset: dict,
        tokenizer: Union[Tokenizer, GPT2Tokenizer],
        config: dict,
        logger: logging.Logger
) -> (ChatbotDataset, ChatbotDataset):

    # Split the dataset into training and validation sets
    train_dataset_size = int(0.9 * len(dataset))
    val_dataset_size = len(dataset) - train_dataset_size
    train_dataset_raw, val_dataset_raw = random_split(dataset, (train_dataset_size, val_dataset_size))

    # Create the torch Dataset objects
    train_dataset = ChatbotDataset(dataset=train_dataset_raw, tokenizer=tokenizer, seq_len=config['seq_len'], logger=logger, transfer_learning=config['transfer_learning'])
    val_dataset = ChatbotDataset(dataset=val_dataset_raw, tokenizer=tokenizer, seq_len=config['seq_len'], logger=logger, transfer_learning=config['transfer_learning'])

    return train_dataset, val_dataset


def custom_collate_fn(batch: list):
    # To avoid any incorrect data from loading into the training device, this function will filter out any invalid data
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
def print_gpu_utilization(logger: logging.Logger, scale: str = 'MB'):
    # Initialize the NVML library and get the handle to the first GPU device
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)

    # Access the used memory by the GPU to provide console output
    if scale == 'MB':
        logger.info(f"GPU memory occupied: {info.used // 1024 ** 2:.2f} MB.")
    else:
        logger.info(f"GPU memory occupied: {info.used // 1024 ** 3:.2f} GB.")


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
def autoregressive_decode(model: Transformer, input_tokens: torch.Tensor, seq_len: int, tokenizer: Tokenizer, device: torch.device):
    sos_token = torch.tensor([tokenizer.token_to_id('[SOS]')], dtype=torch.long, device=device).unsqueeze(0)
    batch_size = input_tokens.size(0)
    decoded_tokens = sos_token.repeat(batch_size, 1)

    for _ in range(seq_len - 1):  # -1 because we already have the sos_token
        with torch.no_grad():
            output = model.forward(src=input_tokens, tgt=decoded_tokens)
            next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
            decoded_tokens = torch.cat((decoded_tokens, next_token), dim=1)

            # Stop decoding if the end-of-sequence token is produced
            if (next_token == tokenizer.token_to_id('[EOS]')).all().item():
                break

    # Pad the sequence if it's shorter than seq_len
    if decoded_tokens.size(1) < seq_len:
        padding = torch.full((batch_size, seq_len - decoded_tokens.size(1)), tokenizer.token_to_id('[PAD]'), device=device, dtype=torch.long)
        decoded_tokens = torch.cat((decoded_tokens, padding), dim=1)

    # Truncate the sequence if it's longer than seq_len
    decoded_tokens = decoded_tokens[:, :seq_len]

    # Adjust the size of decoded_tokens to match the input batch size
    if decoded_tokens.size(0) != batch_size:
        decoded_tokens = decoded_tokens[:batch_size]

    return decoded_tokens


@typechecked
def train_chatbot(raw_dataset: list, config: dict, train_info: dict, logger: logging.Logger):
    # Create a Path object from the model path in the config dict to determine if the model has already been trained
    model_path = Path(config['model_path'])
    if not model_path.exists():
        # Extract the number of epochs from the config dictionary as it will be used multiple times
        num_epochs = config['num_epochs']

        # Preprocess the text data so that it is clean and prepared for training a chatbot
        dataset = preprocess_text_data(dataset=raw_dataset, seq_len=config['seq_len'], logger=logger)

        # Store the cleaned text dataset for later analysis
        train_info['dataset'] = dataset

        # Assign the deep learning to a computing device (GPU, TPU, or CPU)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'The neural network will be running on {device}')

        # Build a tokenizer using the cleaned text dataset if it doesn't exist
        tokenizer = get_or_build_tokenizer(config=config, dataset=dataset)

        # Split the dataset into training and validation sets and convert them into torch Dataset objects
        train_dataset, val_dataset = split_dataset(dataset=dataset, tokenizer=tokenizer, config=config, logger=logger)

        # Create Dataloader objects for the training and validation sets
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=custom_collate_fn, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=custom_collate_fn, pin_memory=True)

        # Get parameters for model construction
        vocab_size = len(tokenizer.get_vocab())
        total_batches = len(train_loader)

        # Different models will be built depending on if transfer learning is enabled or not
        # Different schedulers and optimizers can be used to find how fast a global minimum is found.
        if config['transfer_learning']:
            model = GPT2LMHeadModel.from_pretrained('gpt2')
            model.to(device)
            model.resize_token_embeddings(len(tokenizer))
            criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.convert_tokens_to_ids('<|pad|>'))
            optimizer = AdamW(model.parameters(), lr=config['learning_rate'])
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                        num_training_steps=total_batches * num_epochs)

        else:
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
            criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id('[PAD]'))
            optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

            # scheduler = CosineAnnealingLR(optimizer, T_max=total_batches * config['num_epochs'])
            # scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
            scheduler = CyclicLR(
                optimizer,
                base_lr=config['learning_rate'],
                max_lr=1e-3,
                step_size_up=int(total_batches / 2),
                mode='triangular2'
            )

        # Instantiate the gradient scaler to avoid vanishing gradients
        scaler = GradScaler()

        # Initialize an interator for the epochs to generate a cool loop monitor
        epoch_iterator = tqdm(range(num_epochs), desc=f'Epoch counter')

        # Start training
        for epoch in epoch_iterator:
            # Keep track of progress and time
            epoch_time = time.time()
            train_loss = 0
            last_printed_progress = 0

            # Set model to train mode and start training loop
            model.train()
            for idx, batch in enumerate(train_loader):
                # Keep track of how long it takes to go through one batch and which batch is currently in memory
                train_batch_start_time = time.time()
                progress = (idx + 1) / total_batches * 100

                if int(progress) // 10 > last_printed_progress:
                    last_printed_progress = int(progress) // 10
                    logger.info(f'Epoch {epoch + 1}/{config["num_epochs"]} Training Loop is {progress:.2f}% completed')

                # Extract all the tensors from the DataLoader object and send them to the training device
                encoder_input = batch['encoder_input'].to(device, non_blocking=True)
                decoder_input = batch['decoder_input'].to(device, non_blocking=True)
                encoder_mask = batch['encoder_mask'].to(device, non_blocking=True)
                decoder_mask = batch['decoder_mask'].to(device, non_blocking=True)
                labels = batch['label'].to(device, non_blocking=True)

                # Zero out the gradients from the previous batch
                optimizer.zero_grad()

                # Implement mixed precision by using autocast
                with autocast():
                    # If transfer learning is enabled, the model expects concatenated inputs and masks
                    if config['transfer_learning']:
                        # Concatenate encoder and decoder inputs
                        combined_input = torch.cat((encoder_input, decoder_input), dim=1)  # (batch_size, seq_len * 2)

                        # Create attention mask for the combined input
                        encoder_mask = (encoder_input != tokenizer.pad_token_id).long()  # (batch_size, seq_len)
                        decoder_mask = (decoder_input != tokenizer.pad_token_id).long()  # (batch_size, seq_len)

                        # Combined mask: encoder mask followed by decoder mask
                        combined_mask = torch.cat((encoder_mask, decoder_mask), dim=1)  # (batch_size, seq_len * 2)

                        # Ensure the combined_mask has the same number of dimensions as combined_input
                        combined_mask = combined_mask.unsqueeze(1)  # (batch_size, 1, seq_len * 2)

                        # Forward pass
                        outputs = model(input_ids=combined_input, attention_mask=combined_mask)
                        decoder_output_logits = outputs.logits

                        # Since target labels correspond only to the decoder part, slice the output logits accordingly
                        output = decoder_output_logits[:, encoder_input.size(1):, :]

                        # Calculate loss
                        loss = criterion(output.reshape(-1, vocab_size), labels.view(-1))

                    # Otherwise, the transformer model built in generic_transformer.py will be used
                    else:
                        # Generate output using the transformer model
                        output = model.forward(encoder_input, decoder_input, encoder_mask, decoder_mask)

                        # Calculate loss
                        loss = criterion(output.view(-1, vocab_size), labels.view(-1))

                # If nans are encountered, skip that batch to avoid runtime exceptions
                if torch.any(torch.isnan(loss)) or torch.any(torch.isinf(loss)):
                    logger.info(f"NaN or Inf detected in loss at batch {idx + 1} - {batch['context']}, {batch['response']}")
                    continue

                # Commence backpropagation
                scaler.scale(loss).backward()

                # Clip gradients to avoid exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

                # Update optimizer and scaler
                scaler.step(optimizer)
                scaler.update()

                # Update training loss
                train_loss += loss.item()

                # Update scheduler
                scheduler.step()

                # Keep track of how long it takes to go through one batch
                train_batch_end_time = time.time()
                if train_batch_end_time - train_batch_start_time > 3:
                    logger.info(f'batch {idx + 1} took {train_batch_end_time - train_batch_start_time} seconds')


            # Calculate average training loss
            train_loss /= len(train_loader)

            # Store training metrics in the train_info dictionary
            train_info['train_losses'].append(train_loss)
            train_info['learning_rates'].append(optimizer.param_groups[0]["lr"])

            # Set the model to evaluation mode
            model.eval()

            # Keep track of progress
            val_loss = 0

            # Disable gradient calculation during validation
            with torch.no_grad():
                for idx, batch in enumerate(val_loader):
                    # Keep track of how long it takes to go through one batch and which batch is currently in memory
                    val_batch_start_time = time.time()
                    progress = (idx + 1) / total_batches * 100

                    if int(progress) // 10 > last_printed_progress:
                        last_printed_progress = int(progress) // 10
                        logger.info(f'Epoch {epoch + 1}/{config["num_epochs"]} Training Loop is {progress:.2f}% completed')

                    # Extract all the tensors from the DataLoader object and send them to the training device
                    encoder_input = batch['encoder_input'].to(device, non_blocking=True)
                    encoder_mask = batch['encoder_mask'].to(device, non_blocking=True)
                    decoder_input = batch['decoder_input'].to(device, non_blocking=True)
                    labels = batch['label'].to(device, non_blocking=True)

                    # If transfer learning is enabled, the model expects concatenated inputs and masks
                    if config['transfer_learning']:
                        # Concatenate encoder and decoder inputs
                        combined_input = torch.cat((encoder_input, decoder_input), dim=1)  # (batch_size, seq_len * 2)

                        # Create attention mask for the combined input
                        encoder_mask = (encoder_input != tokenizer.pad_token_id).long()  # (batch_size, seq_len)
                        decoder_mask = (decoder_input != tokenizer.pad_token_id).long()  # (batch_size, seq_len)

                        # Combined mask: encoder mask followed by decoder mask
                        combined_mask = torch.cat((encoder_mask, decoder_mask), dim=1)  # (batch_size, seq_len * 2)

                        # Ensure the combined_mask has the same number of dimensions as combined_input
                        combined_mask = combined_mask.unsqueeze(1)  # (batch_size, 1, seq_len * 2)

                        # Forward pass
                        outputs = model(input_ids=combined_input, attention_mask=combined_mask)
                        decoder_output_logits = outputs.logits

                        # Since target labels correspond only to the decoder part, slice the output logits accordingly
                        output = decoder_output_logits[:, encoder_input.size(1):, :]
                        loss = criterion(output.reshape(-1, vocab_size), labels.view(-1))

                    # Otherwise, the transformer model built in generic_transformer.py will be used
                    else:
                        # Inference validation
                        output_tokens = autoregressive_decode(model, encoder_input, config['seq_len'], tokenizer, device)
                        output = model.forward(encoder_input, output_tokens, encoder_mask, None)
                        loss = criterion(output.view(-1, vocab_size), batch['label'].view(-1).to(device))

                    # If nans are encountered, skip that batch to avoid runtime exceptions
                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.info(f"NaN or Inf detected in loss at batch {idx + 1} - {batch['context']}, {batch['response']}")
                        continue

                    # Update validation loss
                    val_loss += loss.item()

                    # Keep track of how long it takes to go through one batch
                    val_batch_end_time = time.time()
                    if val_batch_end_time - val_batch_start_time > 3:
                        logger.info(f'batch {idx + 1} took {val_batch_end_time - val_batch_start_time} seconds')

            # Calculate average validation loss
            val_loss /= len(val_loader)

            # Store validation metrics in the train_info dictionary
            train_info['val_losses'].append(val_loss)
            train_info['epochs'].append(epoch + 1)

            # Calculate the current patience being experienced by the gradient
            if val_loss < train_info['best_val_loss']:
                train_info['best_val_loss'] = val_loss
                train_info['patience_counter'] = 0
            else:
                train_info['patience_counter'] += 1

            # Performance console output
            monitoring_string = f'Epoch {epoch + 1}/{config["num_epochs"]} took {time.time() - epoch_time:.2f} seconds, Loss: {train_loss},  Validation Loss: {val_loss}, Learning rate: {optimizer.param_groups[0]["lr"]}'
            logger.info(monitoring_string)
            epoch_iterator.set_postfix({f'Epoch {epoch + 1}/{config["num_epochs"]}': monitoring_string})
            print_gpu_utilization(logger=logger)

            # Determine if early stopping is necessary
            if train_info['patience_counter'] >= config['patience']:
                logger.info("Early stopping triggered.")
                break

        torch.save(model.state_dict(), config['model_path'])
        return train_info
