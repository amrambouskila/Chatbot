import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
from transformers import GPT2LMHeadModel, GPT2Tokenizer, logging
import numpy as np
import random
from datasets import load_dataset

logging.set_verbosity_info()


class ChatbotDataset(Dataset):
    def __init__(self, dataset, tokenizer: GPT2Tokenizer, seq_len):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        context = self.dataset[index]['context']
        response = self.dataset[index]['response']
        context_tokens = self.tokenizer.encode(context, truncation=True, max_length=self.seq_len)
        response_tokens = self.tokenizer.encode(response, truncation=True, max_length=self.seq_len)

        context_tokens = [self.tokenizer.pad_token_id] * (self.seq_len - len(context_tokens)) + context_tokens
        response_tokens = [self.tokenizer.pad_token_id] * (self.seq_len - len(response_tokens)) + response_tokens

        encoder_input = torch.tensor(context_tokens, dtype=torch.long)
        decoder_input = torch.tensor([self.tokenizer.bos_token_id] + response_tokens[:-1], dtype=torch.long)
        labels = torch.tensor(response_tokens, dtype=torch.long)

        return {
            'encoder_input': encoder_input,
            'decoder_input': decoder_input,
            'label': labels
        }


def preprocess_text_data(raw_dataset: list):
    dataset = [{'context': raw_dataset[i], 'response': raw_dataset[i + 1]} for i in range(len(raw_dataset) - 1)]
    return dataset


def split_dataset(dataset: list, tokenizer: GPT2Tokenizer, seq_len: int):
    dataset = ChatbotDataset(dataset, tokenizer, seq_len)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset


def custom_collate_fn(batch):
    encoder_inputs = torch.stack([item['encoder_input'] for item in batch])
    decoder_inputs = torch.stack([item['decoder_input'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    return {
        'encoder_input': encoder_inputs,
        'decoder_input': decoder_inputs,
        'label': labels
    }


def train_epoch(model, train_loader, optimizer, scheduler, scaler, device):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        with autocast():
            outputs = model(
                input_ids=batch['encoder_input'].to(device),
                attention_mask=(batch['encoder_input'] != model.config.pad_token_id).to(device),
                labels=batch['label'].to(device)
            )
            loss = outputs.loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def validate_epoch(model, val_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            outputs = model(
                input_ids=batch['encoder_input'].to(device),
                attention_mask=(batch['encoder_input'] != model.config.pad_token_id).to(device),
                labels=batch['label'].to(device)
            )
            loss = outputs.loss
            total_loss += loss.item()
    return total_loss / len(val_loader)


def train_chatbot(raw_dataset: list, config: dict):
    dataset = preprocess_text_data(raw_dataset)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'The neural network will be running on {device}')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset, val_dataset = split_dataset(dataset, tokenizer, config['seq_len'])
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,
                              collate_fn=custom_collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=custom_collate_fn,
                            pin_memory=True)

    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'])
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader) * config['num_epochs'])

    scaler = GradScaler()
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(config['num_epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, scaler, device)
        val_loss = validate_epoch(model, val_loader, device)
        print(f"Epoch {epoch + 1}, Training Loss: {train_loss}, Validation Loss: {val_loss}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), config['model_path'])
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break


def autoregressive_decode(model, input_tokens, seq_len, tokenizer, device):
    model.eval()
    sos_token = torch.tensor([tokenizer.bos_token_id], dtype=torch.long, device=device).unsqueeze(0)
    decoded_tokens = sos_token

    for _ in range(seq_len):
        with torch.no_grad():
            output = model(input_ids=input_tokens, decoder_input_ids=decoded_tokens)
            next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            decoded_tokens = torch.cat((decoded_tokens, next_token), dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break

    return decoded_tokens


def beam_search_decode(model, input_tokens, seq_len, tokenizer, device, num_beams=3):
    model.eval()
    generated = model.generate(
        input_ids=input_tokens,
        max_length=seq_len,
        num_beams=num_beams,
        early_stopping=True
    )
    return generated


def nucleus_sampling_decode(model, input_tokens, seq_len, tokenizer, device, top_p=0.9):
    model.eval()
    generated = model.generate(
        input_ids=input_tokens,
        max_length=seq_len,
        top_p=top_p,
        top_k=0,
        do_sample=True
    )
    return generated


def chatbot_predict(model, sentence, tokenizer: GPT2Tokenizer, seq_len: int, device: torch.device,
                    method='beam_search'):
    input_tokens = tokenizer.encode(sentence, return_tensors='pt', truncation=True, max_length=seq_len).to(device)
    if method == 'beam_search':
        output_tokens = beam_search_decode(model, input_tokens, seq_len, tokenizer, device)
    elif method == 'nucleus_sampling':
        output_tokens = nucleus_sampling_decode(model, input_tokens, seq_len, tokenizer, device)
    else:
        output_tokens = autoregressive_decode(model, input_tokens, seq_len, tokenizer, device)

    output_sentence = tokenizer.decode(output_tokens.squeeze(), skip_special_tokens=True)
    return output_sentence


def chat(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(torch.load(config['model_path']))
    model.to(device)
    model.eval()

    print('Type `quit` to end chatting')
    while True:
        sentence = input('You: ')
        if sentence.lower() == 'quit':
            break
        output = chatbot_predict(model, sentence, tokenizer, config['seq_len'], device)
        print(f'Chatbot: {output}')


def main():
    raw_dataset = load_dataset("Fishball02/anime-subtitle-dragon-ball")['train']['text']
    config = {
        'model_path': './chatbot_model.pt',
        'seq_len': 32,
        'batch_size': 8,
        'd_model': 512,
        'num_epochs': 10,
        'learning_rate': 5e-5
    }
    train_chatbot(raw_dataset=raw_dataset, config=config)
    chat(config=config)


if __name__ == '__main__':
    main()
