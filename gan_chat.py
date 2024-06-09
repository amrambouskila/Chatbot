import time
import torch
import matplotlib
import seaborn as sns
from tokenizers import tokenizers
from generic_transformer import build_transformer
from logger import create_logger
from text_data_api import get_text_data
from train_chatbot import preprocess_text_data, split_dataset, custom_collate_fn, get_or_build_tokenizer
from gan import TransformerGAN
from torch.utils.data import DataLoader
import pytorch_lightning as pl

matplotlib.use('Qt5Agg', force=True)
sns.set()


def train_gan(config, raw_dataset):
    # Preprocess text data
    dataset = preprocess_text_data(dataset=raw_dataset, seq_len=config['seq_len'], logger=logger)

    # Get or build tokenizer
    tokenizer = get_or_build_tokenizer(config=config)

    # Split the dataset into training and validation sets and convert them into torch Dataset objects
    train_dataset, val_dataset = split_dataset(dataset=dataset, tokenizer=tokenizer, config=config, logger=logger)

    # Create Dataloader objects for the training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=custom_collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=custom_collate_fn, pin_memory=True)

    # Create GAN model
    gan_model = TransformerGAN(config)

    # Initialize PyTorch Lightning trainer
    trainer = pl.Trainer(max_epochs=config['num_epochs'], devices=1 if torch.cuda.is_available() else 0)

    # Train the model
    trainer.fit(gan_model, train_loader, val_loader)
    torch.save(chatbot_gan.state_dict(), config['GAN_model_path'])
    return chatbot_gan


def load_model(config: dict) -> TransformerGAN:
    # Create GAN model
    model = TransformerGAN(config=config)

    # Load the trained model
    checkpoint = torch.load(config['GAN_model_path'], map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.load_state_dict(checkpoint['state_dict'])

    return model


def chat_with_model(config: dict):
    model = load_model(config=config)
    model.eval()
    tokenizer = get_or_build_tokenizer(config=config)

    model.to(torch.device('cpu'))

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        # Tokenize the input
        user_tokens = tokenizer.encode(user_input, return_tensors='pt')
        user_tokens = user_tokens[:, :config['seq_len'] - 2]  # Ensure it fits the sequence length

        # Create a fake target tensor
        tgt_tokens = torch.tensor([[config['vocab_size'] - 1]], dtype=torch.long)  # Start with [SOS] token
        tgt_tokens = tgt_tokens.repeat(1, config['seq_len'] - 1)  # Fill with [PAD] tokens

        # Create masks
        src_mask = (user_tokens != tokenizer.pad_token_id).unsqueeze(1).unsqueeze(2)
        tgt_mask = torch.triu(torch.ones((config['seq_len'], config['seq_len']), dtype=torch.uint8),
                              diagonal=1).unsqueeze(0)

        with torch.no_grad():
            output = model(user_tokens, tgt_tokens, src_mask, tgt_mask)

        # Get the predicted tokens
        predicted_tokens = torch.argmax(output, dim=-1).squeeze().tolist()

        # Decode the predicted tokens
        response = tokenizer.decode(predicted_tokens, skip_special_tokens=True)
        print(f"Bot: {response}")


if __name__ == '__main__':
    # Create a config dictionary in order to change deep learning hyperparameters
    config = {
        'subcategory': 'World War II',
        # 'subcategory': 'DBZ',
        'dropout': 0.1,
        'seq_len': 32,
        'batch_size': 128,
        'd_model': 512,
        'h': 8,
        'N': 6,
        'd_ff': 2048,
        'num_epochs': 2,
        'learning_rate': 1e-5,
        'patience': 5,
        'transfer_learning': False,
    }

    # Create automated naming
    suffix = '_TL' if config['transfer_learning'] else ''
    subcategory = config["subcategory"]
    subcategory = subcategory.replace(' ', '_')
    config['model_path'] = f'./{subcategory}_LLM_{config["num_epochs"]}{suffix}.pt'
    config['tokenizer_path'] = f'./{subcategory}_tokenizer.json'

    # Create a logger object to track training progress
    logger = create_logger(__name__, __file__, f'{subcategory}_LLM_{config["num_epochs"]}{suffix}_Chatbot')

    # Load data into memory using project gutenberg's library
    load_data_time = time.time()
    txt_links, all_sentences = get_text_data(subcategory=config["subcategory"], logger=logger)
    # all_sentences = load_dataset("Fishball02/anime-subtitle-dragon-ball")['train']['text']
    logger.info(f'Loading data took {time.time() - load_data_time:.2f} seconds')
    chatbot_gan = train_gan(config=config, raw_dataset=all_sentences)
    chat_with_model(config=config)
