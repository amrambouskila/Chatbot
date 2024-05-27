import time
import torch
import matplotlib
import seaborn as sns
from logger import create_logger
from tokenizers import Tokenizer
from typeguard import typechecked
from datasets import load_dataset
from typing import Optional, Union
from text_data_api import get_text_data
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from chatbot_analytics import word_analytics, plot_losses
from generic_transformer import build_transformer, Transformer
from train_chatbot import train_chatbot, get_or_build_tokenizer

matplotlib.use('Qt5Agg', force=True)
sns.set()


@typechecked
def load_chatbot(device: torch.device, config: dict, vocab_size: int, tokenizer: Optional[Union[Tokenizer, GPT2Tokenizer]]):
    # If transfer learning is enabled, the GPT2 model will be loaded onto the device
    if config['transfer_learning']:
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        model.resize_token_embeddings(len(tokenizer))
        model.load_state_dict(torch.load(config['model_path']))
        model.to(device)
        model.eval()

    # Otherwise, the generic transformer model will be loaded into memory
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

        model.load_state_dict(torch.load(config['model_path']))
        model.to(device)
        model.eval()

    return model


@typechecked
def preprocess_sentence(sentence: str, tokenizer: Union[Tokenizer, GPT2Tokenizer], config: dict, special_tokens: list, device: torch.device):
    seq_len = config['seq_len']
    if config['transfer_learning']:
        tokens = tokenizer.encode(sentence)
    else:
        tokens = tokenizer.encode(sentence).ids

    tokens = [special_tokens[0]] + tokens + [special_tokens[1]]
    tokens = tokens[:seq_len] + [special_tokens[2]] * (seq_len - len(tokens))
    return torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)


@typechecked
def chatbot_predict(
        model: Union[Transformer, GPT2LMHeadModel],
        sentence: str,
        tokenizer: Union[Tokenizer, GPT2Tokenizer],
        config: dict,
        device: torch.device,
):
    # Set the model to evaluation mode
    model.eval()

    # Extract parameters from the config dictionary
    seq_len = config['seq_len']
    transfer_learning = config['transfer_learning']

    # Create a list of special tokens for the model to avoid outputting
    if transfer_learning:
        special_tokens = [tokenizer.convert_tokens_to_ids('<|sos|>'), tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('<|pad|>'), tokenizer.convert_tokens_to_ids('<|unk|>')]
    else:
        special_tokens = [tokenizer.token_to_id('[SOS]'), tokenizer.token_to_id('[EOS]'), tokenizer.token_to_id('[PAD]'), tokenizer.token_to_id('[UNK]')]

    # Preprocess the input sentence and convert it to a tensor
    input_tokens = preprocess_sentence(sentence=sentence, tokenizer=tokenizer, config=config, special_tokens=special_tokens, device=device)

    # Start decoding with the start-of-sequence token
    sos_token = torch.tensor([special_tokens[0]], dtype=torch.long, device=device).unsqueeze(0)
    decoded_tokens = sos_token
    last_token_id = None

    while len(decoded_tokens[0]) < seq_len:
        with torch.no_grad():
            if transfer_learning:
                # Concatenate encoder and decoder inputs
                combined_input = torch.cat((input_tokens, decoded_tokens), dim=1)  # (batch_size, seq_len * 2)

                # Create attention mask for the combined input
                encoder_mask = (input_tokens != tokenizer.pad_token_id).long()  # (batch_size, seq_len)
                decoder_mask = (decoded_tokens != tokenizer.pad_token_id).long()  # (batch_size, seq_len)

                # Combined mask: encoder mask followed by decoder mask
                combined_mask = torch.cat((encoder_mask, decoder_mask), dim=1)  # (batch_size, seq_len * 2)

                # Ensure the combined_mask has the same number of dimensions as combined_input
                combined_mask = combined_mask.unsqueeze(1)  # (batch_size, 1, seq_len * 2)

                # Forward pass
                outputs = model(input_ids=combined_input, attention_mask=combined_mask)
                decoder_output_logits = outputs.logits

                # Since target labels correspond only to the decoder part, slice the output logits accordingly
                output_tokens = decoder_output_logits[:, input_tokens.size(1):, :]
            else:
                output_tokens = model.forward(input_tokens, decoded_tokens)

            next_token_logits = output_tokens[:, -1, :]

            # Mask out the PAD token logits and the last token logits
            next_token_logits[:, special_tokens] = float('-inf')
            if last_token_id is not None:
                next_token_logits[:, last_token_id] = float('-inf')

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

            # print(next_token.item())
            if next_token.item() not in special_tokens:
                decoded_tokens = torch.cat((decoded_tokens, next_token), dim=1)
                last_token_id = next_token.item()

            if next_token.item() == special_tokens[1]:
                if len(decoded_tokens[0]) >= seq_len:
                    break

    output_ids = decoded_tokens.squeeze().tolist()

    if isinstance(output_ids, int):
        output_ids = [output_ids]

    output_sentence = tokenizer.decode(output_ids, skip_special_tokens=True)
    return output_sentence


@typechecked
def chat(config: dict):
    # Assign the device that the model will use to perform calculations
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # get or build the tokenizer corresponding to the text dataset the chatbot was trained on
    tokenizer = get_or_build_tokenizer(config=config)

    # get the vocab size to help build the transformer model and load the chatbot into memory
    vocab_size = len(tokenizer.get_vocab())
    chatbot = load_chatbot(device=device, config=config, vocab_size=vocab_size, tokenizer=tokenizer)

    # Start chatting with the chatbot!
    name = 'Goku'
    print('Type `quit` to end chatting')
    while True:
        sentence = input('You: ')
        if 'quit' == sentence:
            break

        output = chatbot_predict(chatbot, sentence, tokenizer, config, device)
        print(f'{name}: {output}')


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
        'num_epochs': 1,
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

    # Run preliminary analytics on the text data
    word_analytics(sentences=all_sentences)

    # Instantiate a dictionary to track training metrics
    train_info = {
        'train_losses': [],
        'val_losses': [],
        'learning_rates': [],
        'epochs': [],
        'best_val_loss': float('inf'),
        'patience_counter': 0,
    }

    # Train the chatbot
    train_info = train_chatbot(raw_dataset=all_sentences, config=config, train_info=train_info, logger=logger)

    # Run performance analytics
    if len(train_info['train_losses']) == len(train_info['val_losses']) == len(train_info['learning_rates']) == len(train_info['epochs']):
        n_strings = len(train_info['dataset'].keys())
        preprocessed_sentences = [train_info['dataset'][i]['context'] for i in range(n_strings)] + [train_info['dataset'][n_strings - 1]['response']]
        word_analytics(sentences=preprocessed_sentences)
        plot_losses(train_info=train_info)

    # Communicate with the chatbot
    chat(config=config)
