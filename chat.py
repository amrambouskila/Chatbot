import torch
from logger import create_logger
from tokenizers import Tokenizer
from typeguard import typechecked
from datasets import load_dataset
from train_chatbot import train_chatbot
from text_data_api import get_text_data
from chatbot_analytics import run_analytics
from generic_transformer import build_transformer, Transformer


@typechecked
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


@typechecked
def preprocess_sentence(sentence: str, tokenizer: Tokenizer, seq_len: int, device: torch.device):
    tokens = tokenizer.encode(sentence).ids
    tokens = [tokenizer.token_to_id('[SOS]')] + tokens + [tokenizer.token_to_id('[EOS]')]
    tokens = tokens[:seq_len] + [tokenizer.token_to_id('[PAD]')] * (seq_len - len(tokens))
    return torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)


@typechecked
def chatbot_predict(model: Transformer, sentence: str, tokenizer: Tokenizer, seq_len: int, device: torch.device,
                    output_length: int = 32):
    model.eval()
    input_tokens = preprocess_sentence(sentence, tokenizer, seq_len, device)
    special_tokens = [tokenizer.token_to_id('[SOS]'), tokenizer.token_to_id('[EOS]'), tokenizer.token_to_id('[PAD]'),
                      tokenizer.token_to_id('[UNK]')]

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


@typechecked
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


if __name__ == '__main__':
    config = {
        'subcategory': 'World War II',
        'dropout': 0.1,
        'seq_len': 32,
        'batch_size': 128,
        'd_model': 512,
        'h': 8,
        'N': 6,
        'd_ff': 2048,
        'num_epochs': 10,
        'learning_rate': 1e-5
    }

    subcategory = config["subcategory"]
    config['model_path'] = f'./{subcategory}_LLM.pt'
    config['tokenizer_path'] = f'./{subcategory}_tokenizer.json'
    logger = create_logger(__name__, __file__, f'{subcategory}_Chatbot')

    # txt_links, all_sentences = get_text_data(subcategory=subcategory)
    all_sentences = load_dataset("Fishball02/anime-subtitle-dragon-ball")['train']['text']
    train_info = train_chatbot(raw_dataset=all_sentences, config=config)

    if train_info is not None:
        run_analytics(train_info=train_info, config=config)

    chat(config=config)
