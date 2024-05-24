import nltk
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from typeguard import typechecked

matplotlib.use('Qt5Agg', force=True)
sns.set()


@typechecked
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
