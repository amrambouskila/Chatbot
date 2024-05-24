import nltk
import logging
import requests
from bs4 import BeautifulSoup
from typeguard import typechecked

nltk.download('punkt')


@typechecked
def get_subcategory_link(url: str, subcategory: str, logger: logging.Logger):
    # Send a GET request to the genre page
    response = requests.get(url)

    if response.status_code == 200:
        # Parse the HTML content of the genre page
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find the subcategory link that matches the query
        subcategory_link_tag = soup.find('a', href=True, string=lambda text: text and subcategory.lower() in text.lower())

        if subcategory_link_tag:
            subcategory_url = 'https://www.gutenberg.org' + subcategory_link_tag['href']
            return subcategory_url
        else:
            subcategory_links = soup.find_all('a', href=True)
            for link in subcategory_links:
                href = link['href']
                text = link.get_text(strip=True)
                if href and text:
                    logger.info(f"Available Options: {text}, Href: https://www.gutenberg.org{href}")

            raise ValueError(f"No subcategory link found for query: {subcategory}")
    else:
        raise ValueError(f"No subcategory link found for query: {subcategory}")


@typechecked
def get_book_links(subcategory_url: str, logger: logging.Logger):
    start_index = 1
    txt_file_links = {}

    while True:
        if start_index == 1:
            paginated_url = f"{subcategory_url}"
        else:
            paginated_url = f"{subcategory_url}?start_index={start_index}"

        response = requests.get(paginated_url)

        if response.status_code == 200:
            logger.info(f'Accessed {paginated_url}')
            soup = BeautifulSoup(response.content, 'html.parser')
            book_links = soup.find_all('li', class_='booklink')

            # If no more book links are found, break the loop
            if not book_links:
                break

            for i, book in enumerate(book_links):
                title_tag = book.find('span', class_='title')
                subtitle_tag = book.find('span', class_='subtitle')

                if title_tag and subtitle_tag:
                    title = title_tag.text
                    author = subtitle_tag.text

                    book_page_link = 'https://www.gutenberg.org' + book.find('a')['href']
                    book_response = requests.get(book_page_link)
                    if book_response.status_code == 200:
                        book_soup = BeautifulSoup(book_response.content, 'html.parser')
                        txt_link_tag = book_soup.find('a', href=True, string='Plain Text UTF-8')
                        if txt_link_tag:
                            txt_file_link = txt_link_tag['href']
                            txt_file_links[f'{title}_{author}'] = txt_file_link
                        else:
                            logger.info(f'No .txt link found for book {i + 1}')
                    else:
                        logger.info(f"Failed to retrieve the book page for book {i + 1}. Status code: {book_response.status_code}")
                else:
                    logger.info(f'Title or author not found for book {i + 1}')

            if len(book_links) < 25:
                break

            start_index += 25
        else:
            raise ValueError(f"No subcategory link found for query: {subcategory} - Status code: {response.status_code}")

    return txt_file_links


@typechecked
def get_sentences(txt_links: dict, logger: logging.Logger, intro_pct: float = 0.02):
    all_sentences = []

    for link in txt_links.values():
        # Construct the full URL
        full_url = 'https://www.gutenberg.org' + link

        # Fetch the .txt file content
        response = requests.get(full_url)
        if response.status_code == 200:
            content = response.text

            # Tokenize the text into words
            words = nltk.word_tokenize(content)

            # Remove the first and last 2% of the words as untext
            remaining_words = words[int(intro_pct * len(words)):-int(intro_pct * len(words))]

            # Join the remaining words back into a string
            remaining_text = ' '.join(remaining_words)

            # Split the remaining text into sentences
            sentences = nltk.sent_tokenize(remaining_text)

            # Add the sentences to the list of all sentences
            all_sentences.extend(sentences)
        else:
            print(f"Failed to retrieve the .txt file from {full_url}. Status code: {response.status_code}")

    return all_sentences


@typechecked
def get_text_data(subcategory: str, logger: logging.Logger):
    url = 'https://www.gutenberg.org/ebooks/bookshelf/'

    # Get the link to the subcategory page
    subcategory_url = get_subcategory_link(url=url, subcategory=subcategory, logger=logger)

    if subcategory_url:
        # Get the list of book .txt file links from the subcategory page
        txt_links = get_book_links(subcategory_url=subcategory_url, logger=logger)

        # Get all sentences from the list of .txt file links
        all_sentences = get_sentences(txt_links=txt_links, logger=logger)

        # Print the number of sentences and the first few sentences as a sample
        logger.info(f"Total number of sentences: {len(all_sentences)}")
        return txt_links, all_sentences
    else:
        logger.warning("Failed to find subcategory link.")


if __name__ == '__main__':
    subcategory = 'World War II'
    txt_links, all_sentences = get_text_data(subcategory=subcategory)
