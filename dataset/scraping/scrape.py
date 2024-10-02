import os
import requests
from bs4 import BeautifulSoup
import sys

output_folder = 'nepali_paragraphs'
scraped_urls_file = 'scraped_urls.txt'
urls_list = [
    'https://ne.wikipedia.org/wiki/नेपाल',
    'https://ne.wikipedia.org/wiki/%E0%A4%A8%E0%A5%87%E0%A4%AA%E0%A4%BE%E0%A4%B2%E0%A5%80_%E0%A4%AD%E0%A4%BE%E0%A4%B7%E0%A4%BE'
]

if len(sys.argv) == 2:
    url_to_scrape = sys.argv[1]
else:
    url_to_scrape = urls_list[1]

def get_next_file_number(folder_path):
    existing_files = [f for f in os.listdir(folder_path) if f.startswith('paragraph_') and f.endswith('.txt')]
    if not existing_files:
        return 1
    existing_numbers = [int(f.split('_')[1].split('.')[0]) for f in existing_files]
    return max(existing_numbers) + 1

def scrape_wikipedia_paragraphs(url, folder_path):
    # Fetch the content of the Wikipedia page
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all paragraphs
    paragraphs = soup.find_all('p')
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Get the next file number to start naming files
    next_file_number = get_next_file_number(folder_path)
    
    # Save each paragraph to a separate file
    for i, paragraph in enumerate(paragraphs):
        paragraph_text = paragraph.get_text().strip()
        if paragraph_text:  # Only save non-empty paragraphs
            file_name = f'paragraph_{next_file_number + i}.txt'
            with open(os.path.join(folder_path, file_name), 'w', encoding='utf-8') as file:
                file.write(paragraph_text)

def load_scraped_urls(scraped_urls_file):
    if os.path.exists(scraped_urls_file):
        with open(scraped_urls_file, 'r', encoding='utf-8') as file:
            scraped_urls = file.read().splitlines()
    else:
        scraped_urls = []
    return scraped_urls

def save_scraped_url(scraped_urls_file, url):
    with open(scraped_urls_file, 'a', encoding='utf-8') as file:
        file.write(url + '\n')

def main(url_to_scrape, output_folder, scraped_urls_file):
    scraped_urls = load_scraped_urls(scraped_urls_file)
    if url_to_scrape in scraped_urls:
        print(f'The URL "{url_to_scrape}" has already been scraped.')
        return
    scrape_wikipedia_paragraphs(url_to_scrape, output_folder)
    save_scraped_url(scraped_urls_file, url_to_scrape)
    print(f'Successfully scraped and saved content from "{url_to_scrape}".')

if __name__ == "__main__":
    main(url_to_scrape, output_folder, scraped_urls_file)

