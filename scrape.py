from icrawler.builtin import BingImageCrawler

# Constants
TARGET_WORD = "fork"
HARD_NEGATIVES = ["spoon", "knife", "backscratcher", "ladle", "tong", "whisk"]
HARD_PORTION = 0.6
DATASET_SIZE = 100
THREAD_COUNT = 2
CRAWL_DELAY = 2.0
DOWNLOAD_DELAY = 1.0

def scrape_word_images(word: str, count: int, folder_path: str):
    """
    Scrapes online for images of the given word and downloads them to the given folder
    """
    custom_headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
    filters = dict(license='commercial,modify') # for non-copyrighted content only
    crawler = BingImageCrawler(downloader_threads=THREAD_COUNT,  storage={'root_dir': folder_path})
    crawler.session.headers.update(custom_headers)
    crawler.parser.sleep_time = CRAWL_DELAY
    crawler.downloader.sleep_time = DOWNLOAD_DELAY

    crawler.crawl(keyword=word, filters=filters, max_num=int(count))


def main():
    # --- Scraping ---
    # Target word
    scrape_word_images(TARGET_WORD, DATASET_SIZE//2, 'images/positive')

    # Hard negatives
    hard_count = (DATASET_SIZE//2) * HARD_PORTION
    hard_per_word = hard_count / len(HARD_NEGATIVES)
    for word in HARD_NEGATIVES:
        scrape_word_images(word, hard_per_word, 'images/negative')

    # Easy negatives



    

if __name__ == '__main__':
    main()


