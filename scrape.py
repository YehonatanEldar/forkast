from icrawler.builtin import BingImageCrawler

# Constants
TARGET_WORD = "fork"
HARD_NEGATIVES = ["spoon", "knife", "backscratcher", "ladle", "tong", "whisk"]
HARD_PORTION = 0.6
DATASET_SIZE = 100

def scrape_word_images(word: str, count: int, folder_path: str):
    """
    Scrapes online for images of the given word and downloads them to the given folder
    """
    filters = dict(license='commercial,modify') # for non-copyrighted content only
    crawler = BingImageCrawler(downloader_threads=4,  storage={'root_dir': folder_path})

    crawler.crawl(keyword=word, filters=filters, max_num=count)


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


