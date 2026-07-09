import requests
from duckduckgo_search import DDGS

# Constants
TARGET_WORD = "fork"
HARD_NEGATIVES = ["spoon", "knife", "backscratcher", "ladle", "tong", "whisk"]
HARD_PORTION = 0.6
DATASET_SIZE = 100
THREAD_COUNT = 2
CRAWL_DELAY = 2.0
DOWNLOAD_DELAY = 1.0

def get_image_urls(word: str, count: int) -> list[dict]:
    """
    Returns a list of urls to images of the given word
    """
    with DDGS() as ddgs:
        return list(ddgs.images(word, max_results=int(count)))


def download_images(results: list[str], count: int, output_dir: str):
    """
    Downloads the given url list to the given folder path
    """
    download_count = 0

    for item in results:
        if download_count >= count:
            break

        url = item['image']

        try: 
            response = requests.get(url)
            if response.status_code == 200:
                # Copy the contents to a file
                with open(f"{output_dir}/{download_count}.jpg", "wb") as f:
                    f.write(response.content)

                download_count += 1
                print(f"\033[KDownloaded: {download_count}/{count}", end="\r", flush=True)


        except Exception as e:
            print(e)


def main():
    # --- Scraping ---
    results = get_image_urls(TARGET_WORD, 10)
    download_images(results, 10, 'images')
    

if __name__ == '__main__':
    main()


